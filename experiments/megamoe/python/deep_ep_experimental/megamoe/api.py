# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Thin inference-only API; all algorithm selection remains in the native layer."""

from __future__ import annotations

from functools import lru_cache
import importlib
import math
import numbers
import threading
from typing import TYPE_CHECKING

from deep_ep._runtime import Runtime, integer, warn_unclosed
from deep_ep._native import npu_api, require_native
from deep_ep._store import StoreGroup
from .validation import check_routing, collective_check

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributed import ProcessGroup

    WeightPair = tuple[Tensor, Tensor]


@lru_cache(maxsize=1)
def _extension():
    core = require_native("MegaMoE")
    if not hasattr(core, "create_megamoe_runtime"):
        raise RuntimeError("Install a wheel built with DEEPEP_BUILD_MEGAMOE=ON")
    try:
        extension = importlib.import_module(__package__ + "._C")
    except ImportError as error:
        raise ImportError("Cannot load MegaMoE Beta: check matching Torch, torch_npu, CANN and SHMEM") from error
    if getattr(extension, "api_protocol", None) != 13:
        raise RuntimeError("MegaMoE Python/native versions differ; rebuild and reinstall the complete Beta wheel")
    return extension


class _MegaRuntime(Runtime):
    _megamoe = True


def _pair(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be (quantized_data, E8M0_scales)")
    return value


def _check_fp4_weight_format(weight, name):
    import torch_npu

    actual = torch_npu.get_npu_format(weight)
    # 请求 NZ(29) 不等于返回码必为29：torch_npu 将 FP4 C0=32 映射成 uint8 C0=16(50)。
    # 不能放行所有 NZ 子格式，uint8 C0=32(51) 并非本路径的打包布局。
    if actual not in (29, 50):
        raise ValueError(
            f"{name}: packed FP4 weights require NZ format 29 or 50 (NZ_C0_16), "
            f"got format={actual}, dtype={weight.dtype}, shape={tuple(weight.shape)}; "
            "call transform_weights_for_mega_moe on raw ND weights once"
        )


def _check_shared_weight(weight, name):
    import torch
    import torch_npu

    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{name}: shared weights must be FP8 E4M3, got {weight.dtype}; packed FP4 is not accepted")
    actual = torch_npu.get_npu_format(weight)
    if actual not in (29, 51):
        raise ValueError(
            f"{name}: FP8 weights require NZ format 29 or 51 (NZ_C0_32), "
            f"got format={actual}, dtype={weight.dtype}, shape={tuple(weight.shape)}; "
            "call transform_weights_for_mega_moe(..., shared=True) on raw ND weights once"
        )


class SymmBuffer:
    """Collective resource owner. All ranks construct/call/destroy in the same order.

    num_experts is global; intermediate_hidden is the post-SwiGLU width.
    This Beta supports nonempty, same-BS calls and one in-flight operation per buffer.
    """

    def __init__(
        self,
        group: ProcessGroup | StoreGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        activation: str = "swiglu",
        *,
        ranks_per_node: int = 0,
        timeout: int = 120,
        route_weight_before_quant: bool = False,
        variable_token_counts: bool = False,
    ):
        import torch
        import torch.distributed as dist

        if not isinstance(group, (StoreGroup, dist.ProcessGroup)):
            raise TypeError("Pass a StoreGroup or an initialized ProcessGroup explicitly")
        size = group.size() if isinstance(group, StoreGroup) else dist.get_world_size(group)
        error = None
        try:
            integer("num_experts", num_experts, 1, 16384)
            integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 65536)
            integer("num_topk", num_topk, 1, min(32, num_experts))
            integer("hidden", hidden, 1024, 8192)
            integer("intermediate_hidden", intermediate_hidden, 256, 4096)
            integer("num_shared_experts", num_shared_experts, 0, 8)
            integer("world_size", size, 2, 128)
            integer("ranks_per_node", ranks_per_node, 0, size)
            integer("timeout", timeout, 1, 300)
            if not isinstance(route_weight_before_quant, bool):
                raise TypeError("route_weight_before_quant must be bool")
            if not isinstance(variable_token_counts,bool):
                raise TypeError("variable_token_counts must be bool")
            if variable_token_counts and mma_type!='fp8xfp4':
                raise ValueError('variable token counts currently require A8W4')
            if route_weight_before_quant and (mma_type != "fp8xfp4" or num_topk > 8):
                raise ValueError("pre-quant routing requires A8W4 and topK<=8")
            if hidden % 64 or intermediate_hidden % 128:
                raise ValueError("hidden must be divisible by 64; intermediate_hidden by 128")
            if activation != "swiglu" or mma_type not in {"fp8xfp4", "fp8xfp8"}:
                raise ValueError("Only SwiGLU with FP8xFP4/FP8xFP8 is implemented; no BF16xBF16")
            if mma_type != "fp8xfp4" and num_shared_experts:
                raise ValueError("Shared FP8 experts currently require the routed FP8xFP4 kernel")
            if num_experts % size or num_experts // size > 128:
                raise ValueError("Experts must divide EP, with at most 128 local experts")
            if ranks_per_node and size % ranks_per_node:
                raise ValueError("ranks_per_node must divide EP (contiguous node-major ranks)")
        except (TypeError, ValueError) as failure:
            error = str(failure)
        # 在进入 SHMEM 之前共同校验布局，不能各 rank 各自选择通信偏移。
        config = (
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts,
            mma_type,
            activation,
            ranks_per_node,
            timeout,
            route_weight_before_quant,
            variable_token_counts,
        )
        collective_check(group, "configuration", error, config)
        ext = _extension()
        capacity = int(ext.buffer_size(size, num_experts // size, num_max_tokens_per_rank, num_topk, hidden))
        self._lock = threading.Lock()
        self._device = torch.device(f"npu:{npu_api().current_device()}")
        self._plan = self._plan_bs = None
        self.group, self.num_experts = group, num_experts
        self.num_max_tokens_per_rank, self.num_topk = num_max_tokens_per_rank, num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts, self.mma_type = num_shared_experts, mma_type
        self.ranks_per_node = ranks_per_node
        self.route_weight_before_quant = route_weight_before_quant
        self.variable_token_counts = variable_token_counts
        self._graph_plans = {}
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        error = None
        try:
            self.allocate_inputs()
        except Exception as failure:
            error = str(failure)
        # 输入分配失败先通知所有 rank，再进入 SHMEM，避免其他 rank 已在建链等待。
        collective_check(group, "input allocation", error)
        self._runtime = _MegaRuntime(group, capacity, timeout)

    def allocate_inputs(self):
        """Legacy idempotent helper. Inputs are already allocated by the constructor."""
        import torch

        if hasattr(self, "_runtime"):
            self._runtime.ensure_alive()
        if self.x is not None:
            return self
        self.x = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden), dtype=torch.float8_e4m3fn, device=self._device
        )
        self.x_sf = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden // 32), dtype=torch.float8_e8m0fnu, device=self._device
        )
        self.topk_idx = torch.empty(
            (self.num_max_tokens_per_rank, self.num_topk), dtype=torch.int32, device=self._device
        )
        self.topk_weights = torch.empty_like(self.topk_idx, dtype=torch.float32)
        return self

    def _prepare(self, bs):
        integer("BS", bs, 1, self.num_max_tokens_per_rank)
        npu=npu_api()
        capturing=bool(getattr(npu,'is_current_stream_capturing',lambda:False)())
        if self._plan_bs == bs:
            if capturing:self._graph_plans[bs]=self._plan
            return
        # 捕获图保留各自计划，但scratch优先复用，避免每个BS都常驻一份最坏容量。
        workspace_hint=getattr(self._plan,'workspace_tensor',None)
        self._plan = self._plan_bs = None
        if bs in self._graph_plans:
            self._plan,self._plan_bs=self._graph_plans[bs],bs
            return
        runtime = self._runtime
        address = require_native("MegaMoE").megamoe_workspace_address(runtime.native)
        ext = _extension()
        self._plan = ext.Plan(
            self.x[:bs],
            self.intermediate_hidden,
            self.num_topk,
            self.num_experts // self.group.size(),
            self.group.size(),
            self.num_max_tokens_per_rank,
            self.mma_type == "fp8xfp4",
            self.num_shared_experts,
            self.ranks_per_node,
            ext.ffts_config(),
            address,
            runtime.native.capacity,
            -1,
            self.route_weight_before_quant,
            self.variable_token_counts,
            workspace_hint,
        )
        self._plan_bs = bs
        if capturing:self._graph_plans[bs]=self._plan

    def prepare(self, bs: int) -> SymmBuffer:
        """Collectively prepare/reuse one BS before timing; all ranks must call together."""
        error = None
        with self._lock:
            try:
                self._runtime.ensure_alive()
                integer("BS", bs, 1, self.num_max_tokens_per_rank)
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, "prepare", error, None if self.variable_token_counts else bs)
            try:
                runtime = self._runtime
                runtime.stream.wait_stream(npu_api().current_stream(runtime.device))
                with npu_api().stream(runtime.stream):
                    self._prepare(bs)
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, f"prepare BS={bs}", error)
        return self

    def enable_compute_replay(self) -> None:
        """Diagnostic only: freeze the last forward's dispatched inputs/counts.

        Later calls run local GMM1/SwiGLU/GMM2 but do not produce a new MoE output.
        Preparing a different BS creates a fresh, non-replay plan.
        """
        with self._lock:
            self._runtime.ensure_alive()
            if self._plan is None:
                raise RuntimeError("Compute replay requires prepare and one priming forward")
            if not hasattr(self._plan, "enable_compute_replay"):
                raise RuntimeError("Installed wheel lacks compute replay; rebuild this branch (omit --skip-build)")
            self._plan.enable_compute_replay()

    def retain_prepared_plan(self) -> None:
        """Keep the current plan alive for a captured graph; scratch can still be shared."""
        with self._lock:
            self._runtime.ensure_alive()
            if self._plan is None:
                raise RuntimeError('retain requires one prepared plan')
            self._graph_plans[self._plan_bs]=self._plan

    @property
    def workspace_bytes(self) -> int:
        """Prepared local scratch only; excludes symmetric heap, inputs and weights."""
        return self._plan.workspace_bytes if self._plan is not None else 0

    @property
    def symmetric_buffer_bytes(self) -> int:
        return self._runtime.native.capacity

    @property
    def timer_enabled(self) -> bool:
        return bool(_extension().timer_enabled and self.mma_type == "fp8xfp4")

    def allocate_timer(self) -> Tensor:
        """Fresh zeroed device trace; pass only to a separate diagnostic invocation."""
        import torch

        self._runtime.ensure_alive()
        if not self.timer_enabled:
            raise RuntimeError("W4A8 timer requires DEEPEP_MEGAMOE_TIMER=ON")
        return torch.zeros(_extension().timer_numel, dtype=torch.int64, device=self._device)

    def validate_inputs(self, bs: int, *, x: Tensor | None = None) -> None:
        """Collective, synchronizing routing scan. Run after filling inputs, outside timing."""
        error = None
        try:
            import torch

            self._runtime.ensure_alive()
            integer("BS", bs, 1, self.num_max_tokens_per_rank)
            for name, tensor, dtype, shape in (
                ("topk_idx", self.topk_idx, torch.int32, (self.num_max_tokens_per_rank, self.num_topk)),
                ("topk_weights", self.topk_weights, torch.float32, (self.num_max_tokens_per_rank, self.num_topk)),
            ):
                if tensor.device != self._device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    raise ValueError(f"{name} must retain its buffer dtype/device/shape")
            if x is None:
                for name, tensor, dtype, shape in (
                    ("x", self.x, torch.float8_e4m3fn, (self.num_max_tokens_per_rank, self.hidden)),
                    ("x_sf", self.x_sf, torch.float8_e8m0fnu, (self.num_max_tokens_per_rank, self.hidden // 32)),
                ):
                    if (
                        tensor.device != self._device
                        or tensor.dtype != dtype
                        or tuple(tensor.shape) != shape
                        or not tensor.is_contiguous()
                    ):
                        raise ValueError(f"{name} must retain its buffer dtype/device/contiguous shape")
            if x is not None and (
                x.dtype != torch.bfloat16 or x.device != self._device or tuple(x.shape) != (bs, self.hidden)
            ):
                raise ValueError("x override must be BF16 [BS,H] on the buffer device")
            check_routing(
                self.topk_idx[:bs].cpu().tolist(),
                self.topk_weights[:bs].cpu().tolist(),
                self.num_experts,
                self.num_topk,
                allow_invalid=self.variable_token_counts,
            )
        except Exception as failure:
            error = str(failure)
        collective_check(self.group, "input preflight", error, (None if self.variable_token_counts else bs, x is not None))

    def destroy(self):
        with self._lock:
            self._runtime.close()
            self._plan = self._plan_bs = None
            self._graph_plans.clear()
            self.x = self.x_sf = self.topk_idx = self.topk_weights = None

    def __del__(self):
        warn_unclosed(self)


def _interleave_gate_up_128(tensor):
    """Permute the N axis of raw [E, 2I, ...] storage without numeric conversion."""
    experts, columns, *tail = tensor.shape
    return (
        tensor.reshape((experts, 2, columns // 256, 128, *tail))
        .transpose(1, 2)
        .contiguous()
        .reshape((experts, columns, *tail))
    )


def _pack_weight_scale_blocks(scale):
    """N256内按SF pair-major排列；只改变位模式位置，在计时区外完成。"""
    import torch

    e,n,pairs,two = scale.shape
    if n % 256 or two != 2:
        raise ValueError("FP4 blocked weight scales require N divisible by 256 and two UE8M0 per pair")
    shape = scale.shape
    raw = scale.view(torch.uint8).reshape((e,n // 256,256,pairs,2))
    return raw.transpose(2,3).contiguous().reshape(shape).view(torch.float8_e8m0fnu)


class _TilePackedWeightPair(tuple):
    """路由FP4 tile-pack v1；不得与旧版原生库混装。"""


class _RoutedWeightPair(_TilePackedWeightPair):
    """Tuple-compatible immutable prepared weights; both layouts are built off the hot path."""

    def __new__(cls, fused, planar):
        value = super().__new__(cls, fused)
        value._planar = planar
        value._versions = tuple(t._version for t in (*fused, *planar))
        return value

    def select(self, fused):
        if tuple(t._version for t in (*self, *self._planar)) != self._versions:
            raise ValueError("Prepared GMM1 weights/scales were modified in-place; transform raw ND weights again")
        return tuple(self) if fused else self._planar


def _tilepack_fp4_for_nz_cast(weight):
    """Inverse NZ input for physical [N256,K512,N128,K32,N128,16B], including K tails."""
    import torch
    e,n,half_k=weight.shape
    if n % 256 or half_k % 32:
        raise ValueError("FP4 tile-pack requires N%256=0 and K%64=0")
    blocks=[]
    source=weight.reshape((e,n//256,2,128,half_k))
    for begin in range(0,half_k,256):
        width=min(256,half_k-begin)
        block=source[...,begin:begin+width].reshape((e,n//256,2,128,width//16,16))
        blocks.append(block.permute(0,1,2,4,3,5).contiguous().reshape((e,n//256,256*width)))
    physical=torch.cat(blocks,dim=2)
    # format_cast随后转为[K32,N,16B]；先逆排，避免把目标physical再错误转置一次。
    return physical.reshape((e,half_k//16,n,16)).permute(0,2,1,3).contiguous().reshape(weight.shape)


def transform_weights_for_mega_moe(
    l1_weights: WeightPair, l2_weights: WeightPair, activation: str = "swiglu", *, shared: bool = False,
    keep_planar: bool = True
) -> tuple[WeightPair, WeightPair]:
    """Convert raw Ascend packed weights to NZ once, without quantizing again.

    Prepared weights from earlier branches must be regenerated: FP4 GMM2 and paired
    GMM1 now use N256-block SF pair-major storage, without changing external tensor shapes.
    Raw inputs use [gate | up]. Routed FP4 GMM1 uses [gate128, up128] for every
    supported BS (1..65536), with no layout switch above 4096. The legacy planar
    copy is still prepared for tuple compatibility, not selected by routed launches.
    Keep the returned l1 pair intact and immutable. Pass shared=True for
    shared experts: raw FP8 E4M3 [S,N,K] plus E8M0 [S,N,K/64,2] scales,
    both projections stay planar and are converted to ordinary NZ without quantization.
    Shared packed FP4 is no longer accepted. Routed FP4 still uses tile-pack v1.
    This is not the CUDA UTCCP transform. Old prepared FP4 routed weights must
    be regenerated from raw ND inputs; do not transform prepared NZ twice.
    """
    import torch
    import torch_npu

    if activation != "swiglu":
        raise NotImplementedError("Only SwiGLU is supported")
    transformed = []
    for name, value in (("l1_weights", l1_weights), ("l2_weights", l2_weights)):
        weight, scale = _pair(value, name)
        if weight.device.type != "npu" or scale.device != weight.device:
            raise ValueError("Move weights and scales to the same NPU before preparing")
        if weight.ndim != 3:
            raise ValueError(f"{name}: weights must be [E,N,K] or packed [E,N,K/2]")
        if weight.dtype in {torch.int8, torch.uint8} and torch_npu.get_npu_format(weight) in (29, 50):
            raise ValueError(f"{name}: prepare from raw ND weights, not already converted NZ weights")
        if weight.dtype == torch.int8:
            weight = weight.view(torch.uint8)  # 重解释 FP4 的位模式，不能做数值类型转换。
        if weight.dtype not in {torch.uint8, torch.float8_e4m3fn}:
            raise TypeError("Weights must be packed FP4 int8/uint8 or FP8 E4M3")
        if shared and weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name}: shared weights must be FP8 E4M3; packed FP4 is not accepted")
        if shared and torch_npu.get_npu_format(weight) != 2:
            raise ValueError(f"{name}: prepare shared FP8 from raw ND weights, not prepared NZ weights")
        k = weight.shape[-1] * (2 if weight.dtype == torch.uint8 else 1)
        expected = (*weight.shape[:2], k // 64, 2)
        if k % 64:
            raise ValueError(f"{name}: K must be divisible by 64")
        if scale.dtype == torch.int32 and tuple(scale.shape) == (*weight.shape[:2], k // 128) and k % 128 == 0:
            # 仅接受逻辑行优先、尚未做 CUDA UTCCP 重排的 packed E8M0。
            scale = scale.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).reshape(expected)
        if scale.dtype != torch.float8_e8m0fnu or tuple(scale.shape) != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected} or raw packed int32 [E,N,K/128]")
        scale = scale.contiguous()
        if shared:
            weight = torch_npu.npu_format_cast(weight, 29)
            _check_shared_weight(weight, name)
            transformed.append((weight, scale))
            continue
        if weight.dtype == torch.uint8:
            if name == "l1_weights" and not shared:
                if weight.shape[1] % 256:
                    raise ValueError("Routed FP4 GMM1 requires gate/up widths divisible by 128")
                # 两种布局一次准备；独立存储并启用版本计数，禁止原地修改后两份权重失配。
                with torch.inference_mode(False):
                    if keep_planar:
                        planar_weight = torch_npu.npu_format_cast(
                            _tilepack_fp4_for_nz_cast(weight), 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                        )
                        planar_scale = scale.view(torch.uint8).clone().view(torch.float8_e8m0fnu)
                    weight = _interleave_gate_up_128(weight)
                    scale = _pack_weight_scale_blocks(
                        _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu))
                    weight = torch_npu.npu_format_cast(
                        _tilepack_fp4_for_nz_cast(weight), 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                    )
                    _check_fp4_weight_format(weight, name)
                    if keep_planar:
                        _check_fp4_weight_format(planar_weight, name)
                        transformed.append(_RoutedWeightPair((weight, scale), (planar_weight, planar_scale)))
                    else:
                        transformed.append(_TilePackedWeightPair((weight, scale)))
                continue
            # uint8 只是两个 FP4 的存储载体；FP8 布局参数不能代替输入的 FP4 类型声明。
            weight = torch_npu.npu_format_cast(
                weight if shared else _tilepack_fp4_for_nz_cast(weight),
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            _check_fp4_weight_format(weight, name)
            if name == "l2_weights":
                scale = _pack_weight_scale_blocks(scale)
        transformed.append(_TilePackedWeightPair((weight, scale)) if weight.dtype == torch.uint8 and not shared else (weight, scale))
    return tuple(transformed)


def mega_moe(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    timer=None,
):
    """Explicit-input convenience API; prefer fp8_fp4_mega_moe for DeepGEMM-style use.

    x: BF16 [B,H], or (FP8 E4M3 [B,H], E8M0 [B,H/32]).
    FP8 is copied into the original communication record, never dequantized/requantized.
    Routed weights remain FP4; optional shared experts use FP8 E4M3 weights.
    No backward or graph guarantee yet.
    """
    return _run(
        x,
        topk_idx,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=out,
        timer=timer,
    )


def _run(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    timer=None,
    cumulative_stats=None,
    activation_clamp=None,
):
    import torch

    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    data, scales = _pair(x, "x") if isinstance(x, tuple) else (x, None)
    w1, s1 = _pair(l1_weights, "l1_weights")
    w2, s2 = _pair(l2_weights, "l2_weights")
    shared = (None, None, None, None)
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError("Supply both shared weight pairs, or neither")
    if shared_l1_weights is not None:
        sw1, ss1 = _pair(shared_l1_weights, "shared_l1_weights")
        sw2, ss2 = _pair(shared_l2_weights, "shared_l2_weights")
        shared = (sw1, sw2, ss1, ss2)
    b = sym_buffer
    if data.ndim != 2 or not 0 < data.shape[0] <= b.num_max_tokens_per_rank or data.shape[1] != b.hidden:
        raise ValueError("x must be nonempty [BS,H] within the buffer capacity")
    if data.device != b._device:
        raise ValueError("x must be on the buffer's creating device")
    expected = (b.num_experts // b.group.size(), b.intermediate_hidden * 2)
    if tuple(w1.shape[:2]) != expected or tuple(topk_idx.shape) != (data.shape[0], b.num_topk):
        raise ValueError("Weight/routing shapes do not match the buffer configuration")
    if (shared[0].shape[0] if shared[0] is not None else 0) != b.num_shared_experts:
        raise ValueError("Shared expert count does not match the buffer configuration")
    expected_dtype = torch.uint8 if b.mma_type == "fp8xfp4" else torch.float8_e4m3fn
    if w1.dtype != expected_dtype:
        raise TypeError("Weight dtype differs from buffer mma_type")
    prepared_inputs = ()
    if w1.dtype == torch.uint8:
        if not all(isinstance(pair, _TilePackedWeightPair) for pair in (l1_weights,l2_weights)):
            raise ValueError("FP4 tile-pack v1 requires fresh transform_weights_for_mega_moe output; keep its returned l1 pair intact")
        fused = data.shape[0] <= _extension().paired_gate_up_max_tokens
        if isinstance(l1_weights, _RoutedWeightPair):
            prepared_inputs = (*l1_weights, *l1_weights._planar)
            w1, s1 = l1_weights.select(fused)
        elif not fused:
            raise ValueError(
                "Prefill requires both prepared GMM1 layouts; call transform_weights_for_mega_moe "
                "on raw ND weights and keep its returned l1 pair intact"
            )
    if any(t is not None and t.requires_grad for t in (data, scales, topk_weights, w1, s1, w2, s2, out, *shared)):
        raise NotImplementedError("MegaMoE Beta is inference-only")
    if out is not None and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared, *prepared_inputs)
        if tensor is not None
    ):
        raise ValueError("out must not overlap any input or timer storage")
    if topk_idx.dtype != torch.int32:
        raise TypeError("topk_idx must be int32; convert explicitly before timing")
    if w1.dtype == torch.uint8:
        for name, weight in (
            ("l1_weights", w1),
            ("l2_weights", w2),
        ):
            if weight is not None:
                _check_fp4_weight_format(weight, name)
    for name, weight, shape in (
        ("shared_l1_weights", shared[0], (b.num_shared_experts, 2 * b.intermediate_hidden, b.hidden)),
        ("shared_l2_weights", shared[1], (b.num_shared_experts, b.hidden, b.intermediate_hidden)),
    ):
        if weight is not None:
            _check_shared_weight(weight, name)
            if tuple(weight.shape) != shape:
                raise ValueError(f"{name}: expected FP8 shape {shape}, got {tuple(weight.shape)}")
    if cumulative_stats is not None:
        stats = cumulative_stats
        if (
            stats.dtype != torch.int32
            or tuple(stats.shape) != (b.num_experts // b.group.size(),)
            or stats.device != b._device
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the buffer device")
        if any(
            stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, out, timer, *shared, *prepared_inputs)
            if t is not None
        ):
            raise ValueError("Cumulative stats must not alias inputs, output or timer")
    runtime = b._runtime
    with b._lock:
        runtime.ensure_alive()
        caller = npu_api().current_stream(runtime.device)
        capturing=bool(getattr(npu_api(),'is_current_stream_capturing',lambda:False)())
        runtime.stream.wait_stream(caller)
        # 同一 owner 的调用串行进入专属流，禁止不同流覆盖仍在使用的对称空间。
        with npu_api().stream(runtime.stream):
            b._prepare(data.shape[0])
            result = b._plan.forward(data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared,
                                     activation_clamp, out)
            if cumulative_stats is not None:
                cumulative_stats.add_(result[1])
                cumulative_stats.record_stream(runtime.stream)
        if capturing:b._graph_plans[data.shape[0]]=b._plan
        caller.wait_stream(runtime.stream)
        for tensor in result:
            tensor.record_stream(caller)
        return result


def fp8_fp4_mega_moe(
    y: Tensor,
    l1_weights: WeightPair,
    l2_weights: WeightPair,
    sym_buffer: SymmBuffer,
    shared_l1_weights: WeightPair | None = None,
    shared_l2_weights: WeightPair | None = None,
    cumulative_local_expert_recv_stats: Tensor | None = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
    *,
    x: Tensor | None = None,
    timer: Tensor | None = None,
) -> None:
    """Write BF16 y in place, return None; accumulate stats if supplied.

    Positional parameters follow DeepGEMM. Ascend-only keyword extensions:
    x is optional BF16 [BS,H] (quantized internally, NOT BF16 GEMM);
    timer is a fresh buffer.allocate_timer() for a separate diagnostic call.
    Routed weights are applied after GMM2; see README numerical compatibility.
    """
    if recipe != (1, 1, 32) or activation != "swiglu" or fast_math is not True:
        raise NotImplementedError("Beta supports recipe=(1,1,32), SwiGLU, default math only")
    if activation_clamp is not None:
        if (isinstance(activation_clamp, bool) or not isinstance(activation_clamp, numbers.Real)
                or not math.isfinite(activation_clamp) or not 0 < activation_clamp <= 3.4028234663852886e38):
            raise ValueError("activation_clamp must be None or finite positive FP32")
        activation_clamp = float(activation_clamp)
    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    import torch

    sym_buffer._runtime.ensure_alive()
    if y.ndim != 2 or y.dtype != torch.bfloat16:
        raise ValueError("y must be BF16 [BS,H]")
    bs = y.shape[0]
    if x is not None and (x.dtype != torch.bfloat16 or tuple(x.shape) != (bs, sym_buffer.hidden)):
        raise ValueError("x override must be BF16 [BS,H]; omit x for buffered FP8 input")
    _run(
        (sym_buffer.x[:bs], sym_buffer.x_sf[:bs]) if x is None else x,
        sym_buffer.topk_idx[:bs],
        sym_buffer.topk_weights[:bs],
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=y,
        timer=timer,
        cumulative_stats=cumulative_local_expert_recv_stats,
        activation_clamp=activation_clamp,
    )
