"""Experimental fixed-CV, PUT-only URMA MegaMoE API.

Route construction is deliberately visible: ``forward`` prepares routes on
every call. ``forward_prepared`` is a separate, explicitly cached-route path.
Neither path uses a balanced-route template.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib
import threading

from deep_ep._native import npu_api, require_native
from deep_ep._runtime import Runtime, integer, warn_unclosed
from deep_ep._store import StoreGroup
from .urma_routing import DeviceRoutes, prepare_routes
from .validation import collective_check


@lru_cache(maxsize=1)
def _extension():
    core = require_native("URMA MegaMoE")
    if not hasattr(core, "create_megamoe_urma_runtime"):
        raise RuntimeError("Install a wheel built with the MegaMoE URMA runtime")
    extension = importlib.import_module(__package__ + "._C")
    if getattr(extension, "urma_api_protocol", None) != 1:
        raise RuntimeError("URMA Python/native versions differ; rebuild and reinstall the complete wheel")
    if getattr(extension, "urma_tail_schedule", None) != "combine-ready-scan-reduce-world-v1":
        raise RuntimeError("URMA tailCover needs its matching native wheel; rebuild and reinstall the complete wheel")
    if getattr(extension, "urma_compute_schedule", None) != "expert-m256-gateup128-lookahead-v2":
        raise RuntimeError(
            "URMA M-group lookahead needs its matching native wheel; rebuild and reinstall the complete wheel"
        )
    return extension


def _trace_extension():
    """Check the optional trace ABI only when a caller requests tracing."""
    extension = _extension()
    expected = {
        "urma_trace_schema": 1,
        "urma_trace_num_engines": 84,
        "urma_trace_record_elements": 16,
        "urma_trace_header_elements": 16,
        "urma_trace_max_capacity": 65536,
    }
    for name, value in expected.items():
        if getattr(extension, name, None) != value:
            raise RuntimeError(
                f"URMA trace Python/native versions differ ({name}); rebuild and reinstall the complete wheel"
            )
    if not getattr(extension, "urma_timer_enabled", False):
        raise RuntimeError("URMA trace requires a wheel built with DEEPEP_MEGAMOE_TIMER=ON")
    return extension


class _UrmaRuntime(Runtime):
    _megamoe = True
    _megamoe_urma = True

    def __init__(self, group, capacity, timeout=120, *, allow_multi_host=False):
        if type(allow_multi_host) is not bool:
            raise TypeError("allow_multi_host must be bool")
        self._megamoe_urma_multi_host = allow_multi_host
        super().__init__(group, capacity, timeout)


@dataclass(frozen=True)
class PreparedUrmaRoutes:
    """One immutable route plan; the tensor contents must also remain unchanged."""

    owner: object
    native: object
    routing: DeviceRoutes
    bs: int
    received_rows: int
    versions: tuple


class UrmaBuffer:
    """EP4/EP8/EP16 owner; multiple hosts require explicit opt-in on every rank."""

    def __init__(
        self,
        group,
        *,
        hidden=3584,
        intermediate_hidden=1280,
        num_experts=384,
        num_topk=8,
        num_shared_experts=0,
        num_max_tokens_per_rank=8192,
        cv_rows=256,
        gmm1_ahead_mgroups=12,
        max_recv_rows=None,
        timeout=120,
        allow_multi_host=False,
    ):
        import torch

        if not isinstance(group, StoreGroup):
            raise TypeError("URMA experimental API currently requires an explicit StoreGroup")
        error = None
        if max_recv_rows is None:
            max_recv_rows = 2 * num_max_tokens_per_rank * num_topk
        try:
            if group.size() not in (4, 8, 16):
                raise ValueError("URMA MegaMoE requires EP4/EP8/EP16")
            if type(allow_multi_host) is not bool:
                raise TypeError("allow_multi_host must be bool")
            if (hidden, intermediate_hidden, num_experts, num_topk, num_shared_experts) != (3584, 1280, 384, 8, 0):
                raise ValueError("First URMA profile supports H3584/I1280/E384/topk8/S0 only")
            integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 8192)
            integer("cv_rows", cv_rows, 128, 4096)
            integer("gmm1_ahead_mgroups", gmm1_ahead_mgroups, 0, 65536)
            if cv_rows % 128:
                raise ValueError("cv_rows must be a multiple of 128")
            integer("max_recv_rows", max_recv_rows, 1, group.size() * num_max_tokens_per_rank * num_topk)
        except (TypeError, ValueError) as failure:
            error = str(failure)
        collective_check(
            group,
            "URMA configuration",
            error,
            (
                hidden,
                intermediate_hidden,
                num_experts,
                num_topk,
                num_shared_experts,
                num_max_tokens_per_rank,
                cv_rows,
                gmm1_ahead_mgroups,
                max_recv_rows,
                timeout,
                allow_multi_host,
                "combine-ready-scan-reduce-world-v1",
                "expert-m256-gateup128-lookahead-v2",
            ),
        )
        self.group = group
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank, self.max_recv_rows = num_max_tokens_per_rank, max_recv_rows
        self.cv_rows = cv_rows
        self.gmm1_ahead_mgroups = gmm1_ahead_mgroups
        self.allow_multi_host = allow_multi_host
        self._device = torch.device(f"npu:{npu_api().current_device()}")
        self._lock = threading.RLock()
        self._epoch = 0
        self._failure = None
        self._last_prepared = None
        capacity, error = None, None
        try:
            extension = _extension()
            capacity = int(
                extension.urma_buffer_size(
                    group.size(), num_max_tokens_per_rank, num_topk, hidden, max_recv_rows, cv_rows
                )
            )
            self._example = torch.empty(
                (num_max_tokens_per_rank, hidden), device=self._device, dtype=torch.float8_e4m3fn
            )
        except Exception as failure:
            error = str(failure)
        # No rank may enter collective SHMEM initialization after another rank
        # rejected the wheel/layout or failed its ordinary-device allocation.
        collective_check(group, "URMA pre-runtime allocation", error)
        self._runtime = _UrmaRuntime(group, capacity, timeout, allow_multi_host=allow_multi_host)

    @property
    def symmetric_buffer_bytes(self):
        return self._runtime.native.capacity

    @property
    def timer_enabled(self):
        return bool(getattr(_extension(), "urma_timer_enabled", False))

    def allocate_timer(self):
        import torch

        with self._operation("timer allocation"):
            if not self.timer_enabled:
                raise RuntimeError("URMA timer requires a wheel built with its timer support")
            return torch.zeros(int(_extension().urma_timer_numel), dtype=torch.int64, device=self._device)

    def allocate_trace(self, capacity=4096):
        """Allocate bounded per-engine event slices for one diagnostic launch."""
        import torch

        with self._operation("trace allocation"):
            extension = _trace_extension()
            integer("trace capacity", capacity, 1, extension.urma_trace_max_capacity)
            numel = extension.urma_trace_num_engines * (
                extension.urma_trace_header_elements + capacity * extension.urma_trace_record_elements
            )
            return torch.zeros(numel, dtype=torch.int64, device=self._device)

    @contextmanager
    def _operation(self, stage):
        """Fail the collective job, rather than retrying with divergent epochs.

        Store abort wakes peers waiting in host coordination. It cannot cancel
        a kernel already executing; the job launcher must terminate failed jobs.
        Successful cached forwards do not add a host collective to the hot path.
        """
        with self._lock:
            try:
                self._runtime.ensure_alive()
                if self._failure is not None:
                    raise RuntimeError(f"URMA buffer is unusable after a failed operation: {self._failure}")
                yield
            except Exception as error:
                if self._failure is None:
                    self._failure = f"{stage}: {type(error).__name__}: {error}"
                    self.group.abort(self._failure)
                raise

    def _enqueue_forward(self, prepared, operands):
        """Serialize every native launch on the owner's device stream.

        The host RLock alone cannot serialize work on different NPU streams.
        These two event edges also order metadata/input writes and output reads.
        """
        runtime = self._runtime
        npu = npu_api()
        caller = npu.current_stream(runtime.device)
        runtime.stream.wait_stream(caller)
        with npu.stream(runtime.stream):
            result = prepared.native.forward(*operands)
        caller.wait_stream(runtime.stream)
        result.record_stream(caller)
        return result

    def _bind_reduce_dependencies(self, plan, reduce_index):
        """Keep the asynchronous route gather ordered before every forward.

        A prepared plan may be used from a different caller stream. Its hidden
        dependency table must therefore be produced on the owner stream too.
        """
        runtime = self._runtime
        npu = npu_api()
        runtime.stream.wait_stream(npu.current_stream(runtime.device))
        with npu.stream(runtime.stream):
            plan.bind_reduce_dependencies(reduce_index)

    def prepare_routes(self, topk_idx):
        """NPU sort/count/inverse, CPU count exchange, and native descriptor build.

        This is a collective, synchronizing preparation step. FULL_E2E includes
        it; CACHED_FORWARD intentionally does not. Counts are the only routing
        data exchanged through the host Store.
        """
        import torch

        with self._operation("route preparation"):
            error = None
            try:
                if topk_idx.device != self._device or topk_idx.dtype not in (torch.int32, torch.int64):
                    raise ValueError("topk_idx must be int32/int64 on the buffer's NPU")
                if topk_idx.ndim != 2 or topk_idx.shape[1] != self.num_topk:
                    raise ValueError("topk_idx must have shape [BS, topk]")
                bs = topk_idx.shape[0]
                if not 0 < bs <= self.num_max_tokens_per_rank:
                    raise ValueError(
                        f"BS={bs} must be in [1,{self.num_max_tokens_per_rank}]; "
                        "create UrmaBuffer with num_max_tokens_per_rank >= BS"
                    )
                low, high = torch.stack((topk_idx.min(), topk_idx.max())).cpu().tolist()
                if low < 0 or high >= self.num_experts:
                    raise ValueError("expert ids must be in [0, num_experts)")
            except Exception as failure:
                error = str(failure)
            collective_check(
                self.group, "URMA route input", error, tuple(topk_idx.shape) if hasattr(topk_idx, "shape") else None
            )
            routing = None
            try:
                # Keep version counters even when the application invokes the
                # inference-only API inside torch.inference_mode().
                with torch.inference_mode(False), torch.no_grad():
                    routing = prepare_routes(topk_idx, self.num_experts)
                local_counts = routing.expert_counts.cpu().tolist()
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, "URMA device routing", error)
            counts = self.group.all_gather(local_counts, "URMA expert counts")
            local_experts = self.num_experts // self.group.size()
            received_by_rank = [
                sum(sum(source[rank * local_experts : (rank + 1) * local_experts]) for source in counts)
                for rank in range(self.group.size())
            ]
            if any(rows > self.max_recv_rows for rows in received_by_rank):
                raise ValueError(
                    f"URMA receive capacity {self.max_recv_rows} exceeded: {received_by_rank}; "
                    "increase max_recv_rows collectively before creating the buffer"
                )
            plan, error = None, None
            try:
                counts_cpu = torch.tensor(counts, dtype=torch.int64, device="cpu")
                address = require_native("URMA MegaMoE").megamoe_workspace_address(self._runtime.native)
                plan = _extension().UrmaPlan(
                    self._example[:bs],
                    counts_cpu,
                    self.intermediate_hidden,
                    self.num_topk,
                    self.cv_rows,
                    self.max_recv_rows,
                    self.num_max_tokens_per_rank,
                    self.group.rank(),
                    _extension().ffts_config(),
                    address,
                    self._runtime.native.capacity,
                    self.gmm1_ahead_mgroups,
                )
                self._bind_reduce_dependencies(plan, routing.reduce_index)
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, "URMA descriptor preparation", error)
            tensors = (routing.order, routing.reduce_index, routing.expert_counts, routing.expert_offsets)
            return PreparedUrmaRoutes(
                self, plan, routing, bs, received_by_rank[self.group.rank()], tuple(value._version for value in tensors)
            )

    def forward_prepared(self, x, topk_weights, l1_weights, l2_weights, prepared, *, timer=None, out=None, trace=None):
        """Expand/Dispatch/GMM1/SwiGLU/GMM2/return PUT and flag-gated weighted reduction."""
        import torch
        from .api import _check_fp4_weight_format, _RoutedWeightPair

        with self._operation("forward"):
            if not isinstance(prepared, PreparedUrmaRoutes) or prepared.owner is not self:
                raise ValueError("prepared routes belong to a different buffer")
            routing = prepared.routing
            tensors = (routing.order, routing.reduce_index, routing.expert_counts, routing.expert_offsets)
            if tuple(value._version for value in tensors) != prepared.versions:
                raise ValueError("prepared routing tensors were modified in-place")
            if not isinstance(x, tuple) or len(x) != 2:
                raise TypeError("x must be (FP8 E4M3 data, E8M0 scales)")
            data, scales = x
            if data.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
                raise TypeError("URMA requires FP8 E4M3 data and E8M0 scales")
            if tuple(data.shape) != (prepared.bs, self.hidden) or tuple(scales.shape) not in (
                (prepared.bs, self.hidden // 32),
                (prepared.bs, self.hidden // 64, 2),
            ):
                raise ValueError("input or scale shape differs from the prepared routes")
            scales = scales.reshape(prepared.bs, self.hidden // 64, 2)
            if tuple(topk_weights.shape) != (prepared.bs, self.num_topk) or topk_weights.dtype != torch.float32:
                raise ValueError("top-k weights must be float32 [BS, topk]")
            if not isinstance(l1_weights, _RoutedWeightPair):
                raise ValueError("prepare both GMM1 layouts using transform_weights_for_mega_moe")
            # Prepared offline: one physical N256 projection = gate128 + up128.
            weight1, scale1 = l1_weights.select(True)
            weight2, scale2 = l2_weights
            for name, weight in (("l1_weights", weight1), ("l2_weights", weight2)):
                if weight.dtype != torch.uint8:
                    raise TypeError("URMA first profile requires packed FP4 weights")
                _check_fp4_weight_format(weight, name)
            if trace is not None:
                extension = _trace_extension()
                row_elements = extension.urma_trace_num_engines * extension.urma_trace_record_elements
                if (
                    trace.dtype != torch.int64
                    or trace.ndim != 1
                    or not trace.is_contiguous()
                    or trace.numel() % row_elements
                    or not 2 * row_elements <= trace.numel() <= (extension.urma_trace_max_capacity + 1) * row_elements
                ):
                    raise ValueError(
                        "trace must be contiguous 1D int64 with "
                        "84 * (16 + capacity * 16) elements, capacity in [1,65536]"
                    )
                if trace.data_ptr() % 128:
                    raise ValueError("trace storage must be 128-byte aligned; use UrmaBuffer.allocate_trace()")
            operands = (data, scales, topk_weights, weight1, weight2, scale1, scale2, timer, out, trace)
            if any(value is not None and value.device != self._device for value in operands):
                raise ValueError("all tensors must reside on the creating NPU")
            if any(value is not None and value.requires_grad for value in operands):
                raise ValueError("URMA MegaMoE is inference-only")
            if self._epoch >= (1 << 31) - 1:
                raise RuntimeError("URMA epoch exhausted; destroy and recreate the collective runtime")
            self._epoch += 1
            self._last_prepared = prepared
            native_operands = (
                data,
                scales,
                routing.order,
                routing.reduce_index,
                topk_weights,
                weight1,
                weight2,
                scale1,
                scale2,
                timer,
                out,
                self._epoch,
            )
            # Preserve the original positional signature for untraced calls,
            # including older test doubles and wheels used without this feature.
            if trace is not None:
                native_operands += (trace,)
            return self._enqueue_forward(prepared, native_operands)

    def forward(self, x, topk_idx, topk_weights, l1_weights, l2_weights, *, timer=None, out=None, trace=None):
        """Complete invocation, including fresh real-input route preparation."""
        with self._lock:
            prepared = self.prepare_routes(topk_idx)
            options = dict(timer=timer, out=out)
            if trace is not None:
                options["trace"] = trace
            return self.forward_prepared(x, topk_weights, l1_weights, l2_weights, prepared, **options)

    def destroy(self):
        with self._lock:
            if self._runtime.closed:
                return
            if self._failure is not None:
                raise RuntimeError(
                    "URMA job was aborted; terminate all workers instead of entering collective teardown"
                )
            try:
                # Runtime.close synchronizes the dedicated execution stream
                # before releasing the shared workspace and communication domain.
                self._runtime.close()
            except Exception as error:
                self._failure = f"destroy: {type(error).__name__}: {error}"
                self.group.abort(self._failure)
                raise
            self._last_prepared = None
            self._example = None

    def __del__(self):
        warn_unclosed(self)
