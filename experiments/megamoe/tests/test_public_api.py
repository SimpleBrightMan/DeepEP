# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Execute public wrappers using CPU storage doubles; this does not simulate the NPU kernel."""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe.validation import collective_check


class Device(str):
    @property
    def type(self):
        return self.split(":")[0]


DTYPES = dict(
    uint8=np.uint8,
    int8=np.int8,
    int32=np.int32,
    int64=np.int64,
    float32=np.float32,
    bfloat16=np.float32,
    float8_e4m3fn=np.uint8,
    float8_e8m0fnu=np.uint8,
)


class Tensor:
    def __init__(self, data, dtype, device="npu:0"):
        self.data, self.dtype, self.device = np.asarray(data, dtype=DTYPES[dtype]), dtype, Device(device)
        self.requires_grad = False
        self.streams = []
        self.npu_format = 2
        self._version = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, index):
        return Tensor(self.data[index], self.dtype, self.device)

    def view(self, dtype):
        return Tensor(self.data.view(DTYPES[dtype]), dtype, self.device)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), self.dtype, self.device)

    def transpose(self, first, second):
        return Tensor(self.data.swapaxes(first, second), self.dtype, self.device)

    def permute(self, *axes):
        return Tensor(self.data.transpose(axes), self.dtype, self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.dtype, self.device)

    def clone(self):
        return Tensor(self.data.copy(), self.dtype, self.device)

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def untyped_storage(self):
        base = self.data
        while isinstance(base.base, np.ndarray):
            base = base.base
        return SimpleNamespace(data_ptr=lambda: base.ctypes.data)

    def record_stream(self, stream):
        self.streams.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()

    def add_(self, other):
        self.data += other.data
        self._version += 1
        return self


@pytest.fixture
def backend(monkeypatch):
    events = []

    class Group:
        def size(self):
            return 2

    def gather(output, value, **kwargs):
        output[:] = [value] * 2

    dist = SimpleNamespace(ProcessGroup=Group, get_world_size=lambda _: 2, all_gather_object=gather)
    torch = SimpleNamespace(
        **{name: name for name in DTYPES},
        device=Device,
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=DTYPES[dtype]), dtype, device),
        zeros=lambda shape, dtype, device: Tensor(np.zeros(shape), dtype, device),
        empty_like=lambda t, dtype: Tensor(np.empty(t.shape, dtype=DTYPES[dtype]), dtype, t.device),
        distributed=dist,
        inference_mode=lambda _: contextlib.nullcontext(),
        cat=lambda values,dim: Tensor(np.concatenate([t.data for t in values],axis=dim),values[0].dtype,values[0].device),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    def format_cast(t, acl_format, *, customize_dtype=None, input_dtype=None):
        # 检查实际传入的协议，不能把所有 format_cast 参数都静默忽略。
        assert acl_format == 29
        fp8 = t.dtype == "float8_e4m3fn"
        if fp8:
            assert customize_dtype is None and input_dtype is None
        else:
            assert t.dtype == "uint8"
            assert customize_dtype == "float8_e4m3fn" and input_dtype == "float4_e2m1fn_x2"
        events.append(("format_cast", t))
        t = Tensor(t.data.copy(), t.dtype, t.device)
        t.npu_format = 51 if fp8 else 50
        return t

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: t.npu_format, npu_format_cast=format_cast, float4_e2m1fn_x2="float4_e2m1fn_x2"
        ),
    )
    stream = SimpleNamespace(wait_stream=lambda _: events.append("wait"))
    npu = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda _: stream, stream=lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(api, "npu_api", lambda: npu)
    monkeypatch.setattr(api, "require_native", lambda _: SimpleNamespace(megamoe_workspace_address=lambda _: 4096))

    class Runtime:
        def __init__(self, group, capacity, timeout):
            self.device, self.stream, self.native = 0, stream, SimpleNamespace(capacity=capacity)
            self.closed = False

        def ensure_alive(self):
            if self.closed:
                raise RuntimeError("Buffer runtime has been destroyed")

        def close(self):
            events.append("close")
            self.closed = True

    monkeypatch.setattr(api, "_MegaRuntime", Runtime)

    class Plan:
        def __init__(self, example, *args):
            self.bs, self.workspace_bytes = example.shape[0], 4096
            events.append(("prepare", self.bs))

        def forward(self, *args):
            events.append(("launch", args))
            output = args[-1]
            output.data.fill(3)
            return output, Tensor([2, 3], "int32")

    ext = SimpleNamespace(
        buffer_size=lambda *args: 2 << 20,
        Plan=Plan,
        ffts_config=lambda: 123,
        timer_enabled=True,
        timer_numel=64,
        small_batch_max_tokens=256,
        paired_gate_up_max_tokens=65536,
    )
    monkeypatch.setattr(api, "_extension", lambda: ext)
    buffer = api.SymmBuffer(Group(), 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    yield SimpleNamespace(buffer=buffer, events=events, ext=ext, dist=dist)
    buffer.destroy()


def weights():
    prepared = (
        (Tensor(np.zeros((2, 512, 512)), "uint8"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 128)), "uint8"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )
    for weight, _ in prepared:
        weight.npu_format = 50
    return tuple(api._TilePackedWeightPair(pair) for pair in prepared)


@pytest.mark.parametrize("protocol", [8, 9, 10, 11])
def test_clamp_signature_rejects_old_native_protocol(monkeypatch, protocol):
    api._extension.cache_clear()
    monkeypatch.setattr(api, "require_native", lambda _: SimpleNamespace(create_megamoe_runtime=object()))
    monkeypatch.setattr(api.importlib, "import_module", lambda _: SimpleNamespace(api_protocol=protocol))
    with pytest.raises(RuntimeError, match="Python/native versions differ"):
        api._extension()
    api._extension.cache_clear()


def test_prequant_flag_is_frozen_in_prepared_plan(backend, monkeypatch):
    original = backend.ext.Plan
    configs = []
    def plan(*args):
        configs.append(args)
        return original(*args)
    monkeypatch.setattr(backend.ext, "Plan", plan)
    b = api.SymmBuffer(backend.buffer.group, 4, 17, 2, 1024, 256, route_weight_before_quant=True)
    try:
        b.prepare(17)
        assert configs[-1][-4:] == (-1, True, False, None)
        assert b.route_weight_before_quant is True
    finally:
        b.destroy()


@pytest.mark.parametrize("kwargs", [{"num_topk":9,"mma_type":"fp8xfp4","route_weight_before_quant":True},
                                    {"num_topk":2,"mma_type":"fp8xfp8","route_weight_before_quant":True},
                                    {"num_topk":2,"mma_type":"fp8xfp4","route_weight_before_quant":1}])
def test_prequant_contract_rejected_before_runtime_creation(backend, kwargs):
    previous=list(backend.events)
    with pytest.raises(ValueError, match="pre-quant routing|must be bool"):
        api.SymmBuffer(backend.buffer.group, 32, 17, kwargs['num_topk'], 1024, 256,
                      **{k:v for k,v in kwargs.items() if k!='num_topk'})
    assert backend.events == previous


def test_ragged_mask_validation_and_plan_contract(backend,monkeypatch):
    configs=[];original=backend.ext.Plan
    def plan(*args):
        configs.append(args);return original(*args)
    monkeypatch.setattr(backend.ext,'Plan',plan)
    b=api.SymmBuffer(backend.buffer.group,4,17,2,1024,256,variable_token_counts=True)
    try:
        b.topk_idx.data[:]=[-1,-1];b.topk_weights.data[:]=[0,0]
        b.validate_inputs(1);b.prepare(1)
        assert configs[-1][-2] is True
        b.topk_weights.data[0,0]=1
        with pytest.raises(ValueError,match='invalid routes require zero'):b.validate_inputs(1)
    finally:b.destroy()


def test_retained_graph_plan_survives_shape_switch(backend):
    b=backend.buffer;b.prepare(17);first=b._plan;b.retain_prepared_plan()
    b.prepare(1);b.prepare(17)
    assert b._plan is first
    assert [e for e in backend.events if isinstance(e,tuple) and e[0]=='prepare']==[('prepare',17),('prepare',1)]


def test_compact_preparation_omits_unused_planar_copy(backend):
    raw=((Tensor(np.zeros((2,512,512)),'uint8'),Tensor(np.full((2,512,16,2),127),'float8_e8m0fnu')),
         (Tensor(np.zeros((2,1024,128)),'uint8'),Tensor(np.full((2,1024,4,2),127),'float8_e8m0fnu')))
    l1,l2=api.transform_weights_for_mega_moe(*raw,keep_planar=False)
    assert isinstance(l1,api._TilePackedWeightPair) and not isinstance(l1,api._RoutedWeightPair)
    assert len([e for e in backend.events if isinstance(e,tuple) and e[0]=='format_cast'])==2


@pytest.mark.parametrize("bf16", [False, True])
def test_public_workflow_output_stats_timer_and_preparation_reuse(backend, bf16):
    b = backend.buffer
    assert b.x.shape == (17, 1024) and b.x_sf.shape == (17, 32)
    original = b.x
    assert b.allocate_inputs() is b and b.x is original
    x = Tensor(np.ones((17, 1024)), "bfloat16") if bf16 else None
    b.validate_inputs(17, x=x)
    b.prepare(17)
    b.prepare(17)
    y, stats = Tensor(np.zeros((17, 1024)), "bfloat16"), Tensor([7, 11], "int32")
    assert api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x) is None
    np.testing.assert_array_equal(y.data, 3)
    np.testing.assert_array_equal(stats.data, [9, 14])
    timer = b.allocate_timer()
    assert not timer.data.any()
    api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x, timer=timer)
    np.testing.assert_array_equal(stats.data, [11, 17])
    launches = [event[1] for event in backend.events if isinstance(event, tuple) and event[0] == "launch"]
    assert launches[0][8] is None and launches[1][8] is timer
    assert (launches[0][1] is None) == bf16
    assert launches[0][-1] is y and launches[1][-1] is y
    assert (
        sum(event == ("prepare", 17) for event in backend.events if isinstance(event, tuple) and event[0] == "prepare")
        == 1
    )
    assert stats.streams == [b._runtime.stream, b._runtime.stream]
    assert b.workspace_bytes == 4096 and b.symmetric_buffer_bytes == 2 << 20
    b.prepare(1)
    assert b._plan.bs == 1 and b._plan_bs == 1


def test_activation_clamp_is_explicit_per_launch(backend):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    for clamp in (None, 10.0, 2, None):
        api.fp8_fp4_mega_moe(y, *weights(), b, activation_clamp=clamp)
    calls = [event[1] for event in backend.events if isinstance(event, tuple) and event[0] == "launch"]
    assert [call[-2] for call in calls] == [None, 10.0, 2.0, None]
    assert all(call[-1] is y for call in calls)
    assert sum(event == ("prepare", 17) for event in backend.events) == 1


@pytest.mark.parametrize("clamp", [True, 0, -1, float("nan"), float("inf"), 1e40, "10"])
def test_invalid_activation_clamp_rejected_before_launch(backend, clamp):
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    with pytest.raises(ValueError, match="activation_clamp"):
        api.fp8_fp4_mega_moe(y, *weights(), backend.buffer, activation_clamp=clamp)
    assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events)


def test_compute_replay_checks_preparation_and_native_support(backend):
    b = backend.buffer
    with pytest.raises(RuntimeError, match="priming forward"):
        b.enable_compute_replay()
    b.prepare(17)
    with pytest.raises(RuntimeError, match="rebuild this branch"):
        b.enable_compute_replay()
    calls = []
    b._plan.enable_compute_replay = lambda: calls.append(b._plan_bs)
    b.enable_compute_replay()
    assert calls == [17]
    b.prepare(16)
    with pytest.raises(RuntimeError, match="rebuild this branch"):
        b.enable_compute_replay()  # 新 BS 是新计划，不能沿用上一 BS 的快照。


def test_output_and_stats_aliases_rejected_before_launch(backend):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    with pytest.raises(ValueError, match="alias"):
        api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=b.topk_idx[0])
    with pytest.raises(ValueError, match="overlap"):
        api.fp8_fp4_mega_moe(y, *weights(), b, x=y)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("bad", [-1, 4, 0])
def test_collective_routing_preflight_rejects_bad_ids(backend, bad):
    backend.buffer.topk_idx.data[0, 1] = bad
    with pytest.raises(ValueError, match="rank=0.*token=0"):
        backend.buffer.validate_inputs(17)


def test_collective_remote_error_and_bs_mismatch(backend):
    def mismatched(output, value, **kwargs):
        output[:] = [value, (None, 96)]

    backend.dist.all_gather_object = mismatched
    with pytest.raises(ValueError, match="ranks disagree.*rank=1: 96"):
        collective_check(backend.buffer.group, "prepare", None, 17)

    def failed(output, value, **kwargs):
        output[:] = [value, ("out of memory", 17)]

    backend.dist.all_gather_object = failed
    with pytest.raises(ValueError, match="rank=1: out of memory"):
        collective_check(backend.buffer.group, "prepare", None, 17)


def test_timer_disabled_and_destroyed_buffer_fail_explicitly(backend):
    backend.ext.timer_enabled = False
    with pytest.raises(RuntimeError, match="DEEPEP_MEGAMOE_TIMER"):
        backend.buffer.allocate_timer()
    backend.buffer.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), *weights(), backend.buffer)


def packed_tile_reference(raw):
    # 独立按地址序列枚举，不调用生产打包函数；16B段保持packed FP4原始位模式。
    e,n,half_k=raw.shape
    spans=[]
    for expert in range(e):
        for block_n in range(0,n,256):
            for window in range(0,half_k,256):
                for half_n in (0,128):
                    for k32 in range(window,min(window+256,half_k),16):
                        spans.append(raw[expert,block_n+half_n:block_n+half_n+128,k32:k32+16].reshape(-1))
    return np.concatenate(spans)


@pytest.mark.parametrize("n,k", [(256,64),(512,128),(512,768),(256,3072),(256,7168)])
def test_fp4_tile_pack_matches_physical_reader_and_k_tails(backend,n,k):
    raw=np.arange(2*n*k//2,dtype=np.uint32).astype(np.uint8).reshape(2,n,k//2)
    encoded=api._tilepack_fp4_for_nz_cast(Tensor(raw,"uint8"))
    # format_cast的标准NZ物理转置，与CPU storage double故意区分。
    physical=encoded.data.reshape(2,n,k//32,16).transpose(0,2,1,3).reshape(-1)
    np.testing.assert_array_equal(physical,packed_tile_reference(raw))
    assert encoded.shape==raw.shape
    np.testing.assert_array_equal(raw,np.arange(raw.size,dtype=np.uint32).astype(np.uint8).reshape(raw.shape))


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_fp4_and_packed_e8m0_are_reinterpreted_without_numeric_cast(backend, dtype):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    # Distinct row-dependent bytes expose both weight and scale column mismatches.
    row_ids = np.arange(512, dtype=np.uint16)
    raw[0, :, 0] = row_ids % 256
    raw[0, :, 1] = row_ids // 256
    sf = np.empty((1, 512, 4), dtype=np.uint8)
    sf[0, :, 0] = row_ids % 256
    sf[0, :, 1] = row_ids // 256
    sf[0, :, 2:] = np.array([129, 140], dtype=np.uint8)
    pair = Tensor(raw.view(DTYPES[dtype]), dtype), Tensor(sf.view(np.int32), "int32")
    order = np.array([*range(128), *range(256, 384), *range(128, 256), *range(384, 512)])
    prepared = api.transform_weights_for_mega_moe(pair, pair)
    for idx, (weight, scale) in enumerate(prepared):
        rows = order if idx == 0 else np.arange(512)
        physical=weight.data.reshape(1,512,4,16).transpose(0,2,1,3).reshape(-1)
        np.testing.assert_array_equal(physical, packed_tile_reference(raw[:, rows]))
        expected_sf = sf[:, rows].reshape(1,2,256,2,2).transpose(0,1,3,2,4).reshape(1,512,2,2)
        np.testing.assert_array_equal(scale.data, expected_sf)
        assert weight.dtype == "uint8" and scale.dtype == "float8_e8m0fnu"
        assert weight.npu_format == 50
    assert sum(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events) == 3
    planar_weight, planar_scale = prepared[0].select(False)
    physical=planar_weight.data.reshape(1,512,4,16).transpose(0,2,1,3).reshape(-1)
    np.testing.assert_array_equal(physical,packed_tile_reference(raw))
    np.testing.assert_array_equal(planar_scale.data, sf.reshape(1, 512, 2, 2))


def test_shared_fp8_preparation_retains_planar_weights_and_scales(backend):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    sf = np.arange(512 * 2, dtype=np.uint8).reshape(1, 512, 1, 2)
    pair = Tensor(raw, "float8_e4m3fn"), Tensor(sf, "float8_e8m0fnu")
    for idx, (weight, scale) in enumerate(api.transform_weights_for_mega_moe(pair, pair, shared=True)):
        np.testing.assert_array_equal(weight.data, raw)
        np.testing.assert_array_equal(scale.data, sf)
        assert weight.dtype == "float8_e4m3fn" and weight.npu_format == 51
    assert pair[0].npu_format == 2  # 准备不修改原始ND张量。


def shared_weights():
    pairs = []
    for n, k in ((512, 1024), (1024, 256)):
        weight = Tensor(np.zeros((2, n, k)), "float8_e4m3fn")
        weight.npu_format = 51
        pairs.append((weight, Tensor(np.full((2, n, k // 64, 2), 127), "float8_e8m0fnu")))
    return pairs


def test_mixed_routed_fp4_shared_fp8_reaches_native_without_hot_path_conversion(backend):
    backend.buffer.num_shared_experts = 2
    sw1, sw2 = shared_weights()
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), *weights(), backend.buffer,
                        shared_l1_weights=sw1, shared_l2_weights=sw2)
    launch = next(e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch")
    assert launch[9] is sw1[0] and launch[10] is sw2[0]
    assert not any(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_shared_fp4_rejected_at_preparation_and_forward(backend, dtype):
    pair = Tensor(np.zeros((2, 512, 512)), dtype), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")
    with pytest.raises(TypeError, match="shared weights must be FP8 E4M3"):
        api.transform_weights_for_mega_moe(pair, pair, shared=True)
    backend.buffer.num_shared_experts = 2
    with pytest.raises(TypeError, match="shared_l1_weights: shared weights must be FP8 E4M3"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), *weights(), backend.buffer,
                            shared_l1_weights=pair, shared_l2_weights=pair)
    assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events)


@pytest.mark.parametrize("name", ["shared_l1_weights", "shared_l2_weights"])
def test_shared_fp8_wrong_shape_rejected_before_launch(backend, name):
    backend.buffer.num_shared_experts = 2
    pairs = dict(zip(("l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"),
                     (*weights(), *shared_weights())))
    weight, scale = pairs[name]
    wrong = Tensor(np.zeros((*weight.shape[:-1], weight.shape[-1] // 2)), "float8_e4m3fn")
    wrong.npu_format = 51
    pairs[name] = (wrong, scale)
    with pytest.raises(ValueError, match=rf"{name}: expected FP8 shape"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), sym_buffer=backend.buffer, **pairs)
    assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events)


def test_shared_fp8_preparation_rejects_old_nz_and_bad_scale(backend):
    pairs = shared_weights()
    with pytest.raises(ValueError, match="prepare shared FP8 from raw ND"):
        api.transform_weights_for_mega_moe(*pairs, shared=True)
    for weight, _ in pairs:
        weight.npu_format = 2
    pairs[0] = pairs[0][0], Tensor(np.ones((2, 512, 1, 2)), "float8_e8m0fnu")
    with pytest.raises(ValueError, match="expected E8M0 scales"):
        api.transform_weights_for_mega_moe(*pairs, shared=True)


@pytest.mark.parametrize("n,pairs", [(256,48),(512,112),(1024,2)])
def test_production_weight_scale_block_mapping_is_lossless(backend,n,pairs):
    raw = np.arange(2*n*pairs*2,dtype=np.uint32).astype(np.uint8).reshape(2,n,pairs,2)
    packed = api._pack_weight_scale_blocks(Tensor(raw,"float8_e8m0fnu"))
    recovered = packed.data.reshape(2,n//256,pairs,256,2).transpose(0,1,3,2,4).reshape(raw.shape)
    np.testing.assert_array_equal(recovered,raw)
    assert packed.dtype == "float8_e8m0fnu" and packed.shape == raw.shape


def test_weight_scale_block_mapping_rejects_incomplete_n(backend):
    with pytest.raises(ValueError,match="N divisible by 256"):
        api._pack_weight_scale_blocks(Tensor(np.zeros((1,128,2,2)),"float8_e8m0fnu"))


@pytest.mark.parametrize("bf16", [False, True])
def test_routed_layout_stays_paired_above_4096_without_hot_path_cast(backend, bf16):
    b = backend.buffer
    capacity = 8192
    b.num_max_tokens_per_rank = capacity
    b.x = Tensor(np.zeros((capacity, 1024)), "float8_e4m3fn")
    b.x_sf = Tensor(np.zeros((capacity, 32)), "float8_e8m0fnu")
    b.topk_idx = Tensor(np.tile([0, 1], (capacity, 1)), "int32")
    b.topk_weights = Tensor(np.ones((capacity, 2)), "float32")
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    raw[0][0].data[:, :, 0] = np.arange(512, dtype=np.uint16) % 251
    w1, w2 = api.transform_weights_for_mega_moe(*raw)
    # Preparing weights must not share writable storage with the caller's raw scales.
    assert w1.select(False)[1].untyped_storage().data_ptr() != raw[0][1].untyped_storage().data_ptr()
    casts = sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)
    for bs in (96, 256, 257, 1024, 4096, 4097, capacity, 96):
        output = Tensor(np.zeros((bs, 1024)), "bfloat16")
        x = Tensor(np.ones((bs, 1024)), "bfloat16") if bf16 else None
        api.fp8_fp4_mega_moe(output, w1, w2, b, x=x)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        selected = w1.select(True)
        assert launch[4] is selected[0] and launch[6] is selected[1]
        assert b._plan.bs == bs  # actual BS, not max capacity or per-expert M
    assert sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events) == casts
    with pytest.raises(ValueError, match="keep its returned l1 pair intact"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((capacity, 1024)), "bfloat16"), tuple(w1), w2, b)


@pytest.mark.parametrize("layout", [True, False])
@pytest.mark.parametrize("index", [0, 1])
def test_prepared_layout_mutation_is_not_silently_cached(backend, layout, index):
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    w1, _ = api.transform_weights_for_mega_moe(*raw)
    tensor = w1.select(layout)[index]
    tensor.add_(Tensor(np.ones(tensor.shape), tensor.dtype))
    # Even requesting the other layout must fail rather than using an out-of-date copy.
    with pytest.raises(ValueError, match="modified in-place"):
        w1.select(not layout)


def test_fp4_transform_rejects_double_preparation(backend):
    pair = Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.ones((1, 256, 2, 2)), "float8_e8m0fnu")
    pair[0].npu_format = 50
    with pytest.raises(ValueError, match="already converted NZ"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_routed_transform_rejects_incomplete_gate_up_pair(backend):
    pair = Tensor(np.zeros((1, 128, 64)), "uint8"), Tensor(np.ones((1, 128, 2, 2)), "float8_e8m0fnu")
    with pytest.raises(ValueError, match="divisible by 128"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_fp8_weights_remain_nd_without_fp4_format_cast(backend):
    pair = (Tensor(np.zeros((1, 128, 128)), "float8_e4m3fn"), Tensor(np.full((1, 128, 2, 2), 127), "float8_e8m0fnu"))
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair):
        assert weight is pair[0]
        np.testing.assert_array_equal(scale.data, pair[1].data)
    assert not any(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events)


@pytest.mark.parametrize("weight_format", [29, 50])
def test_prepared_fp4_nz_variants_reach_launch(backend, weight_format):
    w1, w2 = weights()
    w1[0].npu_format = w2[0].npu_format = weight_format
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), w1, w2, backend.buffer)
    assert any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("name", ["l1_weights", "l2_weights"])
@pytest.mark.parametrize("weight_format", [2, 4, 51, 52, 53, 54])
def test_invalid_fp4_layout_rejected_with_tensor_context_before_launch(backend, name, weight_format):
    b = backend.buffer
    b.num_shared_experts = 2
    pairs = dict(zip(("l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"), (*weights(), *shared_weights())))
    pairs[name][0].npu_format = weight_format
    with pytest.raises(ValueError, match=rf"{name}:.*got format={weight_format}, dtype=uint8, shape="):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), sym_buffer=b, **pairs)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("name", ["shared_l1_weights", "shared_l2_weights"])
@pytest.mark.parametrize("weight_format", [2, 50, 52])
def test_shared_fp8_wrong_layout_rejected_before_launch(backend, name, weight_format):
    backend.buffer.num_shared_experts = 2
    pairs = dict(zip(("l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"), (*weights(), *shared_weights())))
    pairs[name][0].npu_format = weight_format
    with pytest.raises(ValueError, match=rf"{name}: FP8 weights require NZ.*got format={weight_format}"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), sym_buffer=backend.buffer, **pairs)
    assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events)


def test_format_cast_returning_nd_fails_at_preparation(backend, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "npu_format_cast", lambda t, *a, **k: t)
    pair = (Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.full((1, 256, 2, 2), 127), "float8_e8m0fnu"))
    with pytest.raises(ValueError, match="l1_weights:.*got format=2"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_store_group_public_configuration_and_prepare_without_process_group(backend, monkeypatch):
    from deep_ep._store import StoreGroup

    stages = []
    group = StoreGroup(None, 0, 2, prefix="api-test")

    def gather(value, stage):
        stages.append(stage)
        return [value, value]

    group.all_gather = gather

    def forbidden(*args, **kwargs):
        pytest.fail("Store API accessed a ProcessGroup")

    monkeypatch.setattr(backend.dist, "get_world_size", forbidden)
    monkeypatch.setattr(backend.dist, "all_gather_object", forbidden)
    buffer = api.SymmBuffer(group, 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    buffer.validate_inputs(17)
    buffer.prepare(17)
    buffer.destroy()
    assert stages == [
        "check:configuration",
        "check:input allocation",
        "check:input preflight",
        "check:prepare",
        "check:prepare BS=17",
    ]


def test_non_finite_route_weights_and_bad_capacity_fail_before_launch(backend):
    backend.buffer.topk_weights.data[2, 0] = np.nan
    with pytest.raises(ValueError, match="token=2.*finite"):
        backend.buffer.validate_inputs(17)
    with pytest.raises(ValueError, match="prepare.*BS"):
        backend.buffer.prepare(18)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_plan_allocation_failure_is_collective_and_retryable(backend):
    plan_type = backend.ext.Plan

    def fail(*args):
        raise RuntimeError("workspace allocation failed")

    backend.ext.Plan = fail
    with pytest.raises(ValueError, match="rank=0: workspace allocation failed"):
        backend.buffer.prepare(17)
    assert backend.buffer._plan is None and backend.buffer._plan_bs is None
    backend.ext.Plan = plan_type
    backend.buffer.prepare(17)
    assert backend.buffer._plan_bs == 17
