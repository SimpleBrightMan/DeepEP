"""CPU stream/lifetime doubles; these do not emulate device execution."""

from contextlib import contextmanager
from types import SimpleNamespace
import sys
import threading
from math import prod

import pytest

from deep_ep_experimental.megamoe import urma
from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe.urma_routing import DeviceRoutes


class Stream:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def wait_stream(self, other):
        self.events.append(("wait", self.name, other.name))

    def synchronize(self):
        self.events.append(("sync", self.name))


class Npu:
    def __init__(self, current, events):
        self.current, self.events = current, events

    def current_stream(self, device):
        assert device == 0
        return self.current

    @contextmanager
    def stream(self, target):
        old, self.current = self.current, target
        self.events.append(("enter", target.name))
        try:
            yield
        finally:
            self.current = old
            self.events.append(("restore", old.name))


class Runtime:
    def __init__(self, stream):
        self.stream = stream
        self.device = 0
        self.closed = False

    def ensure_alive(self):
        if self.closed:
            raise RuntimeError("runtime is closed")

    def close(self):
        self.stream.synchronize()
        self.closed = True


def owner_for(stream):
    owner = urma.UrmaBuffer.__new__(urma.UrmaBuffer)
    owner._lock = threading.RLock()
    owner._runtime = Runtime(stream)
    owner._failure = None
    owner._epoch = 0
    owner._last_prepared = object()
    owner._example = object()
    aborts = []
    owner.group = SimpleNamespace(abort=aborts.append, size=lambda: 16)
    return owner, aborts


def test_different_callers_submit_to_one_device_stream_with_both_event_edges(monkeypatch):
    events = []
    stream_a, stream_b, execution = (Stream(name, events) for name in ("a", "b", "owner"))
    npu = Npu(stream_a, events)
    monkeypatch.setattr(urma, "npu_api", lambda: npu)
    owner, aborts = owner_for(execution)
    result = SimpleNamespace(record_stream=lambda stream: events.append(("record", stream.name)))

    def forward(*operands):
        assert npu.current is execution
        events.append(("forward", operands[-1]))
        return result

    prepared = SimpleNamespace(native=SimpleNamespace(forward=forward))
    with owner._operation("first"):
        assert owner._enqueue_forward(prepared, (1,)) is result
    npu.current = stream_b
    with owner._operation("second"):
        assert owner._enqueue_forward(prepared, (2,)) is result
    assert events == [
        ("wait", "owner", "a"),
        ("enter", "owner"),
        ("forward", 1),
        ("restore", "a"),
        ("wait", "a", "owner"),
        ("record", "a"),
        ("wait", "owner", "b"),
        ("enter", "owner"),
        ("forward", 2),
        ("restore", "b"),
        ("wait", "b", "owner"),
        ("record", "b"),
    ]
    assert not aborts
    owner.destroy()
    assert events[-1] == ("sync", "owner")


def test_native_failure_restores_caller_and_poison_prevents_retry_or_teardown(monkeypatch):
    events = []
    caller, execution = Stream("caller", events), Stream("owner", events)
    npu = Npu(caller, events)
    monkeypatch.setattr(urma, "npu_api", lambda: npu)
    owner, aborts = owner_for(execution)

    def forward(*operands):
        raise ValueError("native launch rejected")

    prepared = SimpleNamespace(native=SimpleNamespace(forward=forward))
    with pytest.raises(ValueError, match="native launch"):
        with owner._operation("forward"):
            owner._enqueue_forward(prepared, ())
    assert npu.current is caller
    assert len(aborts) == 1 and "native launch rejected" in aborts[0]
    with pytest.raises(RuntimeError, match="unusable"):
        with owner._operation("retry"):
            pytest.fail("an aborted owner must never retry")
    assert len(aborts) == 1
    with pytest.raises(RuntimeError, match="terminate all workers"):
        owner.destroy()
    assert ("sync", "owner") not in events
    owner._runtime.closed = True  # Model launcher process termination, not a collective cleanup.


def test_dependency_gather_stays_ordered_when_forward_switches_caller_stream(monkeypatch):
    events = []
    caller_a, caller_b, execution = (Stream(name, events) for name in ("a", "b", "owner"))
    npu = Npu(caller_a, events)
    monkeypatch.setattr(urma, "npu_api", lambda: npu)
    owner, _ = owner_for(execution)
    inverse = object()

    def bind(value):
        assert value is inverse and npu.current is execution
        events.append(("gather", npu.current.name))

    def forward(*_):
        assert npu.current is execution
        events.append(("forward", npu.current.name))
        return SimpleNamespace(record_stream=lambda stream: events.append(("record", stream.name)))

    plan = SimpleNamespace(bind_reduce_dependencies=bind, forward=forward)
    owner._bind_reduce_dependencies(plan, inverse)
    npu.current = caller_b
    owner._enqueue_forward(SimpleNamespace(native=plan), ())
    assert events == [
        ("wait", "owner", "a"),
        ("enter", "owner"),
        ("gather", "owner"),
        ("restore", "a"),
        ("wait", "owner", "b"),
        ("enter", "owner"),
        ("forward", "owner"),
        ("restore", "b"),
        ("wait", "b", "owner"),
        ("record", "b"),
    ]
    owner.destroy()


def test_closed_buffer_rejects_reuse_and_destroy_is_idempotent():
    events = []
    owner, _ = owner_for(Stream("owner", events))
    owner.destroy()
    owner.destroy()
    assert events == [("sync", "owner")]
    assert owner._last_prepared is None and owner._example is None
    with pytest.raises(RuntimeError, match="closed"):
        with owner._operation("forward"):
            pytest.fail("a destroyed owner must not launch")


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype
        self.device = "npu:0"
        self.requires_grad = False
        self._version = 0
        self.streams = []
        self.contiguous = True
        self.address = 128

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.address

    def reshape(self, *shape):
        return Tensor(shape, self.dtype)

    def record_stream(self, stream):
        self.streams.append(stream)


def prepared_call_setup(monkeypatch):
    dtypes = dict(float8_e4m3fn="fp8", float8_e8m0fnu="scale", uint8="fp4", float32="float", int64="int64")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(**dtypes))
    monkeypatch.setattr(api, "_check_fp4_weight_format", lambda *_: None)
    events = []
    caller, execution = Stream("caller", events), Stream("owner", events)
    monkeypatch.setattr(urma, "npu_api", lambda: Npu(caller, events))
    owner, aborts = owner_for(execution)
    owner.hidden, owner.num_topk, owner._device = 3584, 8, "npu:0"
    owner.num_experts = 384
    x = Tensor((2048, 3584), "fp8"), Tensor((2048, 112), "scale")
    topk_weights = Tensor((2048, 8), "float")
    w1 = api._RoutedWeightPair((Tensor((), "fp4"), Tensor((), "scale")), (Tensor((), "fp4"), Tensor((), "scale")))
    w2 = Tensor((), "fp4"), Tensor((), "scale")
    epochs = []

    def forward(*operands):
        assert len(operands) == 12  # The existing untraced native signature stays compatible.
        assert operands[1].shape == (2048, 56, 2)
        epochs.append(operands[-1])
        return Tensor((2048, 3584), "bf16")

    def plan():
        routes = DeviceRoutes(*(Tensor((), "int64") for _ in range(4)))
        return urma.PreparedUrmaRoutes(owner, SimpleNamespace(forward=forward), routes, 2048, 16384, (0, 0, 0, 0))

    return owner, aborts, x, topk_weights, w1, w2, epochs, plan


def test_epoch_is_monotonic_across_plans_and_input_scale_shape_is_normalized(monkeypatch):
    owner, aborts, x, probs, w1, w2, epochs, make_plan = prepared_call_setup(monkeypatch)
    first, second = make_plan(), make_plan()
    owner.forward_prepared(x, probs, w1, w2, first)
    owner.forward_prepared(x, probs, w1, w2, second)
    owner.forward_prepared(x, probs, w1, w2, first)
    assert epochs == [1, 2, 3] and owner._epoch == 3
    assert not aborts
    owner.destroy()


@pytest.mark.parametrize(
    "world,bs,expected",
    [(4, 1024, True), (4, 1535, True), (4, 1536, False), (4, 2048, False),
     (8, 767, True), (8, 768, False), (8, 1024, False),
     (16, 383, True), (16, 384, False), (16, 1024, False)],
)
def test_gate_up_layout_uses_actual_expected_expert_rows(monkeypatch, world, bs, expected):
    from dataclasses import replace

    owner, _, _, _, w1, w2, _, make_plan = prepared_call_setup(monkeypatch)
    owner.group.size = lambda: world
    owner.num_max_tokens_per_rank = 8192  # Capacity must not select the layout.
    prepared = replace(make_plan(), bs=bs)
    selected = []

    def forward(*operands):
        selected.append((operands[5], operands[7]))
        return Tensor((bs, 3584), "bf16")

    prepared.native.forward = forward
    x = Tensor((bs, 3584), "fp8"), Tensor((bs, 112), "scale")
    owner.forward_prepared(x, Tensor((bs, 8), "float"), w1, w2, prepared)
    wanted = tuple(w1) if expected else w1._planar
    assert selected[0][0] is wanted[0] and selected[0][1] is wanted[1]
    assert urma.use_fused_gate_up(bs, world) is expected
    owner.destroy()


def test_modified_prepared_routes_abort_before_epoch_or_native_submission(monkeypatch):
    owner, aborts, x, probs, w1, w2, epochs, make_plan = prepared_call_setup(monkeypatch)
    prepared = make_plan()
    prepared.routing.reduce_index._version += 1
    with pytest.raises(ValueError, match="modified in-place"):
        owner.forward_prepared(x, probs, w1, w2, prepared)
    assert epochs == [] and owner._epoch == 0 and len(aborts) == 1
    owner._runtime.closed = True


@pytest.mark.parametrize("capacity", [1, 1024, 2048, 8192])
@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("world", [4, 8, 16])
def test_buffer_passes_explicit_multi_host_flag_to_collective_signature_and_runtime(monkeypatch, allow, world, capacity):
    from deep_ep._store import StoreGroup

    fake_torch = SimpleNamespace(device=lambda name: name, empty=lambda *a, **k: object(), float8_e4m3fn="fp8")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(urma, "npu_api", lambda: SimpleNamespace(current_device=lambda: 0))
    monkeypatch.setattr(urma, "_extension", lambda: SimpleNamespace(urma_buffer_size=lambda *a: 2 << 20))
    checks, created = [], []

    def check(group, stage, error, signature=None):
        assert error is None
        checks.append((stage, signature))

    def runtime(group, capacity, timeout, *, allow_multi_host):
        created.append(allow_multi_host)
        return SimpleNamespace(closed=True)

    monkeypatch.setattr(urma, "collective_check", check)
    monkeypatch.setattr(urma, "_UrmaRuntime", runtime)
    buffer = urma.UrmaBuffer(
        StoreGroup(None, 0, world, prefix="test"),
        num_max_tokens_per_rank=capacity,
        allow_multi_host=allow,
    )
    assert buffer.allow_multi_host is allow
    assert buffer.num_max_tokens_per_rank == capacity
    assert created == [allow]
    assert checks[0][1][-4] is allow
    assert checks[0][1][-3] == "combine-ready-scan-reduce-world-v1"
    assert checks[0][1][-2] == "expert-m256-expected-lt128-gateup-v2"
    assert checks[0][1][-1] == urma.COMMUNICATION_PROTOCOL
    assert buffer.gmm1_ahead_mgroups == 12


@pytest.mark.parametrize(
    "tail", [None, "combine-cv-flag-reduce41", "combine-finished-fast-reduce41", "combine-ready-scan-reduce-world-v1"]
)
def test_native_tail_schedule_must_match_before_runtime_creation(monkeypatch, tail):
    monkeypatch.setattr(urma, "require_native", lambda *_: SimpleNamespace(create_megamoe_urma_runtime=object(), megamoe_urma_communication_protocol=urma.COMMUNICATION_PROTOCOL))
    extension = SimpleNamespace(
        urma_api_protocol=1, urma_tail_schedule=tail, urma_compute_schedule="expert-m256-expected-lt128-gateup-v2",
        urma_communication_protocol=urma.COMMUNICATION_PROTOCOL
    )
    monkeypatch.setattr(urma.importlib, "import_module", lambda *_: extension)
    urma._extension.cache_clear()
    try:
        if tail != "combine-ready-scan-reduce-world-v1":
            with pytest.raises(RuntimeError, match="tailCover needs its matching native wheel"):
                urma._extension()
        else:
            assert urma._extension() is extension
    finally:
        urma._extension.cache_clear()


@pytest.mark.parametrize("schedule", [None, "expert-m256", "expert-m384-n128", "expert-m256-mgroup-lookahead-v1"])
def test_native_mgroup_schedule_must_match_before_runtime_creation(monkeypatch, schedule):
    monkeypatch.setattr(urma, "require_native", lambda *_: SimpleNamespace(create_megamoe_urma_runtime=object(), megamoe_urma_communication_protocol=urma.COMMUNICATION_PROTOCOL))
    extension = SimpleNamespace(
        urma_api_protocol=1, urma_tail_schedule="combine-ready-scan-reduce-world-v1", urma_compute_schedule=schedule
    )
    monkeypatch.setattr(urma.importlib, "import_module", lambda *_: extension)
    urma._extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="M-group lookahead needs its matching native wheel"):
            urma._extension()
    finally:
        urma._extension.cache_clear()


def test_benchmark_summaries_preserve_two_eight_rank_nodes():
    from deep_ep_experimental.megamoe.urma_timing import summarize_profile_rounds

    report = summarize_profile_rounds([[float(rank + 1)] * 4 for rank in range(16)], ranks_per_node=8)
    assert [node["rank_start"] for node in report["per_node"]] == [0, 8]


@pytest.mark.parametrize("old_module", ["core", "experimental"])
def test_commfirst_protocol_rejects_old_native_modules(monkeypatch, old_module):
    protocol = urma.COMMUNICATION_PROTOCOL
    core = SimpleNamespace(create_megamoe_urma_runtime=object(), megamoe_urma_communication_protocol=protocol)
    extension = SimpleNamespace(
        urma_api_protocol=1,
        urma_tail_schedule="combine-ready-scan-reduce-world-v1",
        urma_compute_schedule="expert-m256-expected-lt128-gateup-v2",
        urma_communication_protocol=protocol,
    )
    if old_module == "core":
        del core.megamoe_urma_communication_protocol
    else:
        del extension.urma_communication_protocol
    monkeypatch.setattr(urma, "require_native", lambda *_: core)
    monkeypatch.setattr(urma.importlib, "import_module", lambda *_: extension)
    urma._extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="matching core and experimental"):
            urma._extension()
    finally:
        urma._extension.cache_clear()


def trace_extension(**changes):
    values = dict(
        urma_trace_schema=1,
        urma_trace_num_engines=84,
        urma_trace_record_elements=16,
        urma_trace_header_elements=16,
        urma_trace_max_capacity=65536,
        urma_timer_enabled=True,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("capacity", [None, 1, 65536])
def test_trace_allocation_uses_native_slice_layout_and_owner_device(monkeypatch, capacity):
    allocations = []

    def zeros(numel, *, dtype, device):
        allocations.append((numel, dtype, device))
        return Tensor((numel,), dtype)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=zeros, int64="int64"))
    monkeypatch.setattr(urma, "_extension", lambda: trace_extension())
    owner, aborts = owner_for(Stream("owner", []))
    owner._device = "npu:0"
    trace = owner.allocate_trace() if capacity is None else owner.allocate_trace(capacity)
    expected = 84 * (16 + (4096 if capacity is None else capacity) * 16)
    assert allocations == [(expected, "int64", "npu:0")]
    assert trace.shape == (expected,) and not aborts
    owner.destroy()


@pytest.mark.parametrize("capacity", [0, -1, 65537, True, 1.5])
def test_trace_allocation_rejects_invalid_capacity_before_allocating(monkeypatch, capacity):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(urma, "_extension", lambda: trace_extension())
    owner, aborts = owner_for(Stream("owner", []))
    with pytest.raises(ValueError, match="trace capacity must be an integer"):
        owner.allocate_trace(capacity)
    assert len(aborts) == 1
    owner._runtime.closed = True


@pytest.mark.parametrize(
    "extension, message",
    [
        (SimpleNamespace(), "rebuild and reinstall"),
        (trace_extension(urma_trace_schema=2), "rebuild and reinstall"),
        (trace_extension(urma_timer_enabled=False), "DEEPEP_MEGAMOE_TIMER=ON"),
    ],
)
def test_trace_requires_matching_native_trace_abi_and_timer_build(monkeypatch, extension, message):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(urma, "_extension", lambda: extension)
    owner, aborts = owner_for(Stream("owner", []))
    with pytest.raises(RuntimeError, match=message):
        owner.allocate_trace()
    assert len(aborts) == 1
    owner._runtime.closed = True


@pytest.mark.parametrize("fresh_routes", [False, True])
def test_trace_forward_appends_optional_argument_without_moving_timer_out_or_epoch(monkeypatch, fresh_routes):
    owner, aborts, x, probs, w1, w2, _, make_plan = prepared_call_setup(monkeypatch)
    monkeypatch.setattr(urma, "_extension", lambda: trace_extension())
    prepared = make_plan()
    trace = Tensor((84 * 16 * 5,), "int64")
    timer = Tensor((84 * 8 * 16,), "int64")
    output = Tensor((2048, 3584), "bf16")
    calls = []

    def forward(*operands):
        calls.append(operands)
        return output

    prepared.native.forward = forward
    if fresh_routes:
        monkeypatch.setattr(owner, "prepare_routes", lambda _: prepared)
        result = owner.forward(x, object(), probs, w1, w2, timer=timer, out=output, trace=trace)
    else:
        result = owner.forward_prepared(x, probs, w1, w2, prepared, timer=timer, out=output, trace=trace)
    assert result is output and len(calls) == 1 and len(calls[0]) == 13
    assert calls[0][9:] == (timer, output, 1, trace)
    assert owner._epoch == 1 and not aborts
    owner.destroy()


@pytest.mark.parametrize(
    "fault", ["dtype", "rank", "contiguous", "fraction", "empty", "oversize", "device", "alignment"]
)
def test_invalid_trace_rejected_before_epoch_or_native_launch(monkeypatch, fault):
    owner, aborts, x, probs, w1, w2, epochs, make_plan = prepared_call_setup(monkeypatch)
    monkeypatch.setattr(urma, "_extension", lambda: trace_extension())
    trace = Tensor((84 * 16 * 5,), "int64")
    if fault == "dtype":
        trace.dtype = "float"
    elif fault == "rank":
        trace.shape = (84, 16 * 5)
    elif fault == "contiguous":
        trace.contiguous = False
    elif fault == "fraction":
        trace.shape = (84 * 16 * 5 + 1,)
    elif fault == "empty":
        trace.shape = (84 * 16,)
    elif fault == "oversize":
        trace.shape = (84 * 16 * 65538,)
    elif fault == "device":
        trace.device = "npu:1"
    else:
        trace.address += 8  # A contiguous int64 storage_offset=1 view is still misaligned.
    message = {"device": "creating NPU", "alignment": "128-byte aligned"}.get(fault, "trace must be")
    with pytest.raises(ValueError, match=message):
        owner.forward_prepared(x, probs, w1, w2, make_plan(), trace=trace)
    assert not epochs and owner._epoch == 0 and len(aborts) == 1
    owner._runtime.closed = True


def test_untraced_forward_does_not_require_new_trace_extension_properties(monkeypatch):
    owner, aborts, x, probs, w1, w2, epochs, make_plan = prepared_call_setup(monkeypatch)
    monkeypatch.setattr(urma, "_extension", lambda: pytest.fail("untraced call queried optional trace ABI"))
    owner.forward_prepared(x, probs, w1, w2, make_plan())
    assert epochs == [1] and not aborts
    owner.destroy()
