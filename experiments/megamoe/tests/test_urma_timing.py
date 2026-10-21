"""CPU-only contracts for the benchmark's device-aligned launch block."""

from types import ModuleType, SimpleNamespace
import sys

import pytest

from deep_ep_experimental.megamoe.urma_timing import UrmaLaunchGuard, measure_cached_forward


def fake_runtime(monkeypatch, *, peer_errors=None, initialized=False):
    calls = []
    torch = ModuleType("torch")
    dist = ModuleType("torch.distributed")
    torch.distributed = dist
    torch.float32 = "fp32"
    torch.zeros = lambda shape, **kw: calls.append(("zeros", tuple(shape), kw)) or "finite-input"
    torch.empty_like = lambda value: calls.append(("empty_like", value)) or "output"
    torch.exp = lambda value, out: calls.append(("exp", value, out))
    serial = [0]

    class Event:
        def __init__(self, **kwargs):
            self.index = serial[0]
            serial[0] += 1
            calls.append(("event-create", self.index))

        def record(self):
            calls.append(("record", self.index))

        def elapsed_time(self, end):
            assert any(call[0] == "synchronize" for call in calls)
            calls.append(("elapsed", self.index, end.index))
            return float(self.index + 1)

    torch.npu = SimpleNamespace(Event=Event, synchronize=lambda: calls.append(("synchronize",)))
    dist.is_initialized = lambda: initialized
    dist.PrefixStore = lambda prefix, store: calls.append(("prefix", prefix, store)) or "private-store"
    dist.init_process_group = lambda **kwargs: calls.append(("init", kwargs))
    dist.group = SimpleNamespace(WORLD="owned-hccl-group")
    dist.barrier = lambda group: calls.append(("device-barrier", group))
    dist.destroy_process_group = lambda group: calls.append(("destroy", group))
    group = SimpleNamespace(
        _prefix="unique-session/",
        _store="tcp-store",
        rank=lambda: 3,
        size=lambda: 16,
        all_gather=lambda error, stage: peer_errors if peer_errors is not None else [error] * 16,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    return group, calls


def test_continuous_block_uses_one_guard_and_one_drain_then_discards_warmup(monkeypatch):
    group, calls = fake_runtime(monkeypatch)
    guard = UrmaLaunchGuard(group, shape=(2, 3), repeat=2, timeout=71)
    values = measure_cached_forward(
        lambda: calls.append(("forward",)),
        warmup=1,
        iters=2,
        guard=guard,
    )
    assert values == [3000.0, 5000.0]
    assert [call[0] for call in calls if call[0] in ("device-barrier", "exp", "forward", "synchronize")] == [
        "device-barrier",
        "exp",
        "exp",
        "forward",
        "forward",
        "forward",
        "device-barrier",
        "synchronize",
    ]
    assert [call for call in calls if call[0] == "record"] == [("record", n) for n in range(6)]
    assert [call for call in calls if call[0] == "elapsed"] == [("elapsed", 2, 3), ("elapsed", 4, 5)]
    init = next(call[1] for call in calls if call[0] == "init")
    assert init["backend"] == "hccl" and init["store"] == "private-store"
    assert init["rank"] == 3 and init["world_size"] == 16 and init["timeout"].total_seconds() == 71
    assert ("prefix", "unique-session/urma-benchmark-hccl/", "tcp-store") in calls
    assert guard.describe()["allocated_bytes_per_rank"] == 48
    guard.close()
    guard.close()
    assert calls.count(("destroy", "owned-hccl-group")) == 1


def test_repeat_zero_retains_device_barrier_without_giant_allocation(monkeypatch):
    group, calls = fake_runtime(monkeypatch)
    guard = UrmaLaunchGuard(group, repeat=0)
    guard.begin()
    guard.end()
    assert not any(call[0] in ("zeros", "empty_like", "exp") for call in calls)
    assert calls.count(("device-barrier", "owned-hccl-group")) == 2
    assert guard.describe()["allocated_bytes_per_rank"] == 0


def test_peer_allocation_failure_prevents_collective_hccl_initialization(monkeypatch):
    group, calls = fake_runtime(monkeypatch, peer_errors=[None, "OutOfMemory"])
    with pytest.raises(RuntimeError, match="rank=1: OutOfMemory"):
        UrmaLaunchGuard(group, shape=(2, 3), repeat=1)
    assert not any(call[0] == "init" for call in calls)


def test_refuses_to_replace_an_existing_process_group(monkeypatch):
    group, calls = fake_runtime(monkeypatch, initialized=True)
    with pytest.raises(RuntimeError, match="pre-existing ProcessGroup"):
        UrmaLaunchGuard(group)
    assert not any(call[0] in ("init", "destroy", "zeros") for call in calls)


def test_cli_launch_guard_and_mgroup_options(tmp_path):
    from deep_ep_experimental.megamoe.urma_benchmark import parse_args

    base = ["--output-dir", str(tmp_path)]
    args = parse_args(base)
    assert args.gmm1_ahead_mgroups == 12
    assert tuple(args.exp_shape) == (100000, 20000) and args.exp_repeat == 50
    custom = parse_args(base + ["--exp-shape", "1000", "2000", "--exp-repeat", "0", "--gmm1-ahead-mgroups", "0"])
    assert custom.exp_shape == [1000, 2000] and custom.exp_repeat == custom.gmm1_ahead_mgroups == 0
    for options in (
        ["--exp-shape", "0", "2"],
        ["--exp-repeat", "-1"],
        ["--launch-timeout", "0"],
        ["--gmm1-ahead-mgroups", "-1"],
        ["--gmm1-ahead-mgroups", "65537"],
    ):
        with pytest.raises(SystemExit):
            parse_args(base + options)
