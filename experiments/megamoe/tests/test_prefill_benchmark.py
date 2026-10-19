# Copyright (c) 2026, SimpleBright_Man

"""Full benchmark control flow, mocked at device and profiling boundaries."""

from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api, benchmark as bench, prefill_report, reference, validation
from deep_ep_experimental.megamoe.timer import breakdown
from test_prefill_report import config as actual_config, event


@pytest.mark.parametrize(
    "enabled,selected,timer_on", [(True, None, True), (True, [0, 7], True), (False, None, True), (True, None, False)]
)
def test_full_benchmark_records_native_actual_configuration_and_rank_coverage(
    tmp_path, monkeypatch, enabled, selected, timer_on
):
    state = SimpleNamespace(calls=[], prepared=[], timer_allocations=0, closed=False, destroyed=False, cpu_reads=[])
    actual = actual_config(enabled)

    class Tensor:
        def __init__(self, data, label="device"):
            self.data, self.label = np.asarray(data), label

        def __getitem__(self, index):
            return Tensor(self.data[index], self.label)

        def copy_(self, other):
            self.data[...] = other.data
            return self

        def npu(self):
            return self

        def view(self, _dtype):
            return self

        def cpu(self):
            state.cpu_reads.append(self.label)
            assert self.label == "timer", "perf must not copy output to CPU"
            return self

        def numpy(self):
            return self.data

    torch = ModuleType("torch")
    for dtype in ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16", "float32"):
        setattr(torch, dtype, dtype)
    torch.npu = SimpleNamespace(synchronize=lambda: None)
    torch.from_numpy = Tensor
    torch.empty = lambda shape, **_kwargs: Tensor(np.zeros(shape))
    torch.zeros = torch.empty
    torch.exp = lambda _prefix, *, out: out
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Buffer:
        timer_enabled = True
        workspace_bytes = 123
        symmetric_buffer_bytes = 456

        def __init__(self, _group, *, num_max_tokens_per_rank, **model):
            self.num_max_tokens_per_rank = num_max_tokens_per_rank
            self.prefill_optimization_config = None
            self.x = Tensor(np.zeros((num_max_tokens_per_rank, model["hidden"])))
            self.x_sf = Tensor(np.zeros((num_max_tokens_per_rank, model["hidden"] // 32)))
            self.topk_idx = Tensor(np.zeros((num_max_tokens_per_rank, model["num_topk"])))
            self.topk_weights = Tensor(np.zeros((num_max_tokens_per_rank, model["num_topk"])))

        def validate_inputs(self, bs, **_kwargs):
            assert bs == 8

        def prepare(self, bs):
            state.prepared.append(bs)
            self.prefill_optimization_config = deepcopy(actual)

        def allocate_timer(self):
            state.timer_allocations += 1
            return Tensor(np.zeros(4, dtype=np.int64), "timer")

        def destroy(self):
            state.destroyed = True

    class Group:
        def rank(self):
            return 0

        def size(self):
            return 16

        def barrier(self, _stage):
            pass

        def all_gather(self, value, stage):
            records = [deepcopy(value) for _ in range(16)]
            for rank, record in enumerate(records):
                if isinstance(record, dict) and "rank" in record:
                    record["rank"] = rank
                if (
                    selected is not None
                    and stage.startswith(("timer breakdown", "Prefill bandwidth"))
                    and rank not in selected
                ):
                    records[rank] = None
            return records

        def close(self):
            state.closed = True

    def profile(call, *_args):
        call()
        call()
        return [dict(start_us=100, duration_us=17), dict(start_us=200, duration_us=19)]

    def collective(_group, _stage, error, *_args):
        if error is not None:
            raise ValueError(error)

    monkeypatch.setattr(api, "SymmBuffer", Buffer)
    monkeypatch.setattr(api, "fp8_fp4_mega_moe", lambda *_args, timer, **_kwargs: state.calls.append(timer is not None))
    monkeypatch.setattr(api, "transform_weights_for_mega_moe", lambda *pairs, **_kwargs: pairs)
    monkeypatch.setattr(reference, "make_weights", lambda *_args: [(np.zeros(1), np.zeros(1))] * 2)
    monkeypatch.setattr(reference, "quantize_input", lambda x: (x, np.zeros((8, 2))))
    monkeypatch.setattr(
        bench, "inputs", lambda *_args, **_kwargs: (np.zeros((8, 64)), np.zeros((8, 8)), np.zeros((8, 8)))
    )
    monkeypatch.setattr(bench, "collect_profile", profile)
    monkeypatch.setattr(validation, "collective_check", collective)
    monkeypatch.setattr(prefill_report, "read_events", lambda _raw, rank: [event(rank=rank)])
    monkeypatch.setattr(breakdown, "compact_events", lambda _raw, _rank: [])
    (tmp_path / "_build_info.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bench, "files", lambda _package: tmp_path)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "0")
    args = SimpleNamespace(
        output_dir=tmp_path / "run",
        mode="perf",
        input="fp8",
        routing="balanced",
        timer=timer_on,
        timer_ranks=selected,
        check_tokens=0,
        relative_rmse=0.02,
    )
    config = dict(
        model=dict(hidden=64, intermediate_hidden=128, num_experts=32, num_topk=8, num_shared_experts=0),
        bs_values=[8],
        warmup=1,
        iters=2,
        seed=0,
        alignment=dict(rows=1, cols=1, repeats=1),
    )
    bench.run(args, config, Group())
    output = args.output_dir
    manifest = json.loads((output / "rank_0/manifest.json").read_text())
    assert manifest["api"] == "fp8_fp4_mega_moe"
    assert manifest["timer_ranks"] == (list(range(16)) if selected is None else selected)
    assert manifest["prefill_optimization_configs"] == {"8": actual}
    resources = json.loads((output / "rank_0/bs_8/resources.json").read_text())
    assert resources["prefill_optimization"] == actual
    assert state.prepared == [8] and state.closed and state.destroyed
    assert state.timer_allocations == int(timer_on)
    assert state.calls == ([False, False, False, False, False, True] if timer_on else [False] * 3)
    assert state.cpu_reads == (["timer"] if timer_on else [])
    assert not list(output.rglob("*accuracy*"))
    profile = json.loads((output / "rank_0/bs_8/profiling.json").read_text())
    assert profile["rank_max_mean_us"] == 19
    if timer_on:
        summary = json.loads((output / "bs_8/timer/prefill_bandwidth.json").read_text())
        assert summary["measured_ranks"] == manifest["timer_ranks"]
        assert summary["all_ranks_measured"] == (selected is None)
        assert summary["config"] == actual
        assert summary["status"] == ("ok" if enabled else "unavailable")
        assert (output / "rank_0/bs_8/prefill_bandwidth.json").is_file()
    else:
        assert not list(output.rglob("prefill_bandwidth*"))
