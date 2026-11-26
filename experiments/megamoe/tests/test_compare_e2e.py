import importlib.util
import json
from pathlib import Path

import pytest

from deep_ep_experimental.megamoe.performance import summarize_rounds

spec = importlib.util.spec_from_file_location("compare_e2e", Path(__file__).parents[1] / "scripts/compare_e2e.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_run(path, duration):
    path.mkdir()
    config = dict(bs_values=[32, 96], seed=123, warmup=10, iters=4, model={"num_shared_experts": 0})
    (path / "launch.json").write_text(json.dumps(dict(config=config, hosts=["127.0.0.1"],
        input="fp8", routing="balanced", mode="perf", timer=True)))
    (path / "artifacts.json").write_text(json.dumps(dict(commit="fixture", wheel_sha256="fixture",
        cann={"version": "fixture"}, build={"megamoe": {"torch": "fixture", "torch_npu": "fixture"}})))
    for bs in config["bs_values"]:
        target = path / f"bs_{bs}/rank_0/bs_{bs}/profiling.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(summarize_rounds([[duration + rank] * 4 for rank in range(8)], 8)))
    return path


def test_compares_slowest_rank_without_confusing_minimum(tmp_path):
    a, b = make_run(tmp_path / "a", 100), make_run(tmp_path / "b", 90)
    result = module.compare(a, b, [32, 96])
    assert "| 32 | 107.000 | 97.000 | 10.000 | 9.346% | 100.000 / 90.000 |" in result


@pytest.mark.parametrize("fault", ["replay", "missing_rank", "aggregate", "sample_count", "seed", "cann"])
def test_rejects_incomparable_or_invalid_runs(tmp_path, fault):
    a, b = make_run(tmp_path / "a", 100), make_run(tmp_path / "b", 90)
    path = b / "bs_32/rank_0/bs_32/profiling.json"
    if fault == "seed":
        path = b / "launch.json"
    elif fault == "cann":
        path = b / "artifacts.json"
    data = json.loads(path.read_text())
    if fault == "replay": data["metric"] = "megamoe_compute_replay_duration_us"
    elif fault == "missing_rank": data["per_rank_rounds_us"].pop()
    elif fault == "aggregate": data["rank_max_mean_us"] += 5
    elif fault == "sample_count": data["recorded_rounds"] += 2
    elif fault == "seed": data["config"]["seed"] += 1
    elif fault == "cann": data["cann"]["version"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.compare(a, b, [32, 96])
