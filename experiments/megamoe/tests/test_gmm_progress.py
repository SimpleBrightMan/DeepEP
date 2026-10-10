import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("progress", Path(__file__).parents[1] / "scripts/analyze_gmm_progress.py")
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def event(name, ts, dur, **task):
    return dict(name=name, ph="X", ts=ts, dur=dur, args=dict(group_id=0, rank=0, sub_id=0, **task))


def test_classification_and_ledger():
    task = dict(expert_id=1, row_start=0, row_count=256)
    events = [event("AIC_GMM1_0", 1, 10, **task), event("AIC_GMM1_1", 12, 10, **task),
              event("AIC_GMM2_ISSUE_0", 25, 5, **task),
              event("AIC_GMM2_ISSUE_1", 32, 5, expert_id=2, row_start=0, row_count=256)]
    lane = progress.analyze(events)["lanes"][0]
    assert lane["gap_stats"]["same_phase_same_task_coordinates"]["total_us"] == 1
    assert lane["gap_stats"]["GMM1_to_GMM2"]["total_us"] == 3
    assert lane["gap_stats"]["same_phase_expert_change"]["total_us"] == 2
    assert sum(v["total_us"] for v in lane["interval_stats"].values()) + sum(v["total_us"] for v in lane["gap_stats"].values()) == 36
    assert "GMM2:issue" in lane["interval_stats"]


def test_missing_coordinates_never_infer_batch():
    lane = progress.analyze([event("AIC_GMM2_ISSUE_0", 0, 5), event("AIC_GMM2_ISSUE_1", 6, 5)])["lanes"][0]
    assert set(lane["gap_stats"]) == {"same_phase_unknown_task"}


def test_raw_cycles_and_parallel_lanes():
    raw = [dict(name="AIC_GMM1_0", group_id=g, rank=0, sub_id=0, start_cycle=1000, end_cycle=3000) for g in (0, 1)]
    result = progress.analyze(raw)
    assert len(result["lanes"]) == 2
    assert result["lanes"][0]["interval_stats"]["GMM1:native_boundary"]["total_us"] == 2


def test_overlap_rejected():
    with pytest.raises(ValueError, match="Overlapping"):
        progress.analyze([event("AIC_GMM1_0", 0, 5), event("AIC_GMM2_ISSUE_0", 4, 5)])


def test_no_events_rejected():
    with pytest.raises(ValueError, match="No AIC"):
        progress.analyze([])
