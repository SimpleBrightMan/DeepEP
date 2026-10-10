# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import json

import numpy as np
import pytest

from deep_ep_experimental.megamoe.timer import protocol as p
from deep_ep_experimental.megamoe.timer.breakdown import compact_events, rank_breakdown, write_breakdown
from deep_ep_experimental.megamoe.timer.parse_timer_csv import decode
from deep_ep_experimental.megamoe.timer.trace_parser import raw_events_for_rank


def event(name, begin, end, group=0):
    return dict(name=name, rank=0, group_id=group, sub_id=0, start_cycle=begin * 1000, end_cycle=end * 1000)


def fixture_events(offset=0):
    events = [
        event("AIV_INPUT_BUFFER_INIT", 1, 11),
        event("AIV_INPUT_CROSS_RANK_SYNC", 11, 5011),
        event("AIV_DISPATCH_0", 5011, 5041),
        event("AIC_GMM1_0", 5041, 5081),
        event("AIC_GMM2_0", 5091, 5121),
        event("AIC_GMM1_0", 5041, 5081, 1),
        event("AIV_UNPERMUTE", 5121, 5131),
        event("AIC_GMM2_WAIT_0", 5071, 5091),
    ]
    return [dict(e, start_cycle=e["start_cycle"] + offset, end_cycle=e["end_cycle"] + offset) for e in events]


def test_compute_replay_requires_explicit_scope_and_has_no_fake_dispatch(tmp_path):
    events = [event("AIV_RESET", 1, 4), event("AIC_GMM1_0", 5, 45), event("AIC_GMM2_0", 50, 80)]
    assert rank_breakdown(events, 0)["status"] == "unavailable"
    result = rank_breakdown(events, 0, compute_replay=True)
    assert result["status"] == "ok"
    assert result["measurement"] == "compute_replay"
    assert result["head_us"] is result["launch_us"] is result["dispatch_at_us"] is None
    assert result["setup_us"] == 4 and result["gmm_window_us"] == 75
    assert result["gmm_uncovered_us"] == 5
    write_breakdown([result], tmp_path, "compute replay BS4096")
    assert "COMPUTE REPLAY" in (tmp_path / "timer_breakdown.txt").read_text()


def test_issue_interval_is_not_reported_as_completion():
    events = [event("AIC_GMM1_0", 5, 45), event("AIC_GMM2_ISSUE_0", 50, 80)]
    result = rank_breakdown(events, 0, compute_replay=True)
    assert result["status"] == "ok"
    assert result["gmm2_semantics"] == "issue_not_completion"
    assert result["gmm2_spans_us"] == 30


def test_both_gmms_can_have_independent_issue_semantics():
    events = [event("AIC_GMM1_ISSUE_0", 5, 45), event("AIC_GMM1_END_0", 45, 49),
              event("AIC_GMM2_ISSUE_0", 50, 80)]
    result = rank_breakdown(events, 0, compute_replay=True)
    assert result["status"] == "ok"
    assert result["gmm1_semantics"] == result["gmm2_semantics"] == "issue_not_completion"
    assert result["gmm1_spans_us"] == 40
    assert result["gmm_uncovered_us"] == 5  # End不能并入ISSUE伪装为MMAD时间。


@pytest.mark.parametrize("name", ["AIV_DISPATCH_0", "AIV_COMBINE_0", "AIV_UNPERMUTE", "AIV_INPUT_CROSS_RANK_SYNC", "AIV_FINAL_CROSS_RANK_WAIT"])
def test_compute_replay_rejects_communication_trace(name):
    with pytest.raises(ValueError, match="communication event"):
        rank_breakdown([event(name, 1, 2)], 0, compute_replay=True)


def test_breakdown_keeps_5ms_sync_and_uses_one_critical_lane(tmp_path):
    result = rank_breakdown(fixture_events(), 0)
    assert result["status"] == "ok"
    assert [result[k] for k in ("head_us", "launch_us", "gmm_window_us", "tail_us")] == [5010, 30, 80, 10]
    assert result["visible_span_us"] == 5130
    assert result["gmm1_spans_us"] == 40  # two parallel lanes, not 80
    assert result["gmm_uncovered_us"] == 10
    assert result["input_sync_max_us"] == 5000
    assert rank_breakdown(fixture_events(123456789), 0) == result
    write_breakdown([result], tmp_path, "BS=96")
    report = json.loads((tmp_path / "timer_breakdown.json").read_text())
    assert report["ranks"] == [result]
    assert "input-sync NOT compressed" in (tmp_path / "timer_breakdown.txt").read_text()


def test_missing_stages_not_zero_and_overlap_rejected():
    report = rank_breakdown([event("AIV_DISPATCH", 1, 2)], 0)
    assert report["status"] == "unavailable" and "head_us" not in report
    with pytest.raises(ValueError, match="overlapping"):
        rank_breakdown(fixture_events() + [event("AIC_GMM2_1", 5071, 5121)], 0)


def test_raw_fast_path_matches_csv_breakdown(tmp_path):
    path = tmp_path / "timer.bin"
    # 稀疏 mmap，使用真实协议偏移，避免为单测分配整个打点区。
    raw = np.memmap(path, dtype="<i8", mode="w+", shape=(p.TOTAL_BUFFER_SIZE,))
    for item in fixture_events():
        family = item["name"].removesuffix("_0")
        core = item["group_id"] * 3 + (0 if family.startswith("AIC") else 2)
        if family in p.FIXED_NAMES:
            slot = p.FIXED_NAMES.index(family)
        else:
            kind = p.DYNAMIC_NAMES.index(family)
            slot = len(p.FIXED_NAMES) + kind * p.MAX_DYNAMIC_ITER
            raw[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + kind] = 1
        offset = core * p.COUNTERS_PER_CORE + slot * 2
        raw[offset : offset + 2] = item["start_cycle"], item["end_cycle"]
    raw.flush()
    direct = rank_breakdown(compact_events(raw, 0), 0)
    parsed = tmp_path / "parsed.csv"
    decode(path, parsed)
    assert direct == rank_breakdown(raw_events_for_rank(parsed, 0), 0)
    raw[p.N_TIMING_COUNTER] = p.MAX_DYNAMIC_ITER + 1
    with pytest.raises(ValueError, match="count"):
        compact_events(raw, 0)
    raw._mmap.close()
