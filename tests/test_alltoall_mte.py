# Copyright (c) 2026, SimpleBrightMan

import json
import statistics

import pytest

from benchmarks.alltoall_mte import plan, summarize


def test_parse_token_counts_keeps_order_and_deduplicates():
    assert plan.parse_token_counts("1024,2048,4096") == [1024, 2048, 4096]
    assert plan.parse_token_counts("2048, 1024, 2048") == [2048, 1024]
    with pytest.raises(ValueError):
        plan.parse_token_counts("1024,,2048")
    with pytest.raises(ValueError):
        plan.parse_token_counts("0")
    with pytest.raises(ValueError):
        plan.parse_token_counts("16385")


def test_shape_validation():
    plan.validate_shape(16, 4, 3584)
    with pytest.raises(ValueError):
        plan.validate_shape(16, 3, 3584)
    with pytest.raises(ValueError):
        plan.validate_shape(16, 4, 3583)
    with pytest.raises(ValueError):
        plan.validate_shape(16, 4, 0)


def test_peer_scopes_for_four_card_groups():
    assert plan.peers_for(5, 16, 4, "local") == [5]
    assert plan.peers_for(5, 16, 4, "intra") == [4, 6, 7]
    inter = plan.peers_for(5, 16, 4, "inter")
    assert len(inter) == 12
    assert set(inter) == set(range(16)) - {4, 5, 6, 7}
    assert plan.peers_for(0, 16, 4, "full") == list(range(16))


def test_payload_and_workspace_sizes():
    assert plan.payload_bytes(2048, 3584, 16) == 117440512
    assert plan.workspace_bytes(16, 2048, 3584) == 117440512
    assert plan.bandwidth_gbps(1_000_000_000, 1000.0) == pytest.approx(1.0)


def test_table_columns():
    cases = [
        {
            "tokens": 1024,
            "mode": mode,
            "peers": peers,
            "bandwidth_gbps": 100.0 + index,
        }
        for index, (mode, peers) in enumerate((("local", 1), ("intra", 3), ("inter", 12), ("full", 16)))
    ]
    table = plan.build_table(cases)
    assert "本卡内" in table and "组内下行" in table and "组间下行" in table
    assert "| 1024 | 100.00 | 101.00 | 102.00 | 103.00 |" in table
    per_link = plan.build_per_link_table(cases)
    assert "33.67" in per_link  # 101 / 3


def _rank_report(rank, world, group, tokens, times):
    return {
        "schema": 1,
        "rank": rank,
        "world_size": world,
        "group_size": group,
        "token_bytes": 3584,
        "warmup": 1,
        "iters": len(times),
        "checked": True,
        "cases": [
            {
                "tokens": tokens,
                "mode": mode,
                "peers": peers,
                "bytes": tokens * 3584 * peers,
                "times_ms": times,
                "median_ms": statistics.median(times),
                "min_ms": min(times),
            }
            for mode, peers in (("local", 1), ("intra", 3), ("inter", 12), ("full", 16))
        ],
    }


def test_aggregate_uses_slowest_rank(tmp_path):
    world, group = 4, 2
    for rank in range(world):
        times = [1.0 + rank, 2.0 + rank]
        path = tmp_path / f"rank_{rank:02d}.json"
        path.write_text(json.dumps(_rank_report(rank, world, group, 1024, times)), encoding="utf-8")
    reports = summarize.load_reports(tmp_path)
    cases = summarize.aggregate(reports)
    local = next(case for case in cases if case["mode"] == "local")
    assert local["max_rank_median_ms"] == pytest.approx(4.5)
    assert local["mean_rank_median_ms"] == pytest.approx(3.0)
    assert local["bandwidth_gbps"] == pytest.approx(1024 * 3584 / 4.5 / 1e6)
    assert local["per_destination_gbps"] == pytest.approx(local["bandwidth_gbps"])


def test_load_reports_rejects_mixed_shapes(tmp_path):
    for rank in range(2):
        report = _rank_report(rank, 2, 1, 1024, [1.0])
        if rank == 1:
            report["group_size"] = 2
        (tmp_path / f"rank_{rank:02d}.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit):
        summarize.load_reports(tmp_path)
