# Copyright (c) 2026, SimpleBright_Man

from copy import deepcopy
import json

import pytest

from deep_ep_experimental.megamoe import comm_report, wave_oracle
from comm_report_fixtures import capture


def timer_events():
    spec, measured, _, _ = capture(bs=16)
    return spec, comm_report.read_events(measured, spec["rank"])


def test_timer_only_accounts_recorded_rows_without_oracle_or_audit(monkeypatch):
    spec, events = timer_events()

    def forbidden(*args, **kwargs):
        raise AssertionError("Performance reporting must not reconstruct or audit task identities")

    monkeypatch.setattr(wave_oracle, "plan", forbidden)
    monkeypatch.setattr(comm_report, "read_transfers", forbidden)
    # Performance mode accounts actual timer metadata without comparing the
    # total against the balanced routing formula or any expected workload.
    events[0]["row_count"] += 13
    report = comm_report.timer_rank_report(events, spec["rank"], spec["hidden"])
    assert report["validation"]["strict_wave_core"] is False
    assert report["validation"]["status"] == "skipped"
    assert "_transfers" not in report
    for operation in comm_report.OPERATIONS:
        selected = [e for e in events if e["family"] == f"AIV_{operation}_COMM"]
        rows = sum(e["row_count"] for e in selected)
        duration = sum((e["end_cycle"] - e["start_cycle"]) / 1000 for e in selected)
        size = rows * comm_report.payload_bytes(spec["hidden"], operation)
        result = report["operations"][operation]
        assert result["rows"] == rows
        assert result["bytes"] == size
        assert result["sum_core_us"] == duration
        assert result["mean_core_us"] == duration / 28
        assert result["gbps"] == size / (duration / 28) / 1000
    for wave in report["waves"]:
        assert len(wave["cores"]) == 28
        assert wave["rows"] == sum(core["rows"] for core in wave["cores"])
        assert wave["sum_core_us"] == sum(core["duration_us"] for core in wave["cores"])
        assert all("peers" not in core for core in wave["cores"])


def test_timer_only_saves_wave_core_json_without_inventing_transfer_csv(tmp_path):
    spec, events = timer_events()
    report = comm_report.timer_rank_report(events, spec["rank"], spec["hidden"])
    compact = comm_report.write_rank_detail(report, tmp_path)
    assert set(compact["detail_files"]) == {"json"}
    assert not list(tmp_path.glob("*.csv"))
    detail = json.loads((tmp_path / "rank_00_communication.json").read_text())
    assert detail["waves"] == report["waves"]
    assert detail["validation"]["status"] == "skipped"
    assert "waves" not in compact
    assert len(json.dumps(compact, allow_nan=False)) < 10000


def test_timer_only_summary_requires_explicit_opt_out_of_validation():
    spec, events = timer_events()
    report = comm_report.timer_rank_report(events, spec["rank"], spec["hidden"])
    records = [dict(report, rank=rank) for rank in range(16)]
    with pytest.raises(ValueError, match="strict"):
        comm_report.summarize(records)
    summary = comm_report.summarize(records, require_validation=False)
    assert summary["validation"] == dict(strict_wave_core=False, status="skipped")
    assert any("timer metadata" in note for note in summary["notes"])
    assert not any("Exact copy identities" in note for note in summary["notes"])
    for operation in comm_report.OPERATIONS:
        assert summary["operations"][operation]["gbps_per_card_mean"] == report["operations"][operation]["gbps"]


@pytest.mark.parametrize(
    "field,value",
    [("core", 28), ("wave", -1), ("row_count", 0), ("start_cycle", 0), ("audit_enabled", True)],
)
def test_timer_only_rejects_invalid_timer_format(field, value):
    spec, events = timer_events()
    invalid = deepcopy(events)
    invalid[0][field] = value
    with pytest.raises(ValueError):
        comm_report.timer_rank_report(invalid, spec["rank"], spec["hidden"])
