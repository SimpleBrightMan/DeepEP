import copy
import json
from pathlib import Path

import pytest

from deep_ep_experimental.megamoe.urma_trace import EVENTS, export_trace, filter_trace_waits, decode_trace
from deep_ep_experimental.megamoe.urma_trace_merge import main, merge_trace_collective, merge_trace_files
from test_store import groups, together
from test_urma_trace import add_record, events, trace_buffer


def make_trace():
    raw = trace_buffer(12)
    for index, (name, stage) in enumerate(
        [
            ("WaitDispatch", None),
            ("Gmm1Tile", None),
            ("WaitInputScale", None),
            ("LocalSync", 0),
            ("LocalSync", 1),
            ("FinalPeerHandshake", None),
            ("RankBarrier", None),
            ("LocalSync", 2),
            ("LocalSync", 3),
            ("DispatchQuiet", None),
            ("Gmm2Drain", None),
            ("Reduce", None),
        ]
    ):
        add_record(
            raw,
            event=name,
            start=1000 + index * 100,
            end=1050 + index * 100,
            **({} if stage is None else dict(cv=stage)),
        )
    return raw


def export_ranks(directory, world=2):
    paths, records = [], []
    for rank in range(world):
        path = directory / f"rank_{rank}" / "bs_4096" / "urma_trace.json"
        raw = make_trace()
        # Intentionally different absolute clocks; no rebasing at merge time.
        for index in range(12):
            raw[2, 16 + index * 16 : 18 + index * 16] += rank * 10000000
        metadata = export_trace(raw, path, rank=rank, clock_mhz=1000, include_waits=True)
        paths.append(path)
        records.append(dict(rank=rank, chrome_path=str(path), metadata=metadata, error=None))
    return paths, records


def test_filter_keeps_final_global_sync_and_never_compresses_time():
    complete = decode_trace(make_trace(), rank=0, clock_mhz=1000)
    original = copy.deepcopy(complete)
    filtered = filter_trace_waits(complete)
    rows = events(filtered)
    assert [(row["name"], row["args"].get("sync_stage")) for row in rows] == [
        ("Gmm1Tile [Cube submit]", None),
        ("FinalPeerHandshake", None),
        ("RankBarrier", None),
        ("LocalSync", 2),
        ("LocalSync", 3),
        ("DispatchQuiet [completion wait]", None),
        ("Gmm2Drain [drain]", None),
        ("Reduce", None),
    ]
    assert [row["ts"] for row in rows] == [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    assert complete == original
    assert filtered["metadata"]["record_count"] == 12
    assert filtered["metadata"]["visible_record_count"] == 8
    assert filtered["metadata"]["hidden_wait_count"] == 4
    assert filter_trace_waits(filtered) == filtered


def test_default_export_filters_but_optional_full_export_preserves_all_ids(tmp_path):
    raw = trace_buffer(1)
    for index, name in enumerate(EVENTS):
        if name is None:
            continue
        add_record(raw, engine=index, event=name)
    default, full = tmp_path / "default.json", tmp_path / "full.json"
    export_trace(raw, default, rank=0, clock_mhz=1000)
    export_trace(raw, full, rank=0, clock_mhz=1000, include_waits=True)
    filtered = json.loads(default.read_text())
    assert all(not row["name"].startswith("Wait") for row in events(filtered))
    assert any(row["name"] == "RankBarrier" for row in events(filtered))
    assert len(events(json.loads(full.read_text()))) == sum(name is not None for name in EVENTS)


def test_local_invocation_join_remains_visible_at_the_actual_tail():
    raw = trace_buffer(2)
    add_record(raw, event="Gmm2Tile", start=1000, end=2000)
    add_record(raw, event="LocalSync", cv=3, start=2100, end=9000)
    complete = decode_trace(raw, rank=0, clock_mhz=1000)
    filtered = filter_trace_waits(complete)
    assert max(e["ts"] + e["dur"] for e in events(filtered)) == complete["metadata"]["envelope_us"]


def test_merge_all_16_ranks_preserves_identity_clocks_and_raw_metadata(tmp_path):
    paths, records = export_ranks(tmp_path, 16)
    output = tmp_path / "bs_4096" / "allrank.json"
    result = merge_trace_files(paths, output, expected_metadata=[r["metadata"] for r in records])
    merged = json.loads(output.read_text())
    assert result == merged["metadata"]
    assert result["ranks"] == list(range(16))
    assert result["record_count"] == 16 * 12 and result["visible_record_count"] == 16 * 8
    assert result["hidden_wait_count"] == 16 * 4
    assert len({row["origin_tick"] for row in result["per_rank"]}) == 16
    assert "not synchronized" in result["clock_alignment"]
    rows = events(merged)
    assert {row["pid"] for row in rows} == set(range(16))
    for rank in range(16):
        original = events(filter_trace_waits(json.loads(paths[rank].read_text())))
        assert [row for row in rows if row["pid"] == rank] == original


@pytest.mark.parametrize(
    "fault", ["missing", "wrong_rank", "partial", "dropped", "clock", "identity", "nonfinite", "stale"]
)
def test_bad_merge_does_not_replace_existing_output(tmp_path, fault):
    paths, records = export_ranks(tmp_path)
    trace = json.loads(paths[1].read_text())
    expected = None
    if fault == "missing":
        paths[1].unlink()
    elif fault == "wrong_rank":
        trace["metadata"]["rank"] = 0
    elif fault == "partial":
        trace["traceEvents"].pop()
    elif fault == "dropped":
        trace["metadata"]["dropped"] = 1
    elif fault == "clock":
        trace["metadata"]["clock_mhz"] = 500
    elif fault == "identity":
        trace["traceEvents"][-1]["pid"] = 0
    elif fault == "nonfinite":
        trace["traceEvents"][-1]["ts"] = float("nan")
    else:
        expected = [r["metadata"] for r in records]
        trace["metadata"]["origin_tick"] = "999"
    if fault != "missing":
        paths[1].write_text(json.dumps(trace))
    output = tmp_path / "allrank.json"
    output.write_text("previous successful export")
    with pytest.raises((ValueError, OSError)):
        merge_trace_files(paths, output, world=2, expected_metadata=expected)
    assert output.read_text() == "previous successful export"
    assert not list(tmp_path.glob(".allrank-*"))


def test_filtered_input_merges_without_double_counting_hidden_waits(tmp_path):
    paths, _ = export_ranks(tmp_path)
    for path in paths:
        path.write_text(json.dumps(filter_trace_waits(json.loads(path.read_text()))))
    result = merge_trace_files(paths, tmp_path / "allrank.json", world=2)
    assert result["visible_record_count"] == 16 and result["hidden_wait_count"] == 8


def test_offline_cli_discovers_cases_and_can_keep_waits(tmp_path, capsys):
    export_ranks(tmp_path)
    main([str(tmp_path), "--world-size", "2", "--include-waits"])
    document = json.loads((tmp_path / "bs_4096/allrank.json").read_text())
    assert len(events(document)) == 24
    assert "[URMA ALLRANK]" in capsys.readouterr().out


@pytest.mark.parametrize("shared", [True, False])
def test_collective_merge_shared_or_nonshared_filesystem(tmp_path, monkeypatch, shared):
    paths, records = export_ranks(tmp_path)
    peers = groups(2)
    original = Path.is_file
    if not shared:
        monkeypatch.setattr(Path, "is_file", lambda path: False if path == paths[1] else original(path))
    output = tmp_path / "bs_4096/allrank.json"
    reports = together(peers, lambda group: merge_trace_collective(group, records, output, bs=4096))
    assert reports[0] == reports[1] and reports[0]["error"] is None
    assert reports[0]["metadata"]["rank_count"] == 2
    assert len(events(json.loads(output.read_text()))) == 16
    assert not list(output.parent.glob(".rank-traces-*"))


def test_merge_error_reaches_all_ranks(tmp_path):
    paths, records = export_ranks(tmp_path)
    paths[1].write_text("{broken json")

    def work(group):
        with pytest.raises(RuntimeError, match="allrank export failed"):
            merge_trace_collective(group, records, tmp_path / "allrank.json", bs=4096)

    together(groups(2), work)
    assert not (tmp_path / "allrank.json").exists()
