"""CPU tests for the standalone, lossless URMA Chrome-trace text report."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "urma_trace_to_text.py"


@pytest.fixture(scope="module")
def trace_text():
    spec = importlib.util.spec_from_file_location("standalone_urma_trace_text", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVENTS = {
    "Pack": (0, "compute.copy"),
    "DispatchPutX": (1, "transport.submit"),
    "DispatchPutScale": (2, "transport.submit"),
    "DispatchQuiet": (3, "transport.quiet"),
    "WaitLocalDispatch": (8, "wait.sync"),
    "WaitCombineReady": (9, "wait.sync"),
    "CombinePut": (10, "transport.submit"),
    "CombineQuiet": (11, "transport.quiet"),
    "WaitDispatch": (14, "wait.sync"),
    "Gmm1Tile": (15, "cube.submit"),
    "WaitActivation": (18, "wait.sync"),
    "WaitGmm1": (22, "wait.sync"),
    "LocalSync": (26, "wait.sync"),
    "Gmm1Drain": (27, "compute.drain"),
}


def event(name, ts, dur, *, tid=2, rank=0, peer=None, byte_count=0, **metadata):
    event_id, category = EVENTS[name]
    args = dict(
        peer=peer,
        cv=None,
        fragment=None,
        expert=None,
        row=None,
        rows=None,
        tile_m=None,
        tile_n=None,
        columns=None,
        event_id=event_id,
        bytes=byte_count,
        record_index=0,
        start_tick=str(round(ts * 1000)),
        end_tick=str(round((ts + dur) * 1000)),
        elapsed_ticks=round(dur * 1000),
    )
    args.update(metadata)
    suffix = " [PUT submit]" if category == "transport.submit" else ""
    return dict(ph="X", name=name + suffix, cat=category, pid=rank, tid=tid, ts=ts, dur=dur, args=args)


def document(*records):
    records = deepcopy(records)
    counts = [0] * 84
    for record in records:
        engine = record["tid"]
        record["args"]["record_index"] = counts[engine]
        counts[engine] += 1
    rank = records[0]["pid"] if records else 0
    origin = min((record["ts"] for record in records), default=0)
    end = max((record["ts"] + record["dur"] for record in records), default=0)
    metadata = dict(
        schema=1,
        rank=rank,
        engine_count=84,
        capacity_per_engine=4096,
        records_per_engine=counts,
        record_count=len(records),
        dropped=0,
        clock_mhz=1000.0,
        envelope_us=end - origin,
        envelope_ticks=round((end - origin) * 1000),
        events=dict(Counter(record["name"].split(" [", 1)[0] for record in records)),
    )
    return dict(
        metadata=metadata,
        displayTimeUnit="us",
        traceEvents=[dict(ph="M", pid=rank, tid=0, name="process_name", args={"name": f"rank {rank}"}), *records],
    )


def detail_times(text):
    """Read numeric detail rows without depending on metadata field widths."""
    result = []
    detail = text.split("[DETAIL]", 1)[-1] if "[DETAIL]" in text else ""
    for line in detail.splitlines():
        match = re.match(r"^\s*([\d.]+)\s+([\d.]+)\s+", line)
        if match:
            result.append(tuple(map(float, match.groups())))
    return result


def section_rows(text, name):
    result, selected = [], False
    for line in text.splitlines():
        if line.startswith("["):
            selected = line.startswith(f"[{name}]")
        elif selected and line:
            result.append(line.split())
    return result


def keyed_stats(text, prefix):
    line = next(line for line in text.splitlines() if line.startswith(prefix))
    return {name: float(value) for name, value in re.findall(r"(\w+)=([\d.]+)", line)}


def test_simultaneous_engines_have_distinct_aggregate_sum_union_and_span(trace_text):
    doc = document(
        event("DispatchPutX", 0, 5, peer=1, byte_count=64),
        event("DispatchPutX", 0, 5, tid=5, peer=2, byte_count=64),
    )
    result = trace_text.render_report(doc)
    selected = keyed_stats(result, "SELECTED ")
    assert selected["sum_us"] == 10
    assert selected["union_us"] == selected["span_us"] == 5
    for prefix in ("00/AIV1 ", "01/AIV1 "):
        engine = keyed_stats(result, prefix)
        assert engine["sum_us"] == engine["union_us"] == engine["span_us"] == 5
    # id, count, sum, union, p50, p95, max, first, last, payload bytes.
    op = section_rows(result, "OP")
    assert op == [["1", "2", "10.000", "5.000", "5.000", "5.000", "5.000", "0.000", "5.000", "128"]]


def test_no_recorded_events_are_rejected(trace_text):
    with pytest.raises(ValueError):
        trace_text.render_report(document())


@pytest.mark.parametrize("field,value", [("ts", math.nan), ("ts", math.inf), ("dur", math.nan), ("dur", -1)])
def test_nonfinite_or_negative_event_times_are_rejected(trace_text, field, value):
    doc = document(event("Pack", 0, 1))
    doc["traceEvents"][-1][field] = value
    with pytest.raises(ValueError):
        trace_text.render_report(doc)


def test_multi_rank_trace_is_rejected_instead_of_aligning_independent_clocks(trace_text):
    doc = document(event("Pack", 0, 1, rank=0), event("Pack", 0, 1, tid=5, rank=1))
    with pytest.raises(ValueError):
        trace_text.render_report(doc)


def test_source_record_count_mismatch_is_incomplete(trace_text):
    doc = document(event("DispatchPutX", 0, 5, peer=1), event("DispatchPutX", 0, 5, tid=5, peer=2))
    # A prefiltered JSON retains the source's original count metadata.
    doc["traceEvents"].pop()
    assert "source_status=INCOMPLETE" in trace_text.render_report(doc)


def test_declared_wait_filter_is_distinct_from_lost_records(trace_text):
    from deep_ep_experimental.megamoe.urma_trace import decode_trace, filter_trace_waits
    from test_urma_trace import add_record, trace_buffer

    raw = trace_buffer()
    add_record(raw, event="WaitDispatch")
    add_record(raw, event="DispatchPutX", start=1020, end=1040)
    doc = filter_trace_waits(decode_trace(raw, rank=0, clock_mhz=1000))
    assert "source_status=FILTERED_WAITS" in trace_text.render_report(doc)


def test_per_engine_count_mismatch_is_incomplete_even_when_total_matches(trace_text):
    doc = document(event("DispatchPutX", 0, 5, peer=1))
    doc["metadata"]["records_per_engine"][2] = 0
    doc["metadata"]["records_per_engine"][5] = 1
    assert "source_status=INCOMPLETE" in trace_text.render_report(doc)


def test_report_filters_do_not_mark_a_complete_source_as_incomplete(trace_text):
    doc = document(event("DispatchPutX", 0, 5, peer=1), event("Gmm1Tile", 0, 5, tid=3))
    result = trace_text.render_report(doc, cores={0}, engines={"AIV1"}, detail=True)
    assert "source_status=COMPLETE" in result
    assert detail_times(result) == [(0.0, 5.0)]


def test_detail_preserves_original_times_and_wait_visibility_changes_only_detail(trace_text):
    doc = document(
        event("Pack", 100.000125, 0.000125),
        event("WaitGmm1", 101.75, 2.25),
        event("DispatchPutX", 110.5, 1.75, peer=1, byte_count=4096),
    )
    plain = trace_text.render_report(doc, detail=True)
    waits = trace_text.render_report(doc, detail=True, include_waits=True)
    assert detail_times(plain) == [(100.000125, 0.000125), (110.5, 1.75)]
    assert detail_times(waits) == [(100.000125, 0.000125), (101.75, 2.25), (110.5, 1.75)]
    assert "WaitGmm1" in plain  # Default detail hiding must not omit wait statistics.
    assert keyed_stats(plain, "SELECTED ") == keyed_stats(waits, "SELECTED ")
    assert section_rows(plain, "OP") == section_rows(waits, "OP")


def test_detail_keeps_record_order_for_zero_duration_events_at_the_same_timestamp(trace_text):
    doc = document(
        event("DispatchQuiet", 10, 0, peer=1),
        event("DispatchPutX", 10, 0, peer=1, byte_count=64),
    )
    result = trace_text.render_report(doc, detail=True)
    rows = [row for row in section_rows(result, "DETAIL") if row[0] != "@"]
    assert [int(row[2]) for row in rows] == [3, 1]


def test_detail_uses_fixed_metadata_dictionary_order_and_omits_unknown_values(trace_text):
    doc = document(
        event(
            "DispatchPutX",
            0,
            1,
            peer=7,
            byte_count=4096,
            cv=2,
            fragment=3,
            expert=4,
            row=128,
            rows=32,
            tile_m=0,
            tile_n=256,
            columns=128,
        )
    )
    result = trace_text.render_report(doc, detail=True)
    rows = [row for row in section_rows(result, "DETAIL") if row[0] != "@"]
    assert rows[0][3:] == ["p7", "c2", "f3", "e4", "r128", "l32", "m0", "n256", "w128", "b4096"]


def test_report_does_not_mutate_source_or_convert_submit_time_to_bandwidth(trace_text):
    doc = document(
        event("DispatchPutX", 0, 1, peer=7, byte_count=4096),
        event("DispatchPutScale", 1, 1, peer=7, byte_count=128),
        event("DispatchQuiet", 2, 30, peer=7),
        event("CombinePut", 40, 1, peer=7, byte_count=8192),
        event("CombineQuiet", 41, 50, peer=7),
    )
    original = deepcopy(doc)
    result = trace_text.render_report(doc)
    assert doc == original
    assert not re.search(r"\d\s*(?:GB/s|Gbps|GiB/s)", result)
    assert "DispatchPutX" in result and "CombinePut" in result
    # peer, X count/bytes, scale count/bytes, return count/bytes, D/C quiet time.
    assert section_rows(result, "PEER") == [["7", "1", "4096", "1", "128", "1", "8192", "30.000", "50.000"]]


def test_longest_sync_quiet_and_drain_are_ranked_separately(trace_text):
    doc = document(
        event("WaitGmm1", 0, 4),
        event("WaitDispatch", 5, 8),
        event("DispatchQuiet", 20, 11),
        event("CombineQuiet", 40, 13),
        event("Gmm1Drain", 60, 5),
        event("Gmm1Drain", 70, 10),
    )
    result = trace_text.render_report(doc, top=1)
    for category, expected_time, expected_duration, expected_id in [
        ("wait.sync", 5, 8, 14),
        ("transport.quiet", 40, 13, 11),
        ("compute.drain", 70, 10, 27),
    ]:
        rows = section_rows(result, f"LONG {category}")
        assert len(rows) == 1
        assert tuple(map(float, rows[0][1:3])) == (expected_time, expected_duration)
        assert int(rows[0][3]) == expected_id


def test_cli_writes_default_txt_and_preserves_input(trace_text, tmp_path):
    source = tmp_path / "rank0.trace.json"
    content = json.dumps(document(event("Pack", 0, 1)))
    source.write_text(content, encoding="utf-8")
    assert trace_text.main([str(source)]) in (None, 0)
    result = source.with_suffix(".txt")
    assert result.is_file() and "Pack" in result.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == content


def test_cli_refuses_overwriting_source(trace_text, tmp_path):
    source = tmp_path / "trace.json"
    content = json.dumps(document(event("Pack", 0, 1)))
    source.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        trace_text.main([str(source), "--output", str(source)])
    assert source.read_text(encoding="utf-8") == content


def test_standalone_cli_accepts_core_ranges_and_engine_filter(tmp_path):
    source = tmp_path / "trace.json"
    source.write_text(
        json.dumps(document(event("Pack", 0, 1, tid=47), event("Gmm1Tile", 0, 1, tid=45))), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--detail", "--cores", "0,1,15-17", "--engines", "AIV1"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    report = source.with_suffix(".txt").read_text(encoding="utf-8")
    assert "source_status=COMPLETE" in report
    assert detail_times(report) == [(0.0, 1.0)]


@pytest.mark.parametrize("option", [["--cores", "28"], ["--cores", "4-2"], ["--engines", "AIV2"]])
def test_cli_rejects_invalid_core_or_engine_filters_without_writing_report(trace_text, tmp_path, option):
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(document(event("Pack", 0, 1))), encoding="utf-8")
    with pytest.raises(SystemExit) as failure:
        trace_text.main([str(source), *option])
    assert failure.value.code == 2
    assert not source.with_suffix(".txt").exists()
