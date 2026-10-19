# Copyright (c) 2026, SimpleBright_Man

from copy import deepcopy
import json

import pytest

from deep_ep_experimental.megamoe import prefill_report as report
from deep_ep_experimental.megamoe.timer import protocol as p


def config(enabled=True):
    return dict(
        enabled=enabled,
        requested=enabled,
        dispatch_intra_cores=4 if enabled else 0,
        combine_intra_cores=4 if enabled else 0,
        dispatch_buffer_count=6,
        combine_buffer_count=8 if enabled else 2,
        group_order="paired-xor" if enabled else "original",
        dispatch_metadata_mode="dma" if enabled else "scalar",
        timer_schema_version=p.TIMER_SCHEMA_VERSION,
    )


def event(core=0, start=1000, end=2000, rows=10, operation="DISPATCH", rank=0, wave=0):
    return dict(
        core=core,
        start_cycle=start,
        end_cycle=end,
        row_count=rows,
        operation=operation,
        rank=rank,
        wave=wave,
        expert_id=0,
        row_start=0,
    )


class SparseRaw(dict):
    """Exercise the actual binary offsets without allocating a 283 MB timer."""

    def __len__(self):
        return p.TOTAL_BUFFER_SIZE

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.get(index, 0)


def raw_timer():
    raw = SparseRaw()
    for core in range(28):
        raw[p.TIMER_SCHEMA_OFFSET + core * 3 + 2] = p.TIMER_SCHEMA_VERSION
        raw[p.TIMER_COMPLETION_OFFSET + core * 3 + 2] = 1
    return raw


def put_event(raw, value, family=None, hardware=None, tile=-1):
    family = family or f"AIV_{value['operation']}_COMM"
    kind = p.DYNAMIC_NAMES.index(family)
    hardware = value["core"] * 3 + 2 if hardware is None else hardware
    counter = p.N_TIMING_COUNTER + hardware * p.DYNAMIC_ITER_PER_CORE + kind
    iteration = raw[counter]
    slot = len(p.FIXED_NAMES) + kind * p.MAX_DYNAMIC_ITER + iteration
    timestamp = hardware * p.COUNTERS_PER_CORE + slot * 2
    raw[timestamp], raw[timestamp + 1] = value["start_cycle"], value["end_cycle"]
    metadata = (
        p.TASK_METADATA_OFFSET
        + (hardware * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + kind * p.MAX_DYNAMIC_ITER + iteration) * 2
    )
    raw[metadata] = ((value["expert_id"] + 1) << 32) | value["row_start"]
    raw[metadata + 1] = ((tile + 1) << 32) | value["row_count"]
    raw[p.WAVE_ID_OFFSET + hardware * p.WAVE_ID_PER_CORE + slot] = value["wave"]
    raw[counter] += 1
    raw[p.TIMER_SCHEMA_OFFSET + hardware] = p.TIMER_SCHEMA_VERSION
    raw[p.TIMER_COMPLETION_OFFSET + hardware] = 1


def test_union_merges_cross_core_cross_wave_overlap_and_keeps_uncovered_gaps():
    events = [
        event(0, 1000, 3000, rows=10),
        event(1, 2000, 4000, rows=20, wave=1),
        event(2, 2200, 2300, rows=1, wave=3),
        event(3, 4000, 5000, rows=10),
        event(0, 7000, 8000, rows=10, wave=2),
    ]
    value = report.from_events(events, 0, 3584, config())["operations"]["DISPATCH"]["INTRA"]
    assert value["rows"] == 51
    assert value["payload_bytes"] == 51 * 3712  # FP8 tokens + padded scale, no metadata/staging.
    assert value["envelope_us"] == 7
    assert value["active_us"] == 5  # Not sum(core durations)=6.1, nor their mean.
    assert value["coverage"] == 5 / 7
    assert value["envelope_gbps"] == 51 * 3712 / 7 / 1000
    assert value["active_gbps"] == 51 * 3712 / 5 / 1000


def test_dispatch_combine_and_classes_have_independent_windows():
    events = [
        event(),
        event(4, 1000, 6000),
        event(0, 8000, 18000, operation="COMBINE"),
        event(4, 8000, 28000, operation="COMBINE"),
    ]
    operations = report.from_events(events, 0, 3584, config())["operations"]
    assert operations["DISPATCH"]["INTRA"]["envelope_us"] == 1
    assert operations["DISPATCH"]["CROSS"]["envelope_us"] == 5
    assert operations["COMBINE"]["INTRA"]["envelope_us"] == 10
    assert operations["COMBINE"]["CROSS"]["envelope_us"] == 20
    assert operations["COMBINE"]["CROSS"]["payload_bytes"] == 10 * 3584 * 2
    assert operations["DISPATCH"]["CROSS"]["workers"] == 24


def test_rank_clocks_are_independent_and_subset_coverage_is_explicit():
    first = report.from_events([event()], 0, 3584, config())
    second = report.from_events([event(start=10**15, end=10**15 + 2000, rank=7)], 7, 3584, config())
    result = report.summarize([second, first], [0, 7])
    value = result["operations"]["DISPATCH"]["INTRA"]
    assert result["measured_ranks"] == [0, 7]
    assert result["measured_rank_count"] == 2 and not result["all_ranks_measured"]
    assert value["envelope_us"] == dict(mean=1.5, min=1.0, max=2.0)
    assert value["envelope_gbps"] == pytest.approx(dict(mean=27.84, min=18.56, max=37.12))


def test_zero_traffic_has_no_invented_denominator_or_zero_bandwidth():
    empty = report.rank_report(raw_timer(), 0, 3584, config())
    full = report.from_events([event(rank=1)], 1, 3584, config())
    result = report.summarize([empty, full], [0, 1])
    value = result["operations"]["DISPATCH"]["INTRA"]
    assert value["no_traffic_ranks"] == [0] and value["traffic_rank_count"] == 1
    assert value["payload_bytes"] == dict(mean=18560, min=0, max=37120)
    assert value["envelope_gbps"]["mean"] == 37.12  # Mean over traffic ranks, not empty ranks.
    missing = result["operations"]["COMBINE"]["CROSS"]
    assert missing["coverage"] == missing["active_gbps"] == dict(mean=None, min=None, max=None)


def test_disabled_path_never_misclassifies_baseline_or_decode_as_4_24():
    result = report.rank_report(None, 31, 7168, config(False))
    assert result["status"] == "unavailable" and result["operations"] == {}
    summary = report.summarize([result], [31], world=32)
    assert summary["reason"] == "optimized_prefill_disabled"
    with pytest.raises(ValueError, match="cannot classify"):
        report.from_events([event()], 0, 3584, config(False))


def test_actual_rows_from_outer_metadata_and_nested_spans_are_ignored():
    raw = raw_timer()
    outer = event(rows=13, start=1000, end=5000, wave=9)
    outer.update(row_start=192, expert_id=11)
    put_event(raw, outer)
    put_event(raw, event(rows=999, start=2000, end=3000), family="AIV_DISPATCH")
    assert report.read_events(raw, 0) == [outer]
    value = report.rank_report(raw, 0, 3584, config())["operations"]["DISPATCH"]["INTRA"]
    assert value["rows"] == 13 and value["spans"] == 1 and value["envelope_us"] == 4


@pytest.mark.parametrize(
    "key,value",
    [
        ("core", 28),
        ("core", True),
        ("rank", 1),
        ("operation", "WAIT"),
        ("wave", -1),
        ("expert_id", -1),
        ("row_start", -1),
        ("row_count", 0),
        ("row_count", -1),
        ("row_count", 2**32),
        ("start_cycle", 0),
        ("start_cycle", 1000.0),
        ("start_cycle", float("nan")),
        ("end_cycle", 1000),
        ("end_cycle", 2**63),
    ],
)
def test_invalid_events_are_rejected(key, value):
    bad = event()
    bad[key] = value
    with pytest.raises(ValueError):
        report.from_events([bad], 0, 3584, config())


def test_duplicate_or_overlapping_same_worker_outer_spans_are_rejected():
    for second in (event(), event(start=1500, operation="COMBINE")):
        with pytest.raises(ValueError, match="double-count"):
            report.from_events([event(), second], 0, 3584, config())


@pytest.mark.parametrize(
    "mutation",
    ["old_schema", "incomplete", "missing_worker", "no_stamp", "overflow", "lane", "tile", "missing_metadata"],
)
def test_raw_protocol_rejects_stale_incomplete_and_invalid_records(mutation):
    raw = raw_timer()
    put_event(raw, event())
    if mutation == "old_schema":
        raw[p.TIMER_SCHEMA_OFFSET + 2] = 99
    elif mutation == "incomplete":
        raw[p.TIMER_COMPLETION_OFFSET + 2] = 0
    elif mutation == "missing_worker":
        raw[p.TIMER_SCHEMA_OFFSET + 83] = raw[p.TIMER_COMPLETION_OFFSET + 83] = 0
    elif mutation == "no_stamp":
        raw[p.TIMER_SCHEMA_OFFSET + 2] = raw[p.TIMER_COMPLETION_OFFSET + 2] = 0
    elif mutation == "overflow":
        raw[p.N_TIMING_COUNTER + 2 * p.DYNAMIC_ITER_PER_CORE + p.DYNAMIC_NAMES.index("AIV_DISPATCH_COMM")] = (
            p.MAX_DYNAMIC_ITER + 1
        )
    elif mutation == "lane":
        put_event(raw, event(), hardware=1)
    elif mutation == "tile":
        put_event(raw, event(start=3000, end=4000), tile=0)
    elif mutation == "missing_metadata":
        raw[p.TASK_METADATA_OFFSET + (2 * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + 16 * p.MAX_DYNAMIC_ITER) * 2] = 0
    with pytest.raises(ValueError):
        report.rank_report(raw, 0, 3584, config())


def test_empty_and_short_unstamped_timer_rejected():
    with pytest.raises(ValueError, match="size"):
        p.validate_timer_schema([])
    with pytest.raises(ValueError, match="no completed"):
        p.validate_timer_schema(SparseRaw())


@pytest.mark.parametrize(
    "hidden,dispatch,combine", [(64, 288, 128), (3584, 3712, 7168), (3712, 3968, 7424), (7680, 7936, 15360)]
)
def test_payload_bytes_are_wire_granularity_without_double_count(hidden, dispatch, combine):
    assert report.payload_bytes(hidden, "DISPATCH") == dispatch
    assert report.payload_bytes(hidden, "COMBINE") == combine


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 0),
        ("cycles_per_us", 1),
        ("timestamp_unit", "us"),
        ("time_unit", "ms"),
        ("bandwidth_unit", "GiB/s"),
        ("status", "unavailable"),
    ],
)
def test_summary_rejects_stale_reports_and_bad_units(field, value):
    value_report = report.from_events([event()], 0, 3584, config())
    value_report[field] = value
    with pytest.raises(ValueError):
        report.summarize([value_report], [0])


@pytest.mark.parametrize(
    "field,value", [("payload_bytes", 999), ("envelope_gbps", 999), ("active_us", 2), ("coverage", float("nan"))]
)
def test_summary_rejects_inconsistent_or_nonfinite_measurements(field, value):
    value_report = report.from_events([event()], 0, 3584, config())
    value_report["operations"]["DISPATCH"]["INTRA"][field] = value
    with pytest.raises(ValueError):
        report.summarize([value_report], [0])


def test_missing_duplicate_rank_and_disagreeing_config_rejected():
    first = report.from_events([event()], 0, 3584, config())
    for records, ranks in (([first], [0, 1]), ([first, first], [0]), ([first], [0, 0])):
        with pytest.raises(ValueError):
            report.summarize(records, ranks)
    second = deepcopy(first)
    second.update(rank=1, hidden_size=4096)
    with pytest.raises(ValueError, match="hidden size"):
        report.summarize([first, second], [0, 1])
    with pytest.raises(ValueError, match="EP16"):
        report.summarize([first], [0], world=32)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timer_schema_version", 0),
        ("dispatch_intra_cores", 12),
        ("combine_intra_cores", 16),
        ("group_order", "alltoall"),
        ("combine_buffer_count", 6),
        ("enabled", 1),
    ],
)
def test_actual_configuration_required_for_classification(field, value):
    actual = config()
    actual[field] = value
    with pytest.raises(ValueError):
        report.from_events([], 0, 3584, actual)


def test_report_files_and_printed_coverage_are_json_safe(tmp_path, capsys):
    result = report.summarize([report.from_events([event()], 0, 3584, config())], [0])
    report.write_report(result, tmp_path, "BS=4096")
    assert json.loads((tmp_path / "prefill_bandwidth.json").read_text())["measured_ranks"] == [0]
    output = capsys.readouterr().out
    assert "ranks=1/16" in output and "coverage mean=100.000%" in output
    assert "COMBINE CROSS no traffic; bandwidth=N/A" in output
    assert output == (tmp_path / "prefill_bandwidth.txt").read_text()
