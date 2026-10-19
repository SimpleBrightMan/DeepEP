# Copyright (c) 2026, SimpleBright_Man

"""Rank-local effective bandwidth from completed outer Prefill communication spans."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .timer import protocol as p

SCHEMA_VERSION = 1
REPORT_KIND = "megamoe_prefill_rank_local_bandwidth"
OPERATIONS = ("DISPATCH", "COMBINE")
COHORTS = ("INTRA", "CROSS")
WORKERS = 28
WORLD = 16
FAMILIES = {f"AIV_{operation}_COMM": operation for operation in OPERATIONS}
NOTES = (
    "One-way payload is counted once; metadata, control traffic and local staging are excluded.",
    "INTRA includes self copies and the other three ranks in the local group; it is not a single-link rate.",
    "Envelope uses last completion minus first start on one rank, including gaps with no class span.",
    "Active time is the union of all class spans across cores and waves, excluding only uncovered gaps.",
    "Spans include software and completion waits; active bandwidth is not hardware DMA-active bandwidth.",
    "Each rank is measured independently; device timestamps are never subtracted across ranks.",
    "Dispatch and Combine, and INTRA and CROSS, use different windows; their rates must not be added.",
)


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: expected an integer in the supported range")
    return value


def validate_config(config):
    if not isinstance(config, dict) or type(config.get("enabled")) is not bool:
        raise ValueError("Prefill bandwidth requires the native plan's actual optimization configuration")
    if type(config.get("timer_schema_version")) is not int or config["timer_schema_version"] != p.TIMER_SCHEMA_VERSION:
        raise ValueError("Prefill configuration uses a stale timer schema; rebuild and reinstall")
    if config["enabled"]:
        expected = dict(
            dispatch_intra_cores=4,
            combine_intra_cores=4,
            dispatch_buffer_count=6,
            combine_buffer_count=8,
            group_order="paired-xor",
            dispatch_metadata_mode="dma",
        )
        if any(config.get(key) != value or type(config[key]) is not type(value) for key, value in expected.items()):
            raise ValueError("Prefill bandwidth classification requires the actual DMA/4:24/XOR/D6/C8 configuration")
    return dict(config)


def payload_bytes(hidden, operation):
    _integer(hidden, "hidden", 64, 8192)
    if hidden % 64:
        raise ValueError("Hidden size must be divisible by 64")
    if operation == "DISPATCH":
        return ((hidden + 255) // 256) * 256 + ((hidden // 32 + 31) // 32) * 32
    if operation == "COMBINE":
        return hidden * 2
    raise ValueError("Unknown communication operation")


def read_events(raw, rank):
    """Read only outer COMM records; nested timers never contribute rows or time."""
    _integer(rank, "rank", 0, WORLD - 1)
    p.validate_timer_schema(raw, required_cores=[core * 3 + 2 for core in range(WORKERS)])
    events = []
    for hardware_core in range(p.N_CORE_COUNT):
        for family in FAMILIES:
            kind = p.DYNAMIC_NAMES.index(family)
            count = int(raw[p.N_TIMING_COUNTER + hardware_core * p.DYNAMIC_ITER_PER_CORE + kind])
            if not 0 <= count <= p.MAX_DYNAMIC_ITER:
                raise ValueError("Outer communication timer count overflow/invalid count")
            if count and (hardware_core % 3 != 2 or hardware_core // 3 >= WORKERS):
                raise ValueError("Outer communication events must belong to one of the 28 AIV1 workers")
            for item in range(count):
                slot = len(p.FIXED_NAMES) + kind * p.MAX_DYNAMIC_ITER + item
                offset = hardware_core * p.COUNTERS_PER_CORE + slot * 2
                task = p.task_metadata(raw, hardware_core, slot)
                if task.get("granularity") != "communication_slice":
                    raise ValueError("Outer communication event is missing actual row metadata")
                events.append(
                    dict(
                        rank=rank,
                        core=hardware_core // 3,
                        operation=FAMILIES[family],
                        wave=int(raw[p.WAVE_ID_OFFSET + hardware_core * p.WAVE_ID_PER_CORE + slot]),
                        start_cycle=int(raw[offset]),
                        end_cycle=int(raw[offset + 1]),
                        row_count=int(task["row_count"]),
                        expert_id=int(task["expert_id"]),
                        row_start=int(task["row_start"]),
                    )
                )
    return events


def _measure(events, row_bytes, workers):
    rows = sum(event["row_count"] for event in events)
    size = rows * row_bytes
    if not events:
        return dict(
            status="no_traffic",
            workers=workers,
            rows=0,
            payload_bytes=0,
            spans=0,
            envelope_us=0.0,
            active_us=0.0,
            coverage=None,
            envelope_gbps=None,
            active_gbps=None,
        )
    intervals = sorted((event["start_cycle"], event["end_cycle"]) for event in events)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    envelope_cycles = merged[-1][1] - merged[0][0]
    active_cycles = sum(end - start for start, end in merged)
    envelope_us, active_us = (value / p.CYCLE_TO_US for value in (envelope_cycles, active_cycles))
    return dict(
        status="ok",
        workers=workers,
        rows=rows,
        payload_bytes=size,
        spans=len(events),
        envelope_us=envelope_us,
        active_us=active_us,
        coverage=active_cycles / envelope_cycles,
        envelope_gbps=size / envelope_us / 1000,
        active_gbps=size / active_us / 1000,
    )


def from_events(events, rank, hidden, config):
    """Classify fixed core ownership and merge intervals across every core and wave."""
    _integer(rank, "rank", 0, 127)
    actual = validate_config(config)
    report = dict(
        schema_version=SCHEMA_VERSION,
        kind=REPORT_KIND,
        rank=rank,
        config=actual,
        timestamp_unit="system_cycle",
        cycles_per_us=p.CYCLE_TO_US,
        time_unit="us",
        bandwidth_unit="GB/s",
        hidden_size=hidden,
    )
    if not actual["enabled"]:
        if events:
            raise ValueError("Disabled Prefill optimization cannot classify communication core ownership")
        return dict(report, status="unavailable", reason="optimized_prefill_disabled", operations={})
    _integer(rank, "optimized rank", 0, WORLD - 1)
    by_lane, by_class = defaultdict(list), defaultdict(list)
    for event in events:
        _integer(event.get("rank"), "event rank", 0, WORLD - 1)
        if event.get("rank") != rank or event.get("operation") not in OPERATIONS:
            raise ValueError("Unexpected communication event rank or operation")
        core = _integer(event.get("core"), "communication core", 0, WORKERS - 1)
        for name in ("wave", "expert_id", "row_start"):
            _integer(event.get(name), name)
        start = _integer(event.get("start_cycle"), "start cycle", 1, (1 << 63) - 1)
        end = _integer(event.get("end_cycle"), "end cycle", 1, (1 << 63) - 1)
        _integer(event.get("row_count"), "row count", 1, (1 << 32) - 1)
        if end <= start:
            raise ValueError("Communication interval must have positive completed duration")
        operation = event["operation"]
        by_lane[core].append((start, end))
        split = actual[f"{operation.lower()}_intra_cores"]
        by_class[(operation, "INTRA" if core < split else "CROSS")].append(event)
    for intervals in by_lane.values():
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("Overlapping outer communication spans on one AIV1 would double-count payload")
    operations = {}
    for operation in OPERATIONS:
        row_bytes = payload_bytes(hidden, operation)
        split = actual[f"{operation.lower()}_intra_cores"]
        operations[operation] = {
            name: _measure(by_class[(operation, name)], row_bytes, split if name == "INTRA" else WORKERS - split)
            for name in COHORTS
        }
    return dict(report, status="ok", operations=operations)


def rank_report(raw, rank, hidden, config):
    actual = validate_config(config)
    if not actual["enabled"]:
        return from_events([], rank, hidden, actual)
    return from_events(read_events(raw, rank), rank, hidden, actual)


def _statistics(values):
    if not values:
        return dict(mean=None, min=None, max=None)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite Prefill bandwidth summary value")
    return dict(mean=statistics.mean(values), min=min(values), max=max(values))


def summarize(records, expected_ranks, world=WORLD):
    _integer(world, "EP world", 1)
    ranks = list(expected_ranks)
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("Expected timer ranks must be nonempty and unique")
    for rank in ranks:
        _integer(rank, "expected rank", 0, world - 1)
    observed = [record.get("rank") for record in records]
    if len(set(observed)) != len(observed) or set(observed) != set(ranks):
        raise ValueError("Missing, extra or duplicate timer rank reports")
    first = records[0]
    for record in records:
        _integer(record.get("rank"), "report rank", 0, world - 1)
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != SCHEMA_VERSION
            or record.get("kind") != REPORT_KIND
        ):
            raise ValueError("Stale Prefill bandwidth report schema")
        validate_config(record.get("config"))
        if record["config"] != first["config"]:
            raise ValueError("Ranks disagree on actual Prefill optimization configuration")
        if record.get("hidden_size") != first.get("hidden_size"):
            raise ValueError("Ranks disagree on hidden size")
        if (
            record.get("timestamp_unit") != "system_cycle"
            or record.get("cycles_per_us") != p.CYCLE_TO_US
            or record.get("time_unit") != "us"
            or record.get("bandwidth_unit") != "GB/s"
        ):
            raise ValueError("Incompatible Prefill bandwidth units")
        if record.get("status") != ("ok" if record["config"]["enabled"] else "unavailable"):
            raise ValueError("Prefill report status does not match the actual configuration")
    result = dict(
        schema_version=SCHEMA_VERSION,
        kind=REPORT_KIND,
        config=first["config"],
        hidden_size=first["hidden_size"],
        status=first["status"],
        world_size=world,
        measured_ranks=sorted(ranks),
        measured_rank_count=len(ranks),
        all_ranks_measured=len(ranks) == world,
        time_unit="us",
        bandwidth_unit="GB/s",
        notes=list(NOTES),
        ranks=sorted(records, key=lambda record: record["rank"]),
        operations={},
    )
    if not first["config"]["enabled"]:
        return dict(result, reason="optimized_prefill_disabled")
    if world != WORLD:
        raise ValueError("Optimized Prefill bandwidth requires EP16")
    for operation in OPERATIONS:
        result["operations"][operation] = {}
        for cohort in COHORTS:
            values = [record["operations"][operation][cohort] for record in records]
            active = [value for value in values if value["status"] == "ok"]
            if any(value["status"] not in ("ok", "no_traffic") for value in values):
                raise ValueError("Invalid Prefill cohort status")
            for value in values:
                for key in ("rows", "payload_bytes", "spans"):
                    _integer(value[key], key)
                if value["status"] == "no_traffic":
                    if any(value[key] != 0 for key in ("rows", "payload_bytes", "spans", "envelope_us", "active_us")):
                        raise ValueError("No-traffic cohort contains nonzero measurements")
                    if any(value[key] is not None for key in ("coverage", "envelope_gbps", "active_gbps")):
                        raise ValueError("No-traffic cohort must not invent a bandwidth denominator")
                elif not (
                    0 < value["active_us"] <= value["envelope_us"]
                    and value["rows"] > 0
                    and value["payload_bytes"] > 0
                    and 0 < value["coverage"] <= 1
                    and value["envelope_gbps"] > 0
                    and value["active_gbps"] >= value["envelope_gbps"]
                ):
                    raise ValueError("Invalid positive-traffic cohort measurements")
                if value["status"] == "ok":
                    expected_bytes = value["rows"] * payload_bytes(first["hidden_size"], operation)
                    if value["payload_bytes"] != expected_bytes or not all(
                        math.isclose(value[key], expected, rel_tol=1e-12, abs_tol=1e-12)
                        for key, expected in (
                            ("coverage", value["active_us"] / value["envelope_us"]),
                            ("envelope_gbps", expected_bytes / value["envelope_us"] / 1000),
                            ("active_gbps", expected_bytes / value["active_us"] / 1000),
                        )
                    ):
                        raise ValueError("Prefill bytes, time units and bandwidth are inconsistent")
            result["operations"][operation][cohort] = dict(
                workers=first["config"][f"{operation.lower()}_intra_cores"]
                if cohort == "INTRA"
                else WORKERS - first["config"][f"{operation.lower()}_intra_cores"],
                traffic_rank_count=len(active),
                no_traffic_ranks=[
                    record["rank"] for record, value in zip(records, values) if value["status"] == "no_traffic"
                ],
                **{key: _statistics([value[key] for value in values]) for key in ("payload_bytes", "rows")},
                **{
                    key: _statistics([value[key] for value in active])
                    for key in ("envelope_us", "active_us", "coverage", "envelope_gbps", "active_gbps")
                },
            )
    return result


def write_report(report, directory, case_label):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prefill_bandwidth.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    prefix = f"[PREFILL-BW] {case_label} ranks={report['measured_rank_count']}/{report['world_size']}"
    if report["status"] != "ok":
        lines = [f"{prefix} unavailable: optimized Prefill path is disabled; no core-class rates reported."]
    else:
        lines = [f"{prefix} one-way payload; INTRA includes self; rank-local envelope and interval union."]
        for operation in OPERATIONS:
            for cohort in COHORTS:
                value = report["operations"][operation][cohort]
                if not value["traffic_rank_count"]:
                    lines.append(f"[PREFILL-BW] {operation} {cohort} no traffic; bandwidth=N/A")
                    continue
                envelope, active, coverage = (value[key] for key in ("envelope_gbps", "active_gbps", "coverage"))
                lines.append(
                    f"[PREFILL-BW] {operation} {cohort} GB/s/card "
                    f"envelope mean={envelope['mean']:.3f} rank-min/max={envelope['min']:.3f}/{envelope['max']:.3f}; "
                    f"active-union mean={active['mean']:.3f} rank-min/max={active['min']:.3f}/{active['max']:.3f}; "
                    f"coverage mean={coverage['mean']:.3%} rank-min/max={coverage['min']:.3%}/{coverage['max']:.3%}; "
                    f"traffic-ranks={value['traffic_rank_count']}/{report['measured_rank_count']} workers={value['workers']}"
                )
    text = "\n".join(lines) + "\n"
    (directory / "prefill_bandwidth.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
