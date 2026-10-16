# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Validate concurrent intra-group and staggered cross-group worker timers."""

import csv
import json
import math
from pathlib import Path
import statistics

if __package__:
    from . import report
else:
    import report

ReportError = report.ReportError
distribution = report.distribution
load_reports = report.load_reports
TRAFFIC = report.TRAFFIC
WORKER_COUNTS = {"intra": 12, "cross": 16}
CLASSES = tuple(WORKER_COUNTS)
COMMON_FIELDS = report.COMMON_FIELDS + ("phase_count",)


def _require(condition, message):
    if not condition:
        raise ReportError(message)


def peer_rows(raw):
    """Count the source-prefixed global route stream, independently of task order."""
    modulus = raw["experts"] if raw["scope"] == "first-expert" else raw["world"]
    items = raw["batch"] * raw["topk"]
    rank = raw["rank"]
    return [
        ((peer + 1) * items - 1 - rank) // modulus - (peer * items - 1 - rank) // modulus
        for peer in range(raw["world"])
    ]


def _traffic(rank, peer):
    return "local" if peer == rank else "group" if peer // 4 == rank // 4 else "cross"


def _plan(raw):
    counts = peer_rows(raw)
    groups = [(raw["rank"] // 4 - offset) % 4 for offset in range(4)]
    phases = []
    for group in groups:
        traffic_rows = dict.fromkeys(TRAFFIC, 0)
        for peer in range(group * 4, group * 4 + 4):
            traffic_rows[_traffic(raw["rank"], peer)] += counts[peer]
        phases.append(traffic_rows)
    traffic_rows = {traffic: sum(phase[traffic] for phase in phases) for traffic in TRAFFIC}
    intra_base, intra_extra = divmod(sum(phases[0].values()), 12)
    logical_counts = [intra_base + int(logical < intra_extra) for logical in range(12)] + [0] * 16
    extra_cursor = 0
    for phase in phases[1:]:
        base, extra = divmod(sum(phase.values()), 16)
        for logical in range(16):
            chunk = (logical - extra_cursor) % 16
            logical_counts[12 + logical] += base + int(chunk < extra)
        extra_cursor = (extra_cursor + extra) % 16
    return groups, phases, traffic_rows, logical_counts


def _logical_worker(physical, rotation):
    if physical < 12:
        return (physical + rotation) % 12
    return 12 + (physical - 12 + rotation) % 16


def _class_rows(raw, traffic):
    return (
        (raw["traffic_rows"]["local"] + raw["traffic_rows"]["group"])
        if traffic == "intra"
        else raw["traffic_rows"]["cross"]
    )


def _envelope(raw, workers):
    active = [worker for worker in workers if worker["rows"]]
    return (
        (
            (max(worker["end_cycle"] for worker in active) - min(worker["start_cycle"] for worker in active))
            / raw["cycles_per_us"]
        )
        if active
        else None
    )


def validate_report(raw, *, expected_iterations=None):
    _require(isinstance(raw, dict), "Report must be a JSON object")
    required = set(COMMON_FIELDS) | {
        "schema_version",
        "mode",
        "rank",
        "device",
        "cycles_per_us",
        "traffic_rows",
        "validation",
        "rounds",
        "phase_peer_groups",
        "phase_traffic_rows",
    }
    _require(not required - set(raw), "Missing group report field(s): " + ", ".join(sorted(required - set(raw))))
    _require(
        type(raw["schema_version"]) is int and raw["schema_version"] == 5 and raw["mode"] == "group_round_robin",
        "Unsupported group report schema/mode",
    )
    for name in (
        "world",
        "cores",
        "group_size",
        "batch",
        "hidden",
        "intermediate",
        "experts",
        "topk",
        "row_bytes",
        "scale_elements",
        "buffer_count",
        "repeats",
        "phase_count",
    ):
        report._integer(raw[name], name, 1)
    report._integer(raw["rank"], "rank")
    report._integer(raw["device"], "device")
    _require(
        (raw["world"], raw["group_size"], raw["cores"], raw["buffer_count"], raw["phase_count"]) == (16, 4, 28, 6, 4),
        "Group ordering requires EP16/group4/28 workers/6 slots/4 groups",
    )
    _require(raw["rank"] < 16, "rank must be within world")
    report._mapping(raw["worker_counts"], CLASSES, "worker_counts")
    for traffic in CLASSES:
        report._integer(raw["worker_counts"][traffic], f"worker_counts.{traffic}", 1)
    _require(raw["worker_counts"] == WORKER_COUNTS, "Group ordering requires 12 INTRA and 16 CROSS workers")
    _require(
        raw["batch"] <= 65536
        and raw["experts"] % 16 == 0
        and raw["experts"] // 16 <= 128
        and raw["topk"] <= min(32, raw["experts"]),
        "Invalid group routing dimensions",
    )
    _require(
        1024 <= raw["hidden"] <= 8192
        and raw["hidden"] % 64 == 0
        and 256 <= raw["intermediate"] <= 4096
        and raw["intermediate"] % 128 == 0,
        "Invalid hidden/intermediate dimensions",
    )
    _require(
        raw["scope"] in ("all", "first-expert") and raw["metadata_mode"] in ("scalar", "dma", "preload"),
        "Invalid scope/metadata_mode",
    )
    scale = raw["hidden"] // 32
    padded = (raw["hidden"] + 255) // 256 * 256 + (scale + 31) // 32 * 32
    _require(raw["scale_elements"] == scale and raw["row_bytes"] == padded, "Invalid padded source row size")
    report._positive(raw["cycles_per_us"], "cycles_per_us")
    _require(raw["cycles_per_us"] == 1000, "Group timer conversion requires 1000 cycles/us on Ascend950")
    groups, phases, rows, logical_counts = _plan(raw)
    _require(
        isinstance(raw["phase_peer_groups"], list) and all(type(group) is int for group in raw["phase_peer_groups"]),
        "Invalid source-group order",
    )
    _require(raw["phase_peer_groups"] == groups, "Incorrect GET source-group rotation")
    _require(
        isinstance(raw["phase_traffic_rows"], list) and len(raw["phase_traffic_rows"]) == 4,
        "Expected four source-group row descriptions",
    )
    for phase in raw["phase_traffic_rows"]:
        report._mapping(phase, TRAFFIC, "phase_traffic_rows")
        for traffic in TRAFFIC:
            report._integer(phase[traffic], f"phase_traffic_rows.{traffic}")
    _require(raw["phase_traffic_rows"] == phases, "Incorrect per-group traffic rows")
    report._mapping(raw["traffic_rows"], TRAFFIC, "traffic_rows")
    for traffic in TRAFFIC:
        report._integer(raw["traffic_rows"][traffic], f"traffic_rows.{traffic}")
    _require(raw["traffic_rows"] == rows, "traffic_rows disagree with balanced source-prefixed routing")
    validation = raw["validation"]
    _require(isinstance(validation, dict) and validation.get("passed") is True, "Native output validation failed")
    report._integer(validation.get("checked_rows"), "validation.checked_rows")
    _require(validation["checked_rows"] == sum(rows.values()), "Validation does not cover all requested rows")
    rounds = raw["rounds"]
    _require(isinstance(rounds, list) and rounds, "rounds must be nonempty")
    if expected_iterations is not None:
        _require(len(rounds) == expected_iterations, "Incorrect measured iteration count")
    seen, rotations = [], {}
    for measured in rounds:
        _require(
            isinstance(measured, dict)
            and {"iteration", "worker_rotation", "kernel_elapsed_us", "timer_envelope_us", "workers"} <= set(measured),
            "Malformed group round",
        )
        iteration = report._integer(measured["iteration"], "iteration")
        rotation = report._integer(measured["worker_rotation"], "worker_rotation")
        _require(rotation < 48, "worker_rotation must be in [0,47]")
        seen.append(iteration)
        rotations[iteration] = rotation
        report._positive(measured["kernel_elapsed_us"], "kernel_elapsed_us")
        report._positive(measured["timer_envelope_us"], "timer_envelope_us")
        workers = measured["workers"]
        _require(isinstance(workers, list) and len(workers) == 28, "Each round requires exactly 28 continuous timers")
        physicals = []
        for worker in workers:
            _require(
                isinstance(worker, dict) and {"group_id", "traffic", "rows", "start_cycle", "end_cycle"} <= set(worker),
                "Malformed worker record",
            )
            physical = report._integer(worker["group_id"], "group_id")
            _require(physical < 28, "Physical worker must be within 28 workers")
            _require(
                worker["traffic"] == ("intra" if physical < 12 else "cross"),
                "Worker traffic must match fixed 12 INTRA and 16 CROSS partition",
            )
            physicals.append(physical)
            amount = report._integer(worker["rows"], "worker.rows")
            _require(
                amount == logical_counts[_logical_worker(physical, rotation)] * raw["repeats"],
                "Worker rows disagree with rotated per-group chunk ownership",
            )
            start = report._integer(worker["start_cycle"], "start_cycle")
            end = report._integer(worker["end_cycle"], "end_cycle")
            _require(end >= start and (not amount or (start > 0 and end > start)), "Invalid worker timer interval")
        _require(sorted(physicals) == list(range(28)), "Missing or duplicate physical worker")
        _require(
            sum(worker["rows"] for worker in workers) == sum(rows.values()) * raw["repeats"],
            "Worker row conservation failed",
        )
        envelope = _envelope(raw, workers)
        _require(
            envelope is not None and math.isclose(envelope, measured["timer_envelope_us"], rel_tol=1e-9, abs_tol=1e-6),
            "Reported timer envelope disagrees with the 28 raw intervals",
        )
        tolerance = max(2.0, measured["kernel_elapsed_us"] * 0.1)
        _require(
            envelope <= measured["kernel_elapsed_us"] + tolerance,
            "Timer envelope exceeds ACL kernel duration plus tolerance",
        )
    _require(sorted(seen) == list(range(len(rounds))), "Missing or duplicate measured iteration")
    _require(
        all(rotations[index] == (rotations[0] + index) % 48 for index in range(len(rounds))),
        "Worker rotation must advance once per measured kernel",
    )
    return raw


def rank_measurements(raw, *, expected_iterations=None):
    validate_report(raw, expected_iterations=expected_iterations)
    result = []
    for measured in sorted(raw["rounds"], key=lambda value: value["iteration"]):
        envelope = _envelope(raw, measured["workers"])
        kernel = measured["kernel_elapsed_us"]
        classes = {}
        for traffic in CLASSES:
            rows = _class_rows(raw, traffic) * raw["repeats"]
            byte_count = rows * raw["row_bytes"]
            class_envelope = _envelope(raw, [worker for worker in measured["workers"] if worker["traffic"] == traffic])
            classes[traffic] = dict(
                rows=rows,
                bytes=byte_count,
                envelope_us=class_envelope,
                gbps=byte_count / (class_envelope * 1000) if rows else None,
                shared_timer_window_gbps=byte_count / (envelope * 1000) if rows else None,
                single_acl_window_gbps=byte_count / (kernel * 1000) if rows else None,
            )
        total_bytes = sum(value["bytes"] for value in classes.values())
        result.append(
            dict(
                iteration=measured["iteration"],
                worker_rotation=measured["worker_rotation"],
                kernel_elapsed_us=kernel,
                envelope_us=envelope,
                bytes=total_bytes,
                rows=sum(w["rows"] for w in measured["workers"]),
                gbps=total_bytes / (envelope * 1000),
                single_acl_window_gbps=total_bytes / (kernel * 1000),
                classes=classes,
            )
        )
    return result


def summarize_reports(reports, *, expected_world=None, expected_iterations=None):
    _require(isinstance(reports, list) and reports, "No group reports")
    measurements = [rank_measurements(raw, expected_iterations=expected_iterations) for raw in reports]
    world = 16 if expected_world is None else expected_world
    _require(
        world == 16 and sorted(raw["rank"] for raw in reports) == list(range(16)), "Missing or duplicate rank report"
    )
    _require(len({raw["device"] for raw in reports}) == 16, "Each rank must use a distinct device")
    first = reports[0]
    _require(
        all(all(raw[field] == first[field] for field in COMMON_FIELDS) for raw in reports),
        "Rank reports disagree on configuration",
    )
    rotations = [row["worker_rotation"] for row in measurements[0]]
    _require(
        all([row["worker_rotation"] for row in rounds] == rotations for rounds in measurements),
        "Ranks disagree on measured rotation/iteration count",
    )
    paired = sorted(zip(reports, measurements), key=lambda pair: pair[0]["rank"])
    reports, measurements = [pair[0] for pair in paired], [pair[1] for pair in paired]
    flat = [row for rounds in measurements for row in rounds]
    total = dict(
        gbps_per_card=distribution([row["gbps"] for row in flat]),
        rank_mean_gbps_per_card=distribution(
            [statistics.mean(row["gbps"] for row in rounds) for rounds in measurements]
        ),
        single_acl_window_gbps_per_card=distribution([row["single_acl_window_gbps"] for row in flat]),
        rank_mean_single_acl_window_gbps_per_card=distribution(
            [statistics.mean(row["single_acl_window_gbps"] for row in rounds) for rounds in measurements]
        ),
        bytes_per_round_per_card=distribution([row["bytes"] for row in flat]),
        envelope_us_per_card=distribution([row["envelope_us"] for row in flat]),
        kernel_elapsed_us_per_card=distribution([row["kernel_elapsed_us"] for row in flat]),
    )
    classes = {}
    for traffic in CLASSES:
        values = [row["classes"][traffic] for row in flat]
        classes[traffic] = dict(
            workers=WORKER_COUNTS[traffic],
            gbps_per_card=distribution([row["gbps"] for row in values]),
            rank_mean_gbps_per_card=distribution(
                [
                    statistics.mean(
                        row["classes"][traffic]["gbps"] for row in rounds if row["classes"][traffic]["gbps"] is not None
                    )
                    for rounds in measurements
                    if any(row["classes"][traffic]["gbps"] is not None for row in rounds)
                ]
            ),
            round_rank_gbps={
                stat: distribution(
                    [
                        distribution([rounds[index]["classes"][traffic]["gbps"] for rounds in measurements])[stat]
                        for index in range(len(rotations))
                        if any(rounds[index]["classes"][traffic]["gbps"] is not None for rounds in measurements)
                    ]
                )
                for stat in ("min", "mean", "max")
            },
            bytes_per_round_per_card=distribution([row["bytes"] for row in values]),
            rows_per_round_per_card=distribution([_class_rows(raw, traffic) * raw["repeats"] for raw in reports]),
            envelope_us_per_card=distribution([row["envelope_us"] for row in values]),
        )
        for metric in ("shared_timer_window_gbps", "single_acl_window_gbps"):
            classes[traffic][metric + "_per_card"] = distribution([row[metric] for row in values])
            rank_means = [distribution([row["classes"][traffic][metric] for row in rounds]) for rounds in measurements]
            classes[traffic]["rank_mean_" + metric + "_per_card"] = distribution(
                [value["mean"] for value in rank_means if value is not None]
            )
    phases = []
    for phase in range(4):
        counts = [_plan(raw)[1][phase] for raw in reports]
        phases.append(
            dict(
                phase=phase,
                bytes_per_round_per_card=distribution(
                    [sum(count.values()) * raw["row_bytes"] * raw["repeats"] for count, raw in zip(counts, reports)]
                ),
            )
        )
    return dict(
        schema_version=5,
        mode="group_round_robin",
        configuration={key: first[key] for key in COMMON_FIELDS},
        metric="rank_local_total_payload_gbps",
        class_metric="rank_local_class_payload_gbps",
        percentile_method="linear_interpolation",
        successful_ranks=16,
        successful_rank_rounds=16 * len(rotations),
        checked_rows=sum(raw["validation"]["checked_rows"] for raw in reports),
        notes=[
            "One kernel concurrently runs 12 INTRA workers and 16 CROSS workers, each with a continuous six-slot ring.",
            "CROSS workers visit the three remote source groups without phase drains or phase timers.",
            "TOTAL uses all active workers' min(start)..max(end) on one NPU.",
            "Each class uses only its own active workers' min(start)..max(end) on one NPU; class rates do not add to TOTAL.",
            "INTRA includes same-rank and same-group reads; metadata input reads are excluded from the payload numerator.",
            "Raw timestamps are never subtracted across different NPUs.",
        ],
        timing_validation=dict(status="consistent", tolerance="max(2us, 10% of ACL kernel duration)"),
        total=total,
        classes=classes,
        phase_diagnostics=phases,
        per_rank=[
            dict(
                rank=raw["rank"],
                device=raw["device"],
                validation=raw["validation"],
                phase_peer_groups=raw["phase_peer_groups"],
                rounds=rounds,
            )
            for raw, rounds in zip(reports, measurements)
        ],
    )


def export_timers(raw, directory):
    validate_report(raw)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    active = [worker for measured in raw["rounds"] for worker in measured["workers"] if worker["rows"]]
    origin = min(worker["start_cycle"] for worker in active)
    trace, rows = [], []
    fields = (
        "rank",
        "device",
        "metadata_mode",
        "buffer_count",
        "iteration",
        "worker_rotation",
        "group_id",
        "traffic",
        "rows",
        "bytes",
        "start_cycle",
        "end_cycle",
        "start_us_rank_origin",
        "duration_us",
    )
    for measured in sorted(raw["rounds"], key=lambda row: row["iteration"]):
        for worker in measured["workers"]:
            start = (worker["start_cycle"] - origin) / raw["cycles_per_us"] if worker["rows"] else None
            duration = (worker["end_cycle"] - worker["start_cycle"]) / raw["cycles_per_us"]
            rows.append(
                (
                    raw["rank"],
                    raw["device"],
                    raw["metadata_mode"],
                    raw["buffer_count"],
                    measured["iteration"],
                    measured["worker_rotation"],
                    worker["group_id"],
                    worker["traffic"],
                    worker["rows"],
                    worker["rows"] * raw["row_bytes"],
                    worker["start_cycle"],
                    worker["end_cycle"],
                    start,
                    duration,
                )
            )
            if start is not None:
                trace.append(
                    dict(
                        ph="X",
                        pid=f"rank_{raw['rank']}",
                        tid=f"AIV1_group_{worker['group_id']}",
                        name=f"round_{measured['iteration']}_{worker['traffic'].upper()}",
                        ts=start,
                        dur=duration,
                        cat=worker["traffic"],
                        args=dict(
                            mode="group_round_robin",
                            metadata_mode=raw["metadata_mode"],
                            buffer_count=6,
                            worker_rotation=measured["worker_rotation"],
                            source_group_order=(
                                raw["phase_peer_groups"][:1]
                                if worker["traffic"] == "intra"
                                else raw["phase_peer_groups"][1:]
                            ),
                            rows=worker["rows"],
                            bytes=worker["rows"] * raw["row_bytes"],
                            time_origin="rank_first_measured_start",
                        ),
                    )
                )
    csv_path = directory / f"rank_{raw['rank']}_workers.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    trace_path = directory / f"rank_{raw['rank']}_trace.json"
    trace_path.write_text(
        json.dumps(
            dict(
                traceEvents=trace,
                displayTimeUnit="us",
                mode="group_round_robin",
                metadata_mode=raw["metadata_mode"],
                buffer_count=6,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, trace_path


def print_summary(summary):
    config = summary["configuration"]
    print(
        f"[GROUP_ROUND_ROBIN] BS={config['batch']} metadata_mode={config['metadata_mode']} slots=6 workers=INTRA:12/CROSS:16; "
        "concurrent in one kernel; CROSS GET order=own-1,own-2,own-3 (mod 4), no phase timers.",
        flush=True,
    )
    total = summary["total"]
    rates, acl = total["rank_mean_gbps_per_card"], total["rank_mean_single_acl_window_gbps_per_card"]
    print(
        f"TOTAL GB/s/card mean={rates['mean']:.3f} rank-min/max={rates['min']:.3f}/{rates['max']:.3f}; "
        f"single-ACL mean={acl['mean']:.3f} rank-min/max={acl['min']:.3f}/{acl['max']:.3f}; "
        f"bytes/card/round={total['bytes_per_round_per_card']['mean']:g} "
        f"timer-span-us={total['envelope_us_per_card']['mean']:.3f} "
        f"ACL-kernel-us={total['kernel_elapsed_us_per_card']['mean']:.3f}",
        flush=True,
    )
    for traffic in CLASSES:
        value = summary["classes"][traffic]
        rate = value["rank_mean_gbps_per_card"]
        if rate is None:
            print(f"{traffic.upper()} GB/s/card=N/A; no payload rows; workers={WORKER_COUNTS[traffic]}", flush=True)
            continue
        round_rate = value["round_rank_gbps"]["mean"]
        span = value["envelope_us_per_card"]
        byte_count = value["bytes_per_round_per_card"]
        print(
            f"{traffic.upper()} GB/s/card mean={rate['mean']:.3f} rank-min/max={rate['min']:.3f}/{rate['max']:.3f} "
            f"round-mean p50/p95={round_rate['p50']:.3f}/{round_rate['p95']:.3f} "
            f"bytes/card/round={byte_count['min']:g}..{byte_count['max']:g} "
            f"class-envelope-us mean={span['mean']:.3f} min/max={span['min']:.3f}/{span['max']:.3f} "
            f"workers={WORKER_COUNTS[traffic]}",
            flush=True,
        )
