# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Validate native Dispatch reports and aggregate rank-local concurrent traffic."""

import csv
import json
import math
from pathlib import Path
import statistics

TRAFFIC = ("local", "group", "cross")
COMMON_FIELDS = (
    "world",
    "cores",
    "group_size",
    "batch",
    "hidden",
    "intermediate",
    "experts",
    "topk",
    "scope",
    "row_bytes",
    "scale_elements",
    "buffer_count",
    "repeats",
    "worker_counts",
)


class ReportError(ValueError):
    """Incomplete or inconsistent measurements must not become a performance result."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ReportError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ReportError(f"{name} must be finite and positive")
    return value


def _mapping(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ReportError(f"{name} must contain exactly {', '.join(keys)}")
    return value


def _percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values):
    """Linear-interpolation percentiles; no observations produce null, never zero."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    return dict(
        min=min(values),
        mean=statistics.mean(values),
        max=max(values),
        p50=_percentile(values, 0.5),
        p95=_percentile(values, 0.95),
    )


def balanced_traffic_rows(raw):
    """Count route residues in source streams without enumerating token/top-k rows."""
    world, experts, rank = raw["world"], raw["experts"], raw["rank"]
    source_items = raw["batch"] * raw["topk"]
    selected_experts = 1 if raw["scope"] == "first-expert" else experts // world
    result = dict.fromkeys(TRAFFIC, 0)
    for peer in range(world):
        traffic = (
            "local" if peer == rank else "group" if peer // raw["group_size"] == rank // raw["group_size"] else "cross"
        )
        start, stop = peer * source_items, (peer + 1) * source_items
        for expert in range(selected_experts):
            residue = expert * world + rank
            # Number of r == residue (mod experts) in the half-open interval.
            result[traffic] += (stop - 1 - residue) // experts - (start - 1 - residue) // experts
    return result


def _planned_workers(raw, traffic_rows):
    result = []
    for traffic in TRAFFIC:
        workers = raw["worker_counts"][traffic]
        base, extra = divmod(traffic_rows[traffic], workers)
        result.extend((traffic, (base + int(worker < extra)) * raw["repeats"]) for worker in range(workers))
    return result


def _timer_envelope(raw, measured):
    active = [worker for worker in measured["workers"] if worker["rows"]]
    return (
        (
            (max(worker["end_cycle"] for worker in active) - min(worker["start_cycle"] for worker in active))
            / raw["cycles_per_us"]
        )
        if active
        else None
    )


def validate_report(raw, *, expected_iterations=None):
    if not isinstance(raw, dict):
        raise ReportError("Report must be a JSON object")
    required = set(COMMON_FIELDS) | {
        "schema_version",
        "rank",
        "device",
        "cycles_per_us",
        "traffic_rows",
        "validation",
        "rounds",
    }
    missing = required - set(raw)
    if missing:
        raise ReportError("Missing report field(s): " + ", ".join(sorted(missing)))
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ReportError("Unsupported schema_version")
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
    ):
        _integer(raw[name], name, 1)
    _integer(raw["rank"], "rank")
    _integer(raw["device"], "device")
    if not 4 <= raw["world"] <= 16 or not 3 <= raw["cores"] <= 32:
        raise ReportError("world/cores are outside the native planner limits")
    if raw["rank"] >= raw["world"]:
        raise ReportError("rank must be within world")
    if (
        not 1 < raw["group_size"] < raw["world"]
        or raw["world"] % raw["group_size"]
        or raw["experts"] % raw["world"]
        or raw["topk"] > min(32, raw["experts"])
    ):
        raise ReportError("Invalid world/group_size/expert/top-k layout")
    if raw["experts"] > 16384 or raw["experts"] // raw["world"] > 128 or raw["batch"] > 65536:
        raise ReportError("Batch/expert capacity exceeds native planner limits")
    if not 1024 <= raw["hidden"] <= 8192 or raw["hidden"] % 64:
        raise ReportError("hidden is outside the native planner limits")
    if not 256 <= raw["intermediate"] <= 4096 or raw["intermediate"] % 128 or not 2 <= raw["buffer_count"] <= 6:
        raise ReportError("intermediate/buffer_count are outside the native planner limits")
    if raw["scope"] not in ("all", "first-expert"):
        raise ReportError("Invalid scope")
    scale_elements = (raw["hidden"] + 31) // 32
    row_bytes = (raw["hidden"] + 255) // 256 * 256 + (scale_elements + 31) // 32 * 32
    if raw["scale_elements"] != scale_elements or raw["row_bytes"] != row_bytes:
        raise ReportError("row_bytes/scale_elements disagree with padded FP8 + E8M0 layout")
    _positive(raw["cycles_per_us"], "cycles_per_us")
    for name, minimum in (("traffic_rows", 0), ("worker_counts", 1)):
        _mapping(raw[name], TRAFFIC, name)
        for traffic in TRAFFIC:
            _integer(raw[name][traffic], f"{name}.{traffic}", minimum)
    if sum(raw["worker_counts"].values()) != raw["cores"]:
        raise ReportError("worker_counts must sum to cores")
    expected_rows = balanced_traffic_rows(raw)
    if raw["traffic_rows"] != expected_rows:
        raise ReportError(f"traffic_rows disagree with balanced routing and scope: expected {expected_rows}")
    planned_workers = _planned_workers(raw, expected_rows)
    validation = raw["validation"]
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ReportError(f"rank {raw['rank']}: native output validation did not pass")
    _integer(validation.get("checked_rows"), "validation.checked_rows")
    if validation["checked_rows"] != sum(raw["traffic_rows"].values()):
        raise ReportError("validation.checked_rows does not cover the requested traffic")
    rounds = raw["rounds"]
    if not isinstance(rounds, list) or not rounds:
        raise ReportError("rounds must be a nonempty list")
    if expected_iterations is not None and len(rounds) != expected_iterations:
        raise ReportError(f"Expected {expected_iterations} iterations, found {len(rounds)}")
    iterations = []
    for measured in rounds:
        if not isinstance(measured, dict) or not {"iteration", "kernel_elapsed_us", "workers"} <= set(measured):
            raise ReportError("Malformed round")
        iterations.append(_integer(measured["iteration"], "iteration"))
        _positive(measured["kernel_elapsed_us"], "kernel_elapsed_us")
        workers = measured["workers"]
        if not isinstance(workers, list) or len(workers) != raw["cores"]:
            raise ReportError("Each round must contain all cores")
        groups, rows, counts = [], dict.fromkeys(TRAFFIC, 0), dict.fromkeys(TRAFFIC, 0)
        for worker in workers:
            if not isinstance(worker, dict) or not {"group_id", "traffic", "rows", "start_cycle", "end_cycle"} <= set(
                worker
            ):
                raise ReportError("Malformed worker record")
            group = _integer(worker["group_id"], "group_id")
            if group >= raw["cores"]:
                raise ReportError("group_id exceeds cores")
            groups.append(group)
            traffic = worker["traffic"]
            if traffic not in TRAFFIC:
                raise ReportError(f"Unknown traffic: {traffic!r}")
            amount = _integer(worker["rows"], "worker.rows")
            if (traffic, amount) != planned_workers[group]:
                raise ReportError(f"group_id {group} traffic/rows disagree with its planned worker partition")
            start = _integer(worker["start_cycle"], "start_cycle")
            end = _integer(worker["end_cycle"], "end_cycle")
            if (amount and (start == 0 or end <= start)) or (not amount and end < start):
                raise ReportError("Invalid worker timing interval")
            rows[traffic] += amount
            counts[traffic] += 1
        if len(set(groups)) != raw["cores"]:
            raise ReportError("Duplicate group_id in round")
        if counts != raw["worker_counts"]:
            raise ReportError("Worker traffic allocation disagrees with worker_counts")
        if any(rows[t] != raw["traffic_rows"][t] * raw["repeats"] for t in TRAFFIC):
            raise ReportError("Worker rows disagree with traffic_rows * repeats")
        envelope = _timer_envelope(raw, measured)
        tolerance = max(2.0, measured["kernel_elapsed_us"] * 0.1)
        if envelope is not None and envelope > measured["kernel_elapsed_us"] + tolerance:
            raise ReportError(
                f"rank {raw['rank']} iteration {measured['iteration']}: timer envelope {envelope:g}us exceeds "
                f"ACL kernel duration {measured['kernel_elapsed_us']:g}us + tolerance {tolerance:g}us; "
                "check counter conversion and the within-NPU clock domain"
            )
    if sorted(iterations) != list(range(len(rounds))):
        raise ReportError("Missing or duplicate iteration")
    return raw


def rank_measurements(raw, *, expected_iterations=None):
    validate_report(raw, expected_iterations=expected_iterations)
    result = []
    for measured in sorted(raw["rounds"], key=lambda item: item["iteration"]):
        classes = {}
        for traffic in TRAFFIC:
            active = [worker for worker in measured["workers"] if worker["traffic"] == traffic and worker["rows"]]
            amount = raw["traffic_rows"][traffic] * raw["repeats"]
            byte_count = amount * raw["row_bytes"]
            # Subtract on the same device only. Keep integer cycle differences exact.
            envelope = (
                ((max(w["end_cycle"] for w in active) - min(w["start_cycle"] for w in active)) / raw["cycles_per_us"])
                if active
                else None
            )
            classes[traffic] = dict(
                rows=amount,
                bytes=byte_count,
                workers=raw["worker_counts"][traffic],
                active_workers=len(active),
                envelope_us=envelope,
                gbps=byte_count / (envelope * 1000.0) if envelope is not None else None,
                kernel_window_gbps=byte_count / (measured["kernel_elapsed_us"] * 1000.0),
            )
        envelope = _timer_envelope(raw, measured)
        result.append(
            dict(
                iteration=measured["iteration"],
                kernel_elapsed_us=measured["kernel_elapsed_us"],
                timer_envelope_us=envelope,
                timer_to_acl_ratio=envelope / measured["kernel_elapsed_us"] if envelope is not None else None,
                classes=classes,
            )
        )
    return result


def summarize_reports(reports, *, expected_world=None, expected_iterations=None):
    if not reports:
        raise ReportError("No rank reports")
    for raw in reports:
        validate_report(raw, expected_iterations=expected_iterations)
    world = expected_world if expected_world is not None else reports[0]["world"]
    ranks = [raw["rank"] for raw in reports]
    if len(ranks) != len(set(ranks)):
        raise ReportError("Duplicate rank report")
    if sorted(ranks) != list(range(world)):
        raise ReportError(f"Missing rank report(s): expected 0..{world - 1}, found {sorted(ranks)}")
    if len({raw["device"] for raw in reports}) != world:
        raise ReportError("Each native rank must use a distinct device")
    reports = sorted(reports, key=lambda item: item["rank"])
    first = reports[0]
    if any(any(raw[name] != first[name] for name in COMMON_FIELDS) for raw in reports):
        raise ReportError("Rank reports disagree on benchmark configuration")
    if first["world"] != world or any(len(raw["rounds"]) != len(first["rounds"]) for raw in reports):
        raise ReportError("Ranks disagree on world or iteration count")
    measurements = [rank_measurements(raw) for raw in reports]
    per_rank, per_round = [], []
    for raw, rounds in zip(reports, measurements):
        classes = {traffic: distribution([row["classes"][traffic]["gbps"] for row in rounds]) for traffic in TRAFFIC}
        per_rank.append(dict(rank=raw["rank"], device=raw["device"], validation=raw["validation"], classes=classes))
    for index in range(len(first["rounds"])):
        classes = {}
        for traffic in TRAFFIC:
            values = [rank[index]["classes"][traffic] for rank in measurements]
            classes[traffic] = dict(
                gbps_per_card=distribution([value["gbps"] for value in values]),
                active_ranks=sum(value["gbps"] is not None for value in values),
                bytes_per_card=distribution([value["bytes"] for value in values]),
                envelope_us_per_card=distribution([value["envelope_us"] for value in values]),
                kernel_window_gbps_per_card=distribution([value["kernel_window_gbps"] for value in values]),
            )
        per_round.append(dict(iteration=index, classes=classes))
    classes = {}
    for traffic in TRAFFIC:
        values = [row["classes"][traffic] for rank in measurements for row in rank]
        round_rates = [row["classes"][traffic]["gbps_per_card"] for row in per_round]
        classes[traffic] = dict(
            workers=first["worker_counts"][traffic],
            gbps_per_card=distribution([value["gbps"] for value in values]),
            rank_mean_gbps_per_card=distribution(
                [rank["classes"][traffic]["mean"] for rank in per_rank if rank["classes"][traffic] is not None]
            ),
            round_rank_gbps={
                stat: distribution([value[stat] for value in round_rates if value is not None])
                for stat in ("min", "mean", "max")
            },
            rows_per_round_per_card=distribution([raw["traffic_rows"][traffic] * raw["repeats"] for raw in reports]),
            bytes_per_round_per_card=distribution([value["bytes"] for value in values]),
            envelope_us_per_card=distribution([value["envelope_us"] for value in values]),
            kernel_window_gbps_per_card=distribution([value["kernel_window_gbps"] for value in values]),
        )
    return dict(
        schema_version=1,
        mode="concurrent_local_group_cross",
        configuration={name: first[name] for name in COMMON_FIELDS},
        metric="rank_local_class_payload_gbps",
        percentile_method="linear_interpolation",
        notes=[
            "Three traffic classes execute concurrently in one kernel.",
            "Each class uses its own active workers' min(start)..max(end) on the same NPU.",
            "LOCAL includes local reads; measured spans also include local output and metadata writes.",
            "Kernel-window contributions use the ACL-event whole-kernel duration and a different timing scope.",
            "Zero-row classes have no bandwidth observation; rank clocks are never subtracted across NPUs.",
        ],
        successful_ranks=world,
        successful_rank_rounds=world * len(first["rounds"]),
        checked_rows=sum(raw["validation"]["checked_rows"] for raw in reports),
        timing_validation=dict(
            status="consistent",
            tolerance="max(2us, 10% of ACL kernel duration)",
            timer_to_acl_ratio=distribution([row["timer_to_acl_ratio"] for rank in measurements for row in rank]),
            timer_envelope_us=distribution([row["timer_envelope_us"] for rank in measurements for row in rank]),
        ),
        classes=classes,
        per_round=per_round,
        per_rank=per_rank,
    )


def load_reports(directory, world):
    reports = []
    for rank in range(world):
        path = Path(directory) / f"rank_{rank}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ReportError(f"Cannot read rank {rank} report {path}: {error}") from error
        if not isinstance(raw, dict) or raw.get("rank") != rank:
            raise ReportError(f"Rank identity mismatch in {path}")
        reports.append(raw)
    return reports


def export_timers(raw, directory):
    """Export per-rank traces; every lane shares one rank-local measured origin."""
    validate_report(raw)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    active = [worker for measured in raw["rounds"] for worker in measured["workers"] if worker["rows"]]
    origin = min(worker["start_cycle"] for worker in active) if active else 0
    trace = []
    path = directory / f"rank_{raw['rank']}_workers.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "rank",
                "device",
                "iteration",
                "group_id",
                "traffic",
                "rows",
                "bytes",
                "start_cycle",
                "end_cycle",
                "start_us_rank_origin",
                "duration_us",
            )
        )
        for measured in sorted(raw["rounds"], key=lambda item: item["iteration"]):
            for worker in measured["workers"]:
                duration = (worker["end_cycle"] - worker["start_cycle"]) / raw["cycles_per_us"]
                start = (worker["start_cycle"] - origin) / raw["cycles_per_us"] if worker["rows"] else None
                writer.writerow(
                    (
                        raw["rank"],
                        raw["device"],
                        measured["iteration"],
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
                                rank=raw["rank"],
                                device=raw["device"],
                                rows=worker["rows"],
                                bytes=worker["rows"] * raw["row_bytes"],
                                time_origin="rank_first_measured_start",
                            ),
                        )
                    )
    output = directory / f"rank_{raw['rank']}_trace.json"
    output.write_text(json.dumps(dict(traceEvents=trace, displayTimeUnit="us"), indent=2), encoding="utf-8")
    return path, output
