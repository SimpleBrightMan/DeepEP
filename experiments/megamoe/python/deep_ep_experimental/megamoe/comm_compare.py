# Copyright (c) 2026, SimpleBright_Man

"""Compare audited full-MegaMoE and communication-only captures, without an NPU.

Identical transfers do not imply identical timings: removing compute changes
memory contention and rank progress. This module measures that difference and
never calibrates or rescales a simulator result to make it match a reference.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

OPERATIONS = ("DISPATCH", "COMBINE")
WORKERS = 28
TRANSFER_FIELDS = ("operation", "wave", "core", "expert", "row", "source_rank", "source_token", "topk_slot")


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / "communication.json"
    data = _load(path)
    if data.get("schema_version") != 2:
        raise ValueError(f"{path}: expected a schema-2 communication.json report")
    if data.get("metric") != "one_way_payload_bytes_over_mean_28_core_summed_span_time":
        raise ValueError(f"{path}: incompatible communication timing metric")
    ranks = data.get("ranks", [])
    if sorted(row["rank"] for row in ranks) != list(range(16)):
        raise ValueError(f"{path}: need exactly all 16 rank reports")
    return path.parent, {row["rank"]: row for row in ranks}


def _detail_path(directory, rank, extension):
    # Use adjacent files so copied reports work and stale absolute filenames
    # cannot silently borrow the other run's audit if a candidate file is lost.
    adjacent = directory / f"rank_{rank:02d}_communication.{extension}"
    if adjacent.is_file():
        return adjacent
    raise ValueError(f"rank={rank}: missing {extension} detail beside {directory / 'communication.json'}")


def _nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: expected a finite nonnegative number")
    return value


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}: expected a nonnegative integer")
    return value


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9)


def _rank_data(directory, rank, compact):
    detail = _load(_detail_path(directory, rank, "json"))
    if detail.get("rank") != rank:
        raise ValueError(f"rank={rank}: detail belongs to another rank")
    for key in ("validation", "operations"):
        if detail.get(key) != compact.get(key):
            raise ValueError(f"rank={rank}: summary and detailed {key} disagree")
    validation = detail["validation"]
    if validation.get("strict_wave_core") is not True:
        raise ValueError(f"rank={rank}: capture lacks strict actual-transfer validation")
    wave_count = _integer(validation["wave_count"], "wave_count")
    if wave_count == 0:
        raise ValueError("No communication waves in capture")
    waves = {}
    for wave in detail["waves"]:
        operation, index = wave["operation"], wave["wave"]
        key = operation, index
        if operation not in OPERATIONS or type(index) is not int or not 0 <= index < wave_count or key in waves:
            raise ValueError(f"rank={rank}: duplicate/invalid operation or wave")
        if wave.get("workers") != WORKERS or sorted(c["core"] for c in wave["cores"]) != list(range(WORKERS)):
            raise ValueError(f"rank={rank} {key}: expected all 28 distinct physical communication cores")
        cores = {core["core"]: core for core in wave["cores"]}
        for core in cores.values():
            rows = _integer(core["rows"], "rows")
            size = _integer(core["bytes"], "bytes")
            duration = _nonnegative(core["duration_us"], "duration_us")
            _integer(core["spans"], "spans")
            if (rows == 0 and (size != 0 or duration != 0)) or (rows > 0 and (size <= 0 or duration <= 0)):
                raise ValueError(f"rank={rank} {key} core={core['core']}: inconsistent empty/nonempty task")
        for field in ("rows", "bytes"):
            if wave[field] != sum(core[field] for core in cores.values()):
                raise ValueError(f"rank={rank} {key}: inconsistent wave {field}")
        if not _close(wave["sum_core_us"], sum(core["duration_us"] for core in cores.values())):
            raise ValueError(f"rank={rank} {key}: inconsistent wave timing total")
        if not _close(wave["mean_core_us"], wave["sum_core_us"] / WORKERS):
            raise ValueError(f"rank={rank} {key}: incompatible core averaging")
        waves[key] = cores
    if set(waves) != {(op, wave) for op in OPERATIONS for wave in range(wave_count)}:
        raise ValueError(f"rank={rank}: missing operation/wave details")
    for operation in OPERATIONS:
        total = detail["operations"][operation]
        selected = [core for (op, _), cores in waves.items() if op == operation for core in cores.values()]
        for field in ("rows", "bytes"):
            if total[field] != sum(core[field] for core in selected):
                raise ValueError(f"rank={rank} {operation}: inconsistent operation {field}")
        duration = sum(core["duration_us"] for core in selected)
        if not _close(total["sum_core_us"], duration) or not _close(total["mean_core_us"], duration / WORKERS):
            raise ValueError(f"rank={rank} {operation}: inconsistent operation timing total")
    transfers = Counter()
    with _detail_path(directory, rank, "csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["rank", *TRANSFER_FIELDS]:
            raise ValueError(f"rank={rank}: unsupported actual-transfer CSV fields")
        for row in reader:
            if int(row["rank"]) != rank or row["operation"] not in OPERATIONS:
                raise ValueError(f"rank={rank}: actual-transfer CSV rank/operation mismatch")
            values = tuple(int(row[field]) for field in TRANSFER_FIELDS[1:])
            if min(values) < 0 or values[0] >= wave_count or values[1] >= WORKERS or values[4] >= 16:
                raise ValueError(f"rank={rank}: invalid actual-transfer identity")
            transfers[(row["operation"], *values)] += 1
    if sum(transfers.values()) != validation["audited_transfers"] or any(count != 1 for count in transfers.values()):
        raise ValueError(f"rank={rank}: missing or duplicate actual-transfer rows")
    counts = Counter((row[0], row[1], row[2]) for row in transfers)
    peers = defaultdict(Counter)
    for row in transfers:
        peers[row[:3]][row[5]] += 1
    for (operation, wave), cores in waves.items():
        for core, item in cores.items():
            key = operation, wave, core
            if counts[key] != item["rows"]:
                raise ValueError(f"rank={rank} {key}: actual-transfer count differs from timer report")
            reported_peers = {peer["source_rank"]: peer["rows"] for peer in item["peers"]}
            if reported_peers != dict(peers[key]):
                raise ValueError(f"rank={rank} {key}: actual-transfer sources differ from timer report")
    return detail, waves, transfers


def _time_pair(reference, candidate, rtol, atol):
    delta = candidate - reference
    return dict(
        reference_us=reference,
        candidate_us=candidate,
        delta_us=delta,
        ratio=candidate / reference if reference else (1.0 if candidate == 0 else None),
        relative_error=delta / reference if reference else (0.0 if candidate == 0 else None),
        within_tolerance=abs(delta) <= atol + rtol * reference,
    )


def compare_reports(reference_path, candidate_path, *, timing_rtol=0.10, timing_atol_us=0.0):
    """Require exact per-wave/core copy identities, then compare summed timings.

    Workload mismatch raises ValueError. Timing mismatch is returned as evidence,
    with `timing_match=False`; the caller chooses whether it is a failing gate.
    Tolerance applies separately to each rank/core's accumulated Dispatch and
    Combine time. Per-wave differences are also included, but are diagnostic.
    """
    _nonnegative(timing_rtol, "timing_rtol")
    _nonnegative(timing_atol_us, "timing_atol_us")
    reference_dir, references = _summary(reference_path)
    candidate_dir, candidates = _summary(candidate_path)
    ranks = []
    all_cores = defaultdict(list)
    all_wave_cores = defaultdict(list)
    for rank in range(16):
        reference, ref_waves, ref_tasks = _rank_data(reference_dir, rank, references[rank])
        candidate, new_waves, new_tasks = _rank_data(candidate_dir, rank, candidates[rank])
        for field in ("wave_count", "m_groups_per_wave", "dispatch_intra", "combine_intra", "logical_wave_sha256"):
            if reference["validation"][field] != candidate["validation"][field]:
                raise ValueError(f"rank={rank}: workload mismatch in {field}; compare captures with the same schedule")
        if ref_tasks != new_tasks:
            missing, unexpected = ref_tasks - new_tasks, new_tasks - ref_tasks
            raise ValueError(
                f"rank={rank}: exact wave/core/token workload mismatch; "
                f"missing={list(missing)[:3]}, unexpected={list(unexpected)[:3]}"
            )
        operations = {}
        for operation in OPERATIONS:
            waves = []
            for wave in range(reference["validation"]["wave_count"]):
                cores = []
                for core in range(WORKERS):
                    ref, new = ref_waves[(operation, wave)][core], new_waves[(operation, wave)][core]
                    if any(ref[field] != new[field] for field in ("rows", "bytes", "spans", "peers")):
                        raise ValueError(
                            f"rank={rank} {operation} wave={wave} core={core}: workload/timer-boundary mismatch"
                        )
                    item = dict(
                        core=core,
                        rows=ref["rows"],
                        bytes=ref["bytes"],
                        **_time_pair(ref["duration_us"], new["duration_us"], timing_rtol, timing_atol_us),
                    )
                    cores.append(item)
                    all_wave_cores[operation].append(dict(rank=rank, wave=wave, **item))
                waves.append(dict(wave=wave, cores=cores))
            cores = []
            for core in range(WORKERS):
                selected = [wave["cores"][core] for wave in waves]
                item = dict(
                    core=core,
                    rows=sum(value["rows"] for value in selected),
                    bytes=sum(value["bytes"] for value in selected),
                    **_time_pair(
                        sum(value["reference_us"] for value in selected),
                        sum(value["candidate_us"] for value in selected),
                        timing_rtol,
                        timing_atol_us,
                    ),
                )
                cores.append(item)
                all_cores[operation].append(dict(rank=rank, **item))
            operations[operation] = dict(cores=cores, waves=waves)
        ranks.append(dict(rank=rank, operations=operations))
    operations = {}
    for operation in OPERATIONS:
        cores, wave_cores = all_cores[operation], all_wave_cores[operation]
        outside = [item for item in cores if not item["within_tolerance"]]
        active = [item for item in cores if item["rows"]]
        operations[operation] = dict(
            count=len(cores),
            active_cores=len(active),
            within_tolerance=len(cores) - len(outside),
            outside_tolerance=len(outside),
            timing_match=not outside,
            wave_timing_match=all(item["within_tolerance"] for item in wave_cores),
            wave_cores_outside_tolerance=sum(not item["within_tolerance"] for item in wave_cores),
            max_abs_relative_error=max((abs(item["relative_error"]) for item in active), default=0.0),
            max_abs_delta_us=max((abs(item["delta_us"]) for item in cores), default=0.0),
            worst_cores=sorted(active, key=lambda item: abs(item["relative_error"]), reverse=True)[:10],
        )
    timing_match = all(value["timing_match"] for value in operations.values())
    return dict(
        schema_version=1,
        reference=str(Path(reference_path)),
        candidate=str(Path(candidate_path)),
        workload_match=True,
        timing_match=timing_match,
        passed=timing_match,
        timing_rtol=timing_rtol,
        timing_atol_us=timing_atol_us,
        timing_gate="every rank/core's sum of communication spans over all waves; Dispatch and Combine separately",
        notes=[
            "Exact copy identities, payload bytes, physical cores and timer span counts must match.",
            "Each rank/core passes when abs(candidate-reference) <= timing_atol_us + timing_rtol*reference.",
            "Per-wave timing comparisons are diagnostic; the timing gate uses per-core sums over waves.",
            "Removing compute changes contention and rank progress; identical tasks do not guarantee equal timings.",
            "These captures contain one timer sample each; repeat both runs to assess hardware variation.",
        ],
        operations=operations,
        ranks=ranks,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", required=True, type=Path, help="Full MegaMoE communication.json or its directory"
    )
    parser.add_argument(
        "--candidate", required=True, type=Path, help="Communication-only communication.json or directory"
    )
    parser.add_argument("--timing-rtol", type=float, default=0.10)
    parser.add_argument("--timing-atol-us", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Save complete per-rank/core/wave comparisons")
    parser.add_argument(
        "--require-timing-match", action="store_true", help="Exit 1 if any core's summed time misses tolerance"
    )
    args = parser.parse_args(argv)
    try:
        result = compare_reports(
            args.reference, args.candidate, timing_rtol=args.timing_rtol, timing_atol_us=args.timing_atol_us
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"[COMPARISON FAILED] {error}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for operation, value in result["operations"].items():
        print(
            f"[COMPARE] {operation} exact-workload=PASS "
            f"core-time-within-tolerance={value['within_tolerance']}/{value['count']} "
            f"max-relative-difference={value['max_abs_relative_error']:.2%} "
            f"max-absolute-difference-us={value['max_abs_delta_us']:.3f}"
        )
    if args.require_timing_match and not result["timing_match"]:
        parser.exit(1, "[COMPARISON FAILED] Per-core communication timings exceed the requested tolerance.\n")
    return result


if __name__ == "__main__":
    main()
