# Copyright (c) 2026, SimpleBright_Man

"""Strict wave/core/task accounting and separately measured D/C bandwidth."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .timer import protocol as p
from . import wave_oracle

OPERATIONS = ("DISPATCH", "COMBINE")
FAMILIES = tuple(f"AIV_{op}_COMM" for op in OPERATIONS)
WORKERS = 28


def payload_bytes(hidden, operation):
    if hidden <= 0 or hidden % 64:
        raise ValueError("hidden must be a positive multiple of 64")
    if operation == "DISPATCH":
        return ((hidden + 255) // 256) * 256 + ((hidden // 32 + 31) // 32) * 32
    if operation == "COMBINE":
        return hidden * 2
    raise ValueError(f"Unknown operation {operation}")


def read_events(raw, rank):
    if len(raw) not in (p.TOTAL_BUFFER_SIZE, p.COMM_AUDIT_BUFFER_SIZE):
        raise ValueError("Timer buffer does not match the installed protocol")
    events = []
    for core in range(p.N_CORE_COUNT):
        for family in FAMILIES:
            index = p.DYNAMIC_NAMES.index(family)
            count = int(raw[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + index])
            if not 0 <= count <= p.MAX_DYNAMIC_ITER:
                raise ValueError(f"rank={rank} core={core}: invalid {family} counter")
            if count and (core % 3 != 2 or core // 3 >= WORKERS):
                raise ValueError("Communication events must belong to one of the 28 AIV1 workers")
            for item in range(count):
                logical = len(p.FIXED_NAMES) + index * p.MAX_DYNAMIC_ITER + item
                offset = core * p.COUNTERS_PER_CORE + logical * 2
                start, end = int(raw[offset]), int(raw[offset + 1])
                wave = int(raw[p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical])
                metadata = p.task_metadata(raw, core, logical)
                if not end > start > 0 or wave < 0 or metadata.get("row_count", 0) <= 0:
                    raise ValueError(f"rank={rank} core={core} {family}: incomplete timing/row metadata")
                events.append(
                    dict(
                        rank=rank,
                        core=core // 3,
                        family=family,
                        wave=wave,
                        start_cycle=start,
                        end_cycle=end,
                        m_groups_per_wave=int(raw[p.COMM_M_GROUPS_PER_WAVE_OFFSET]),
                        audit_enabled=bool(raw[p.COMM_AUDIT_ENABLE_OFFSET]),
                        **metadata,
                    )
                )
    return events


def read_transfers(raw, rank):
    """Decode actual per-row audit records; never infer identities from timers."""
    if len(raw) != p.COMM_AUDIT_BUFFER_SIZE or int(raw[p.COMM_AUDIT_ENABLE_OFFSET]) == 0:
        raise ValueError("Expected a task-audit buffer with audit enabled")
    records = []
    for hardware_core in range(p.N_CORE_COUNT):
        count = int(raw[p.COMM_AUDIT_COUNT_OFFSET + hardware_core])
        if not 0 <= count <= p.COMM_AUDIT_MAX_RECORDS_PER_CORE:
            raise ValueError(f"rank={rank} core={hardware_core}: communication audit overflow/invalid count")
        if count and (hardware_core % 3 != 2 or hardware_core // 3 >= WORKERS):
            raise ValueError("Task audit records must belong to one of the 28 AIV1 workers")
        for item in range(count):
            offset = p.COMM_AUDIT_RECORD_OFFSET + (hardware_core * p.COMM_AUDIT_MAX_RECORDS_PER_CORE + item) * 3
            wave, task, route = map(int, raw[offset : offset + 3])
            operation = task >> 56
            if operation not in (0, 1) or wave < 0 or task < 0 or route < 0:
                raise ValueError("Invalid actual-transfer audit record")
            records.append(
                dict(
                    rank=rank,
                    operation=OPERATIONS[operation],
                    wave=wave,
                    core=hardware_core // 3,
                    expert_id=(task >> 32) & 0xFFFFFF,
                    row_start=task & 0xFFFFFFFF,
                    row_count=1,
                    source_rank=(route >> 48) & 0xFFFF,
                    topk_slot=(route >> 32) & 0xFFFF,
                    source_token=route & 0xFFFFFFFF,
                )
            )
    return records


def expected_rows(bs, topk, rank):
    counts = {"INTRA": 0, "CROSS": 0}
    length = bs * topk
    for source in range(16):
        start = source * length
        first = start + (rank - start) % 16
        rows = max(0, (start + length - 1 - first) // 16 + 1)
        counts["INTRA" if source // 4 == rank // 4 else "CROSS"] += rows
    return counts


def _transfer(record, rank):
    if record.get("rank") != rank or record.get("row_count", 1) != 1:
        raise ValueError("Audit transfer must identify one actual row on the expected rank")
    return wave_oracle.Transfer(
        record["operation"],
        int(record["wave"]),
        int(record["core"]),
        int(record["expert_id"]),
        int(record["row_start"]),
        int(record["source_rank"]),
        int(record["source_token"]),
        int(record["topk_slot"]),
    )


def _measure(size, rows, durations):
    total = sum(durations)
    mean = total / WORKERS
    if rows and mean <= 0:
        raise ValueError("Nonempty communication has no measured duration")
    return dict(
        rows=rows,
        bytes=size,
        sum_core_us=total,
        mean_core_us=mean,
        workers=WORKERS,
        gbps=size / mean / 1000 if mean else None,
    )


def rank_report(
    events,
    rank,
    bs,
    hidden,
    topk,
    dispatch_intra,
    combine_intra,
    *,
    experts=384,
    intermediate=1280,
    ranks_per_node=16,
    transfers=None,
):
    """Validate an independent audit, then account only the measured timer call.

    The audit call records actual copy identities separately. Its instrumented
    timings are intentionally never used for bandwidth.
    """
    if transfers is None:
        raise ValueError("Strict wave/core reporting requires a separate actual-transfer audit")
    oracle = wave_oracle.plan(
        bs, hidden, intermediate, experts, topk, rank, dispatch_intra, combine_intra, ranks_per_node
    )
    actual = [_transfer(record, rank) for record in transfers]
    expected = oracle["transfers"]
    missing, unexpected = Counter(expected) - Counter(actual), Counter(actual) - Counter(expected)
    if missing or unexpected:
        raise ValueError(
            f"rank={rank}: wave/core/token fidelity mismatch; "
            f"missing={list(missing.items())[:3]}, unexpected={list(unexpected.items())[:3]}"
        )
    expected_counts = Counter((r.operation, r.wave, r.core, r.expert) for r in expected)
    measured_counts, event_groups, core_events = Counter(), defaultdict(list), defaultdict(list)
    expected_positions = defaultdict(set)
    for row in expected:
        expected_positions[(row.operation, row.wave, row.core, row.expert)].add(row.row)
    measured_positions = defaultdict(set)
    slices = {(s["wave"], s["expert"]): s for s in oracle["slices"]}
    for event in events:
        if event.get("audit_enabled") is not False:
            raise ValueError("Bandwidth must use timer-only spans, not the task-audit call")
        if event.get("m_groups_per_wave") != oracle["m_groups_per_wave"]:
            raise ValueError("Measured device wave budget differs from independent model tiling")
        family = event["family"]
        if family not in FAMILIES or event["rank"] != rank:
            raise ValueError("Unexpected communication event family/rank")
        operation = family.split("_")[1]
        split = dispatch_intra if operation == "DISPATCH" else combine_intra
        core, wave, expert = event["core"], event["wave"], event["expert_id"]
        if not 0 <= core < WORKERS or event["row_count"] <= 0 or not event["end_cycle"] > event["start_cycle"] > 0:
            raise ValueError("Invalid communication event")
        wanted_family = f"AIV_{operation}_COMM"
        if family != wanted_family:
            raise ValueError("Communication event belongs to the wrong schedule/cohort")
        key = (operation, wave, core, expert)
        if key not in expected_counts:
            raise ValueError(f"Unexpected wave/core/expert timer task: {key}")
        if split:
            if event["row_start"] != slices[(wave, expert)]["row_start"]:
                raise ValueError("Split timer refers to the wrong original expert slice")
        else:
            positions = set(range(event["row_start"], event["row_start"] + event["row_count"]))
            if positions & measured_positions[key]:
                raise ValueError("Duplicate original-schedule timer rows")
            measured_positions[key].update(positions)
        measured_counts[key] += event["row_count"]
        event_groups[(operation, wave, core)].append(event)
        core_events[core].append(event)
    if measured_counts != expected_counts:
        raise ValueError(f"rank={rank}: measured per-wave/core/expert row counts differ from the original workload")
    for key, positions in measured_positions.items():
        if positions != expected_positions[key]:
            raise ValueError(f"rank={rank}: measured original row interval differs at {key}")
    for core, items in core_events.items():
        ordered = sorted(items, key=lambda e: e["start_cycle"])
        if any(left["end_cycle"] > right["start_cycle"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"rank={rank} core={core}: overlapping D/C timing spans would double-count time")
    rows_by_worker, peers_by_worker = Counter(), defaultdict(Counter)
    for row in actual:
        key = row.operation, row.wave, row.core
        rows_by_worker[key] += 1
        peers_by_worker[key][row.source_rank] += 1
    waves = []
    for operation in OPERATIONS:
        row_bytes = payload_bytes(hidden, operation)
        for wave in range(oracle["wave_count"]):
            cores, selected = [], []
            for core in range(WORKERS):
                key = operation, wave, core
                items = event_groups[key]
                elapsed = sum((item["end_cycle"] - item["start_cycle"]) / p.CYCLE_TO_US for item in items)
                rows = rows_by_worker[key]
                cores.append(
                    dict(
                        core=core,
                        rows=rows,
                        bytes=rows * row_bytes,
                        duration_us=elapsed,
                        spans=len(items),
                        peers=[
                            dict(
                                source_rank=peer,
                                traffic="INTRA" if peer // 4 == rank // 4 else "CROSS",
                                rows=count,
                                bytes=count * row_bytes,
                            )
                            for peer, count in sorted(peers_by_worker[key].items())
                        ],
                    )
                )
                selected.extend(items)
            rows = sum(core["rows"] for core in cores)
            measured = _measure(rows * row_bytes, rows, [core["duration_us"] for core in cores])
            envelope = (
                ((max(e["end_cycle"] for e in selected) - min(e["start_cycle"] for e in selected)) / p.CYCLE_TO_US)
                if selected
                else 0.0
            )
            waves.append(dict(operation=operation, wave=wave, **measured, envelope_us=envelope, cores=cores))
    operations = {}
    for operation in OPERATIONS:
        selected = [w for w in waves if w["operation"] == operation]
        rows = sum(w["rows"] for w in selected)
        if rows != sum(expected_rows(bs, topk, rank).values()):
            raise ValueError("Payload row total is inconsistent with independent balanced routing")
        size = rows * payload_bytes(hidden, operation)
        operations[operation] = _measure(size, rows, [w["sum_core_us"] for w in selected])
        operations[operation]["peer_rows"] = [
            sum(
                peers_by_worker[(operation, wave, core)][peer]
                for wave in range(oracle["wave_count"])
                for core in range(WORKERS)
            )
            for peer in range(16)
        ]
        operations[operation]["intra_rows"] = expected_rows(bs, topk, rank)["INTRA"]
        operations[operation]["cross_rows"] = expected_rows(bs, topk, rank)["CROSS"]
    identities = sorted(
        (r.operation, r.wave, r.expert, r.row, r.source_rank, r.source_token, r.topk_slot) for r in actual
    )
    digest = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    return dict(
        rank=rank,
        operations=operations,
        waves=waves,
        validation=dict(
            strict_wave_core=True,
            audited_transfers=len(actual),
            wave_count=oracle["wave_count"],
            m_groups_per_wave=oracle["m_groups_per_wave"],
            logical_wave_sha256=digest,
            dispatch_intra=dispatch_intra,
            combine_intra=combine_intra,
        ),
        _transfers=actual,
    )


def write_rank_detail(report, directory):
    """Persist full detail locally; return a small, JSON-safe StoreGroup payload."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"rank_{report['rank']:02d}_communication"
    json_path, csv_path = directory / f"{stem}.json", directory / f"{stem}.csv"
    visible = {key: value for key, value in report.items() if not key.startswith("_")}
    json_path.write_text(json.dumps(visible, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", *wave_oracle.Transfer._fields])
        for row in sorted(report["_transfers"]):
            writer.writerow([report["rank"], *row])
    return dict(
        rank=report["rank"],
        operations=report["operations"],
        validation=report["validation"],
        detail_files=dict(json=str(json_path), csv=str(csv_path)),
    )


def summarize(records):
    if sorted(r["rank"] for r in records) != list(range(16)):
        raise ValueError("Need exactly all 16 rank reports")
    operations = {}
    for operation in OPERATIONS:
        values = [r["operations"][operation]["gbps"] for r in records]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"Missing/nonpositive bandwidth for {operation}")
        operations[operation] = dict(
            gbps_per_card_mean=statistics.mean(values), gbps_per_card_min=min(values), gbps_per_card_max=max(values)
        )
    if any(not r["validation"]["strict_wave_core"] for r in records):
        raise ValueError("All ranks must pass strict actual-transfer fidelity validation")
    return dict(
        schema_version=2,
        metric="one_way_payload_bytes_over_mean_28_core_summed_span_time",
        notes=[
            "Dispatch and Combine are measured and reported separately.",
            "For each operation: GB/s = payload bytes / (sum of all 28 cores' span durations / 28) / 1000.",
            "Durations are in microseconds. Idle cores contribute zero and remain in the denominator of 28.",
            "Per-wave bandwidth uses the same formula. Envelopes are diagnostic only, never the denominator.",
            "Dispatch counts padded FP8+scale GET payload; Combine counts BF16 PUT payload, without RX+TX doubling.",
            "Payload excludes route indices, metadata, flags and other control traffic.",
            "Exact copy identities come from a separate audit call; bandwidth uses a timer-only call.",
            "Dispatch spans include prefix reload and ownership planning; Combine includes ownership arithmetic and the successful readiness probe.",
            "Combine producer waits and failed readiness probes are excluded. Empty communication tasks record zero time.",
            "These are mean-core-time effective rates, not wall-clock link throughput or E2E rates.",
            "Operator latency is measured separately with both timer and task audit disabled.",
        ],
        operations=operations,
        ranks=records,
    )
