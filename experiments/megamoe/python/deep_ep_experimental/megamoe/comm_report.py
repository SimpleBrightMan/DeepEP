# Copyright (c) 2026, SimpleBright_Man

"""Payload accounting and rank-local communication windows; no Torch dependency."""

from collections import defaultdict
import math
import statistics

from .timer import protocol as p

FAMILIES = tuple(f"AIV_{op}_{kind}" for op in ("DISPATCH", "COMBINE") for kind in ("INTRA", "CROSS"))


def payload_bytes(hidden, operation):
    if hidden <= 0 or hidden % 64:
        raise ValueError("hidden must be a positive multiple of 64")
    if operation == "DISPATCH":
        # FP8 input + E8M0 scale, including DMA alignment; local metadata excluded.
        return ((hidden + 255) // 256) * 256 + ((hidden // 32 + 31) // 32) * 32
    if operation == "COMBINE":
        return hidden * 2  # BF16 weighted expert output; one-way PUT payload.
    raise ValueError(f"Unknown operation {operation}")


def read_events(raw, rank):
    if len(raw) != p.TOTAL_BUFFER_SIZE:
        raise ValueError("Timer buffer does not match the installed protocol")
    events = []
    for core in range(p.N_CORE_COUNT):
        for family in FAMILIES:
            index = p.DYNAMIC_NAMES.index(family)
            count = int(raw[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + index])
            if not 0 <= count <= p.MAX_DYNAMIC_ITER:
                raise ValueError(f"rank={rank} core={core}: invalid {family} counter")
            if count and (core % 3 != 2 or core // 3 >= 28):
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
                        **metadata,
                    )
                )
    return events


def expected_rows(bs, topk, rank):
    """Independent count for the repository's balanced rank-first route stream."""
    counts = {"INTRA": 0, "CROSS": 0}
    length = bs * topk
    for source in range(16):
        start = source * length
        first = start + (rank - start) % 16
        rows = max(0, (start + length - 1 - first) // 16 + 1)
        counts["INTRA" if source // 4 == rank // 4 else "CROSS"] += rows
    return counts


def rank_report(events, rank, bs, hidden, topk, dispatch_intra, combine_intra):
    if not 0 <= rank < 16 or not 1 <= dispatch_intra <= 27 or not 1 <= combine_intra <= 27:
        raise ValueError("Expected EP16 and positive INTRA/CROSS worker counts")
    groups = defaultdict(list)
    totals = defaultdict(int)
    for event in events:
        family = event["family"]
        if family not in FAMILIES or event["rank"] != rank:
            raise ValueError("Unexpected communication event family/rank")
        _, operation, kind = family.split("_")
        split = dispatch_intra if operation == "DISPATCH" else combine_intra
        if not 0 <= event["core"] < 28 or (event["core"] < split) != (kind == "INTRA"):
            raise ValueError("Communication event belongs to the wrong worker cohort")
        if event["row_count"] <= 0 or not event["end_cycle"] > event["start_cycle"] > 0 or event["wave"] < 0:
            raise ValueError("Invalid communication event")
        groups[(operation, kind, event["wave"])].append(event)
        totals[(operation, kind)] += event["row_count"]
    expected = expected_rows(bs, topk, rank)
    for operation in ("DISPATCH", "COMBINE"):
        for kind, count in expected.items():
            if totals[(operation, kind)] != count:
                raise ValueError(f"rank={rank} {operation}/{kind}: timer rows {totals[(operation, kind)]} != {count}")
    waves = []
    for (operation, kind, wave), items in sorted(groups.items()):
        rows = sum(item["row_count"] for item in items)
        elapsed = (max(e["end_cycle"] for e in items) - min(e["start_cycle"] for e in items)) / p.CYCLE_TO_US
        size = rows * payload_bytes(hidden, operation)
        waves.append(
            dict(
                operation=operation,
                traffic=kind,
                wave=wave,
                rows=rows,
                bytes=size,
                envelope_us=elapsed,
                gbps=size / elapsed / 1000,
                active_workers=len({e["core"] for e in items}),
                slices=len(items),
            )
        )
    classes = {}
    for operation in ("DISPATCH", "COMBINE"):
        for kind in ("INTRA", "CROSS"):
            selected = [r for r in waves if r["operation"] == operation and r["traffic"] == kind]
            size = sum(r["bytes"] for r in selected)
            elapsed = sum(r["envelope_us"] for r in selected)
            classes[f"{operation}_{kind}"] = dict(
                bytes=size,
                rows=totals[(operation, kind)],
                summed_wave_envelopes_us=elapsed,
                gbps=size / elapsed / 1000 if elapsed else None,
            )
    return dict(rank=rank, classes=classes, waves=waves)


def summarize(records):
    if sorted(r["rank"] for r in records) != list(range(16)):
        raise ValueError("Need exactly all 16 rank reports")
    classes = {}
    for name in (f"{op}_{kind}" for op in ("DISPATCH", "COMBINE") for kind in ("INTRA", "CROSS")):
        values = [r["classes"][name]["gbps"] for r in records]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"Missing/nonpositive bandwidth for {name}")
        classes[name] = dict(
            gbps_per_card_mean=statistics.mean(values), gbps_per_card_min=min(values), gbps_per_card_max=max(values)
        )
    return dict(
        schema_version=1,
        metric="one_way_payload_bytes_over_summed_per_wave_class_envelopes",
        notes=[
            "INTRA includes same-card copies and same-group peers.",
            "Each envelope is rank-local max(end)-min(start) within one wave and worker class.",
            "Wave envelopes include slice gaps, producer waits and scheduling skew; they are not link busy time.",
            "Envelopes can overlap across waves/classes; do not add rates or interpret their sum as E2E.",
            "Dispatch counts padded FP8+scale GET bytes; Combine counts BF16 PUT bytes. No RX+TX doubling.",
            "Communication comes from a separate instrumented call; operator latency uses timer-off profiler.",
        ],
        classes=classes,
        ranks=records,
    )
