"""Validate URMA detailed trace schema 1 and export a rank-local Chrome trace.

Raw records separate submission, waiting and drain. The default display joins
complete GMM/Dequant context batches, retaining the original scopes for metrics.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

TRACE_SCHEMA = 1
ENGINE_COUNT = 84
HEADER_WORDS = RECORD_WORDS = 16
UNKNOWN = (1 << 64) - 1
EVENTS = (
    "Pack",
    "DispatchPutX",
    "DispatchPutScale",
    "DispatchQuiet",
    "DispatchPublish",
    "SelfDispatch",
    "WaitStartPeer",
    "PublishStart",
    "WaitLocalDispatch",
    "WaitCombineReady",
    "CombinePut",
    "CombineQuiet",
    "SelfCombine",
    "FinalPeerHandshake",
    "WaitDispatch",
    "Gmm1Tile",
    "Gmm1TileDrain",
    "Gmm1Dequant",
    "WaitActivation",
    "Gmm2Tile",
    "Gmm2Dequant",
    "Gmm2Drain",
    "WaitGmm1",
    "ActivationSlice",
    "RankBarrier",
    "Reduce",
    "LocalSync",
    "Gmm1Drain",
    "InputScaleGather",
    "WaitInputScale",
    "CombinePublish",
    "WaitReduceReady",
    "ReduceDrain",
    "ReservedJson33",
    "ReservedJson34",
    "ReservedJson35",
    "ReservedJson36",
    "ReservedJson37",
    "ReservedJson38",
    "Gmm1Batch",
    "Gmm2Batch",
    "Gmm1DequantBatch",
    "Gmm2DequantBatch",
)
RAW_EVENT_IDS = tuple(index for index in range(len(EVENTS)) if not 33 <= index <= 38)
NATIVE_BATCH_IDS = (39, 40, 41, 42)
METADATA_FIELDS = ("peer", "cv", "fragment", "expert", "row", "rows", "tile_m", "tile_n", "columns")
SYNC_STAGES = {0: "pack", 1: "compute/return", 2: "rank barrier", 3: "local invocation completion"}


class TraceOverflowError(ValueError):
    """An incomplete trace must not be used as if it contained every event."""


def event_style(event):
    if event in ("Gmm1Batch", "Gmm2Batch", "Gmm1DequantBatch", "Gmm2DequantBatch"):
        return (
            event + " [device batch]",
            "cube.batch" if "Dequant" not in event else "compute.dequant_batch",
            "good" if "Dequant" not in event else "rail_animation",
            "one device scope per expert context; includes internal waits and drain; "
            "first input wait is outside AIC scope; no internal timing records",
        )
    if event == "Reduce":
        return (
            event,
            "compute.copy",
            "rail_animation",
            "reduction scope; may include input readiness waits and final output drain",
        )
    if event in ("DispatchPutX", "DispatchPutScale", "CombinePut"):
        return (
            event + " [PUT submit]",
            "transport.submit",
            "thread_state_runnable",
            "enqueue scope; not remote completion",
        )
    if event in ("Gmm1Tile", "Gmm2Tile"):
        return event + " [Cube submit]", "cube.submit", "good", "submission scope; not Cube completion"
    if event.endswith("Quiet"):
        return (
            event + " [completion wait]",
            "transport.quiet",
            "thread_state_iowait",
            "wait for prior peer PUT completion",
        )
    if event.endswith("Drain"):
        return event + " [drain]", "compute.drain", "thread_state_iowait", "explicit pipeline drain scope"
    if event.startswith("Wait") or event in ("RankBarrier", "LocalSync", "FinalPeerHandshake"):
        return event, "wait.sync", "thread_state_iowait", "dependency or synchronization wait"
    if event in ("DispatchPublish", "CombinePublish", "PublishStart"):
        return event, "signal.publish", "rail_response", "signal publication scope"
    return event, "compute.copy", "rail_animation", "recorded device scope"


def decode_trace(raw, *, rank, clock_mhz):
    """Return Chrome trace JSON. All tick subtraction uses Python integers first.

    Input is an int64/uint64 array or the little-endian raw bytes written by the
    benchmark. Signed storage is reinterpreted as uint64, including sentinels.
    Each rank is zeroed independently; joining files cannot align rank clocks.
    """
    import numpy as np

    if type(rank) is not int or not 0 <= rank < 16:
        raise ValueError("rank must be an integer in [0,16)")
    if isinstance(clock_mhz, bool) or not math.isfinite(clock_mhz) or clock_mhz <= 0:
        raise ValueError("clock_mhz must be positive and finite")
    if isinstance(raw, (bytes, bytearray, memoryview)):
        if len(raw) % 8:
            raise ValueError("trace byte length must be a multiple of 8")
        values = np.frombuffer(raw, dtype="<u8")
    else:
        values = np.asarray(raw)
        if values.dtype.kind not in "iu" or values.dtype.itemsize != 8:
            raise ValueError("trace requires 64-bit integer storage")
        values = values.view(np.dtype(values.dtype.byteorder + "u8")).reshape(-1)
    if values.size < ENGINE_COUNT * HEADER_WORDS:
        raise ValueError("trace is missing engine headers")
    if int(values[0]) != TRACE_SCHEMA:
        raise ValueError("trace schema is missing or incompatible")
    capacity = int(values[1])
    if not 1 <= capacity <= 65536:
        raise ValueError("trace capacity must be in [1,65536]")
    stride = HEADER_WORDS + capacity * RECORD_WORDS
    if values.size != ENGINE_COUNT * stride:
        raise ValueError(f"trace size disagrees with capacity={capacity}: expected {ENGINE_COUNT * stride} words")
    slices = values.reshape(ENGINE_COUNT, stride)
    counts, overflows = [], []
    for engine, data in enumerate(slices):
        schema, engine_capacity, count, dropped, engine_id = map(int, data[:5])
        if schema != TRACE_SCHEMA:
            raise ValueError(f"engine={engine}: incompatible schema={schema}")
        if engine_capacity != capacity:
            raise ValueError(f"engine={engine}: capacity={engine_capacity} differs from {capacity}")
        if engine_id != engine:
            raise ValueError(f"engine={engine}: invalid engine id={engine_id}")
        if count > capacity or dropped:
            overflows.append(f"engine={engine} count={count} capacity={capacity} dropped={dropped}")
        if any(int(value) for value in data[5:HEADER_WORDS]):
            raise ValueError(f"engine={engine}: nonzero reserved header fields")
        counts.append(count)
    if overflows:
        raise TraceOverflowError("URMA trace overflow; increase --trace-capacity; " + "; ".join(overflows))

    records = []
    for engine, (data, count) in enumerate(zip(slices, counts)):
        previous_end = None
        for ordinal in range(count):
            offset = HEADER_WORDS + ordinal * RECORD_WORDS
            row = tuple(map(int, data[offset : offset + RECORD_WORDS]))
            start, end, event_id = row[:3]
            if event_id not in RAW_EVENT_IDS:
                raise ValueError(f"engine={engine} record={ordinal}: unknown event id={event_id}")
            if end < start or start == UNKNOWN or end == UNKNOWN:
                raise ValueError(f"engine={engine} record={ordinal}: invalid ticks start={start} end={end}")
            if previous_end is not None and start < previous_end:
                raise ValueError(f"engine={engine} record={ordinal}: overlapping or out-of-order event ticks")
            previous_end = end
            if any(row[13:]):
                raise ValueError(f"engine={engine} record={ordinal}: nonzero reserved record fields")
            if row[3] != UNKNOWN and row[3] >= 16:
                raise ValueError(f"engine={engine} record={ordinal}: peer must be in [0,16) or unknown")
            records.append((start, end, engine, ordinal, row))
    if not records:
        raise ValueError("trace has no recorded events")
    origin = min(record[0] for record in records)
    last = max(record[1] for record in records)
    metadata = dict(
        schema=TRACE_SCHEMA,
        rank=rank,
        engine_count=ENGINE_COUNT,
        capacity_per_engine=capacity,
        records_per_engine=counts,
        record_count=len(records),
        dropped=0,
        raw_bytes=int(values.nbytes),
        clock_mhz=float(clock_mhz),
        clock_source="caller-supplied ticks/us; not detected from hardware",
        origin_tick=str(origin),
        last_tick=str(last),
        envelope_ticks=last - origin,
        envelope_us=(last - origin) / clock_mhz,
        clock_alignment="each rank starts at its own earliest event; cross-rank clocks are not synchronized",
        scope="independent diagnostic pass; not the formal performance samples or aggregate bandwidth pass",
        interpretation="PUT/Cube submit bars do not measure asynchronous completion; inspect quiet/drain/wait separately",
        events=dict(Counter(EVENTS[record[4][2]] for record in records)),
    )
    events = [dict(ph="M", pid=rank, tid=0, name="process_name", args=dict(name=f"rank {rank} (independent clock)"))]
    for engine in range(ENGINE_COUNT):
        block, kind = divmod(engine, 3)
        events.extend(
            [
                dict(
                    ph="M",
                    pid=rank,
                    tid=engine,
                    name="thread_name",
                    args=dict(name=f"block {block:02d} {('AIC', 'AIV0', 'AIV1')[kind]}"),
                ),
                dict(ph="M", pid=rank, tid=engine, name="thread_sort_index", args=dict(sort_index=engine)),
            ]
        )
    for start, end, engine, ordinal, row in sorted(records, key=lambda value: (value[0], value[2], value[3])):
        event = EVENTS[row[2]]
        name, category, color, meaning = event_style(event)
        args = {field: None if value == UNKNOWN else value for field, value in zip(METADATA_FIELDS, row[3:12])}
        if event == "LocalSync":
            stage = args.pop("cv")
            args.update(sync_stage=stage, sync_stage_name=SYNC_STAGES.get(stage, "unknown"))
        if row[2] in NATIVE_BATCH_IDS:
            args.update(batch=args.pop("fragment"), timing_scope="device_batch")
            for field in ("row", "rows", "tile_m", "tile_n", "columns"):
                args.pop(field)
        args.update(
            event_id=row[2],
            bytes=row[12],
            record_index=ordinal,
            start_tick=str(start),
            end_tick=str(end),
            elapsed_ticks=end - start,
            meaning=meaning,
        )
        events.append(
            dict(
                ph="X",
                name=name,
                cat=category,
                cname=color,
                pid=rank,
                tid=engine,
                ts=(start - origin) / clock_mhz,
                dur=(end - start) / clock_mhz,
                args=args,
            )
        )
    return dict(traceEvents=events, displayTimeUnit="us", metadata=metadata)


def filter_trace_waits(trace):
    """Hide dependency waits without moving timestamps or losing final barriers."""
    kept, hidden = [], 0
    for event in trace["traceEvents"]:
        if event.get("ph") == "X" and event.get("cat") == "wait.sync":
            name = event["name"]
            final_barrier = name in ("RankBarrier", "FinalPeerHandshake") or (
                name == "LocalSync" and event.get("args", {}).get("sync_stage") == 2
            )
            if not final_barrier:
                hidden += 1
                continue
        kept.append(event)
    metadata = dict(trace["metadata"])
    per_engine = [0] * ENGINE_COUNT
    for event in kept:
        if event.get("ph") == "X":
            per_engine[event["tid"]] += 1
    metadata.update(
        visible_record_count=sum(per_engine),
        visible_records_per_engine=per_engine,
        hidden_wait_count=metadata.get("hidden_wait_count", 0) + hidden,
        wait_filter="dependency waits and local sync hidden; final rank barrier/handshake retained; timestamps unchanged",
    )
    return dict(trace, traceEvents=kept, metadata=metadata)


def _coalesce_tile_drains(trace):
    """Join adjacent AIC tile/drain records without changing device synchronization.

    Group drains belong to a tile only when this engine submitted exactly one
    tile in that group. Record ordinals prevent joining across hidden waits.
    IDs 33/34 are JSON-only composite events, never raw device event IDs.
    """
    lanes = {}
    for index, event in enumerate(trace["traceEvents"]):
        if event.get("ph") == "X" and event["tid"] % 3 == 0:
            lanes.setdefault((event["pid"], event["tid"]), []).append((index, event))

    def context(event):
        return tuple(event["args"].get(key) for key in ("peer", "cv", "fragment", "expert"))

    replacements, removed = {}, set()
    for lane in lanes.values():
        counts = Counter(
            (event["args"]["event_id"], context(event)) for _, event in lane if event["args"]["event_id"] in (15, 19)
        )
        for position, (index, tile) in enumerate(lane):
            event_id = tile["args"]["event_id"]
            if event_id not in (15, 19) or tile["args"].get("fragment") is None:
                continue
            parts = [tile]
            candidates = (16, 27) if event_id == 15 else (21,)
            for expected in candidates:
                following = position + len(parts)
                if following >= len(lane):
                    break
                drain_index, drain = lane[following]
                args = drain["args"]
                if (
                    args["event_id"] != expected
                    or context(drain) != context(tile)
                    or args["record_index"] != parts[-1]["args"]["record_index"] + 1
                    or int(args["start_tick"]) < int(parts[-1]["args"]["end_tick"])
                ):
                    break
                if expected == 16:
                    if any(args.get(key) != tile["args"].get(key) for key in METADATA_FIELDS):
                        break
                elif counts[(event_id, context(tile))] != 1:
                    break
                parts.append(drain)
                removed.add(drain_index)
            if len(parts) == 1:
                continue
            start, end = int(tile["args"]["start_tick"]), int(parts[-1]["args"]["end_tick"])
            args = dict(
                tile["args"],
                event_id=33 if event_id == 15 else 34,
                end_tick=str(end),
                elapsed_ticks=end - start,
                bytes=sum(part["args"].get("bytes", 0) for part in parts),
                meaning="submission through recorded drain end; includes waits/teardown, not pure Cube execution",
                source_records=[
                    {key: part["args"][key] for key in ("event_id", "record_index", "start_tick", "end_tick")}
                    for part in parts
                ],
            )
            replacements[index] = dict(
                tile,
                name=("Gmm1Tile" if event_id == 15 else "Gmm2Tile") + " [compute + drain]",
                cat="cube.compute_drain",
                dur=(end - start) / trace["metadata"]["clock_mhz"],
                args=args,
            )
    if not removed:
        return trace
    events = [
        replacements.get(index, event) for index, event in enumerate(trace["traceEvents"]) if index not in removed
    ]
    counts = [0] * ENGINE_COUNT
    for event in events:
        if event.get("ph") == "X":
            counts[event["tid"]] += 1
    metadata = dict(
        trace["metadata"],
        coalesced_record_count=trace["metadata"].get("coalesced_record_count", 0) + len(removed),
        visible_record_count=sum(counts),
        visible_records_per_engine=counts,
        gmm_display=trace["metadata"].get(
            "gmm_display", "paired AIC tile/drain envelopes; raw counts retained; IDs 33/34 are JSON-only composites"
        ),
    )
    return dict(trace, traceEvents=events, metadata=metadata)


def _source_events(event, clock):
    """Expand legacy tile composites for matching; leave new batches intact."""
    args = event["args"]
    if args["event_id"] not in (33, 34):
        return [event]
    parts = []
    for source in args["source_records"]:
        raw_args = {key: value for key, value in args.items() if key != "source_records"}
        raw_args.update(source)
        raw_args["metadata_inferred_from_legacy_tile"] = True
        start, end = int(source["start_tick"]), int(source["end_tick"])
        name, category, color, meaning = event_style(EVENTS[source["event_id"]])
        raw_args.update(elapsed_ticks=end - start, meaning=meaning)
        parts.append(
            dict(
                event,
                name=name,
                cat=category,
                cname=color,
                ts=event["ts"] + (start - int(args["start_tick"])) / clock,
                dur=(end - start) / clock,
                args=raw_args,
            )
        )
    return parts


def coalesce_gmm_tiles(trace):
    """Display each complete AIC/AIV0 expert context through its shared drain.

    Internal dependency waits stay in source_records, never in active_ticks.
    Missing/filtered records or a stage switch stop matching. No new device
    synchronization is introduced. IDs 35..38 are JSON-only batch envelopes;
    legacy 33/34 tile composites remain readable and can join a complete batch.
    """
    clock = trace["metadata"]["clock_mhz"]
    lanes = {}
    for index, event in enumerate(trace["traceEvents"]):
        if event.get("ph") == "X" and event["tid"] % 3 in (0, 1):
            lane = lanes.setdefault((event["pid"], event["tid"]), [])
            lane.extend((index, part) for part in _source_events(event, clock))

    # engine kind / first event -> terminal drain, permitted intermediate IDs,
    # display ID/name/category. AIV0 has no separate activation-input wait.
    specs = {
        (0, 15): (27, {15, 16, 14, 29}, 35, "Gmm1Batch", "cube.batch"),
        (0, 19): (21, {19, 18}, 36, "Gmm2Batch", "cube.batch"),
        (1, 17): (27, {17}, 37, "Gmm1DequantBatch", "compute.dequant_batch"),
        (1, 20): (21, {20}, 38, "Gmm2DequantBatch", "compute.dequant_batch"),
    }
    replacements, removed = {}, set()
    absorbed_waits = 0
    for (_, engine), lane in lanes.items():
        # A filtered/missing prefix could otherwise make a later tile look
        # like the start of a complete context. Conservatively require full
        # lane provenance before introducing batch semantics. Existing batch
        # records remain opaque for matching but expose their raw ordinals.
        ordinals = [
            record["record_index"]
            for _, event in lane
            for record in (
                event["args"]["source_records"] if event["args"]["event_id"] in (35, 36, 37, 38) else [event["args"]]
            )
        ]
        if ordinals != list(range(len(ordinals))):
            continue
        position = 0
        blocked = set()
        while position < len(lane):
            first_index, first = lane[position]
            first_args = first["args"]
            blocked.discard((first_args["event_id"], first_args.get("expert")))
            spec = specs.get((engine % 3, first_args["event_id"]))
            if spec is None or first_args.get("expert") is None or first_args.get("fragment") is None:
                position += 1
                continue
            terminal, permitted, composite_id, name, category = spec
            if (terminal, first_args["expert"]) in blocked:
                position += 1
                continue
            parts = [(first_index, first)]
            last_submit = first
            finished = False
            for following in range(position + 1, len(lane)):
                index, part = lane[following]
                args, previous = part["args"], parts[-1][1]["args"]
                event_id = args["event_id"]
                if (
                    args.get("expert") != first_args["expert"]
                    or args.get("fragment") is None
                    or args["record_index"] != previous["record_index"] + 1
                    or int(args["start_tick"]) < int(previous["end_tick"])
                    or event_id not in permitted | {terminal}
                ):
                    break
                if event_id == 16 and any(args.get(key) != last_submit["args"].get(key) for key in METADATA_FIELDS):
                    break
                if event_id == terminal and args["fragment"] != last_submit["args"]["fragment"]:
                    break
                parts.append((index, part))
                if event_id == first_args["event_id"]:
                    last_submit = part
                if event_id == terminal:
                    finished = True
                    break
            if not finished:
                blocked.add((terminal, first_args["expert"]))
                position += 1
                continue
            # Never consume only half of a historical composite display event.
            end_position = position + len(parts)
            owners = {index for index, _ in parts}
            if (
                position > 0
                and lane[position - 1][0] in owners
                or end_position < len(lane)
                and lane[end_position][0] in owners
            ):
                position += 1
                continue
            source_records = [dict(part["args"]) for _, part in parts]
            start, end = int(first_args["start_tick"]), int(source_records[-1]["end_tick"])
            active = sum(
                int(part["args"]["end_tick"]) - int(part["args"]["start_tick"])
                for _, part in parts
                if part["cat"] != "wait.sync"
            )
            waiting = sum(
                int(part["args"]["end_tick"]) - int(part["args"]["start_tick"])
                for _, part in parts
                if part["cat"] == "wait.sync"
            )
            batch_args = {key: None for key in METADATA_FIELDS}
            batch_args.update(
                expert=first_args["expert"],
                fragments=list(dict.fromkeys(record["fragment"] for record in source_records)),
                tile_count=sum(record["event_id"] == first_args["event_id"] for record in source_records),
                event_id=composite_id,
                record_index=first_args["record_index"],
                start_tick=str(start),
                end_tick=str(end),
                elapsed_ticks=end - start,
                active_ticks=active,
                dependency_wait_ticks=waiting,
                unrecorded_gap_ticks=end - start - active - waiting,
                bytes=sum(record.get("bytes", 0) for record in source_records),
                meaning="expert context envelope through shared drain; includes dependency waits and scalar gaps; "
                "use source_records for non-wait recorded time, not this envelope",
                source_records=source_records,
            )
            replacements[first_index] = dict(
                first, name=name + " [batch + drain]", cat=category, dur=(end - start) / clock, args=batch_args
            )
            removed.update(owners - {first_index})
            absorbed_waits += sum(part["cat"] == "wait.sync" for _, part in parts)
            position = end_position
    if not replacements:
        return _coalesce_tile_drains(trace)
    events = [
        replacements.get(index, event) for index, event in enumerate(trace["traceEvents"]) if index not in removed
    ]
    counts = [0] * ENGINE_COUNT
    for event in events:
        if event.get("ph") == "X":
            counts[event["tid"]] += 1
    metadata = dict(
        trace["metadata"],
        coalesced_record_count=trace["metadata"].get("coalesced_record_count", 0) + len(removed),
        coalesced_wait_count=trace["metadata"].get("coalesced_wait_count", 0) + absorbed_waits,
        visible_record_count=sum(counts),
        visible_records_per_engine=counts,
        gmm_display="AIC and AIV0 expert batch envelopes through shared drain; "
        "source_records retain waits and original scopes; IDs 35..38 are JSON-only",
    )
    return _coalesce_tile_drains(dict(trace, traceEvents=events, metadata=metadata))


def export_trace(raw, output, *, rank, clock_mhz, include_waits=False, combine_gmm=True):
    """Validate every record before writing the Chrome JSON file."""
    trace = decode_trace(raw, rank=rank, clock_mhz=clock_mhz)
    if combine_gmm:
        trace = coalesce_gmm_tiles(trace)
    if not include_waits:
        trace = filter_trace_waits(trace)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return trace["metadata"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="One rank's urma_trace.bin; not the aggregate urma_timer.bin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--include-waits", action="store_true", help="Restore dependency waits from the raw binary")
    parser.add_argument("--split-gmm-events", action="store_true", help="Keep raw Cube submit/drain bars separate")
    parser.add_argument(
        "--clock-mhz", type=float, required=True, help="Counter ticks/us; use the benchmark's clock setting"
    )
    args = parser.parse_args(argv)
    try:
        metadata = export_trace(
            args.raw.read_bytes(),
            args.output,
            rank=args.rank,
            clock_mhz=args.clock_mhz,
            include_waits=args.include_waits,
            combine_gmm=not args.split_gmm_events,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"URMA trace export failed: {error}\n")
    print(json.dumps(dict(output=str(args.output), **metadata), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
