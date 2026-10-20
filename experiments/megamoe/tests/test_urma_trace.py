import json
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_trace import (
    ENGINE_COUNT,
    EVENTS,
    HEADER_WORDS,
    RECORD_WORDS,
    UNKNOWN,
    TraceOverflowError,
    decode_trace,
    export_trace,
    main,
)


def trace_buffer(capacity=4):
    raw = np.zeros((ENGINE_COUNT, HEADER_WORDS + capacity * RECORD_WORDS), dtype=np.uint64)
    raw[:, 0] = 1
    raw[:, 1] = capacity
    raw[:, 4] = np.arange(ENGINE_COUNT)
    return raw


def add_record(raw, engine=2, event="DispatchPutX", start=1000, end=1010, **metadata):
    index = int(raw[engine, 2])
    record = raw[engine, HEADER_WORDS + index * RECORD_WORDS : HEADER_WORDS + (index + 1) * RECORD_WORDS]
    record[:3] = [start, end, EVENTS.index(event)]
    record[3:12] = UNKNOWN
    fields = ("peer", "cv", "fragment", "expert", "row", "rows", "tile_m", "tile_n", "columns", "bytes")
    for key, value in metadata.items():
        record[3 + fields.index(key)] = value
    raw[engine, 2] += 1
    return record


def events(value):
    return [event for event in value["traceEvents"] if event["ph"] == "X"]


def test_export_fields_thread_mapping_and_submission_semantics():
    raw = trace_buffer()
    add_record(raw, peer=4, cv=2, fragment=3, expert=9, row=256, rows=128, columns=3584, bytes=458752)
    add_record(raw, event="DispatchQuiet", start=1020, end=1100, peer=4)
    add_record(raw, engine=3, event="Gmm1Tile", start=1011, end=1021, tile_m=0, tile_n=256)
    result = decode_trace(raw, rank=7, clock_mhz=1000)
    rows = events(result)
    assert [(row["tid"], row["name"]) for row in rows] == [
        (2, "DispatchPutX [PUT submit]"),
        (3, "Gmm1Tile [Cube submit]"),
        (2, "DispatchQuiet [completion wait]"),
    ]
    assert rows[0]["pid"] == 7 and rows[0]["ts"] == 0 and rows[0]["dur"] == 0.01
    assert rows[0]["args"]["bytes"] == 458752 and rows[0]["args"]["rows"] == 128
    assert rows[0]["args"]["tile_m"] is None
    assert "not remote completion" in rows[0]["args"]["meaning"]
    assert "not Cube completion" in rows[1]["args"]["meaning"]
    names = {row["tid"]: row["args"]["name"] for row in result["traceEvents"] if row["name"] == "thread_name"}
    assert names[0] == "block 00 AIC" and names[1] == "block 00 AIV0" and names[2] == "block 00 AIV1"
    assert names[83] == "block 27 AIV1"
    assert result["metadata"]["record_count"] == 3


def test_every_event_id_has_a_name_and_no_fabricated_async_events():
    raw = trace_buffer(1)
    for event_id, name in enumerate(EVENTS):
        add_record(raw, engine=event_id, event=name)
    result = decode_trace(raw, rank=0, clock_mhz=1000)
    assert len(events(result)) == len(EVENTS) == 33
    assert {row["ph"] for row in result["traceEvents"]} == {"M", "X"}
    assert {row["args"]["event_id"] for row in events(result)} == set(range(len(EVENTS)))


def test_historical_tail_cover_events_keep_completion_signal_and_hide_only_ready_wait():
    from deep_ep_experimental.megamoe.urma_trace import filter_trace_waits

    raw = trace_buffer(4)
    add_record(raw, engine=2, event="CombinePublish", start=1000, end=1010, peer=3, cv=2)
    add_record(raw, engine=1, event="WaitReduceReady", start=1000, end=1015, row=7, rows=1)
    add_record(raw, engine=1, event="Reduce", start=1015, end=1020, row=7, rows=1)
    add_record(raw, engine=1, event="ReduceDrain", start=1020, end=1030)
    filtered = events(filter_trace_waits(decode_trace(raw, rank=0, clock_mhz=1000)))
    assert [row["args"]["event_id"] for row in filtered] == [30, 25, 32]
    assert filtered[0]["cat"] == "signal.publish"
    assert filtered[1]["ts"] == 0.015


def test_reduce_envelope_is_one_visible_interval_including_internal_waits():
    from deep_ep_experimental.megamoe.urma_trace import filter_trace_waits

    raw = trace_buffer(1)
    add_record(raw, engine=1, event="Reduce", start=1000, end=901000, row=100, rows=100, bytes=100 * 3584 * 2)
    filtered = events(filter_trace_waits(decode_trace(raw, rank=0, clock_mhz=1000)))
    assert len(filtered) == 1
    event = filtered[0]
    assert event["name"] == "Reduce" and event["dur"] == 900.0
    assert event["args"]["rows"] == 100 and event["args"]["bytes"] == 716800
    assert "readiness waits" in event["args"]["meaning"] and "output drain" in event["args"]["meaning"]


def test_expert_group_trace_keeps_multisource_identity_and_scale_preparation():
    raw = trace_buffer()
    add_record(raw, engine=47, event="InputScaleGather", start=1000, end=1020, fragment=0, expert=0, rows=256)
    add_record(raw, engine=0, event="WaitInputScale", start=1005, end=1021, fragment=0, expert=0)
    add_record(raw, engine=0, event="Gmm1Tile", start=1021, end=1050, fragment=0, expert=0, rows=256, columns=256)
    decoded = events(decode_trace(raw, rank=0, clock_mhz=1000))
    assert [row["cat"] for row in decoded] == ["compute.copy", "wait.sync", "cube.submit"]
    assert all(row["args"]["peer"] is None and row["args"]["cv"] is None for row in decoded)


def test_local_sync_cv_field_is_a_named_sync_stage():
    raw = trace_buffer()
    add_record(raw, event="LocalSync", cv=2)
    event = events(decode_trace(raw, rank=0, clock_mhz=1000))[0]
    assert "cv" not in event["args"]
    assert event["args"]["sync_stage"] == 2 and event["args"]["sync_stage_name"] == "rank barrier"
    assert event["cname"] == "thread_state_iowait"


@pytest.mark.parametrize(
    "engine,gmm1_event,gmm2_event",
    [(0, "Gmm1Tile", "Gmm2Tile"), (1, "Gmm1Dequant", "Gmm2Dequant")],
)
def test_cv_lookahead_trace_preserves_reordered_fragment_and_peer_identity(engine, gmm1_event, gmm2_event):
    # One source CV may contain several expert fragments; the next source's
    # cv_id restarts at zero. Neither CV nor fragment IDs are time coordinates.
    sequence = [
        (gmm1_event, 0, 0, 0, 0),
        (gmm1_event, 0, 0, 1, 1),
        (gmm1_event, 0, 1, 2, 1),
        (gmm2_event, 0, 0, 0, 0),
        (gmm2_event, 0, 0, 1, 1),
        (gmm1_event, 1, 0, 3, 0),
        (gmm2_event, 0, 1, 2, 1),
        (gmm2_event, 1, 0, 3, 0),
    ]
    raw = trace_buffer(len(sequence))
    for index, (name, peer, cv, fragment, expert) in enumerate(sequence):
        add_record(
            raw,
            engine=engine,
            event=name,
            start=1000 + index * 20,
            end=1010 + index * 20,
            peer=peer,
            cv=cv,
            fragment=fragment,
            expert=expert,
            tile_m=0,
            tile_n=0,
        )
    result = decode_trace(raw, rank=7, clock_mhz=1000)
    rows = events(result)
    assert [
        (
            EVENTS[row["args"]["event_id"]],
            row["args"]["peer"],
            row["args"]["cv"],
            row["args"]["fragment"],
            row["args"]["expert"],
        )
        for row in rows
    ] == sequence
    assert [row["ts"] for row in rows] == [index * 0.02 for index in range(len(sequence))]
    assert all(row["tid"] == engine and row["dur"] == 0.01 for row in rows)
    assert result["metadata"]["record_count"] == len(sequence)


@pytest.mark.parametrize("origin", [(1 << 53) + 101, (1 << 63) + 101, UNKNOWN - 100])
def test_large_ticks_are_subtracted_as_integers_before_float_conversion(origin):
    raw = trace_buffer()
    add_record(raw, start=origin, end=origin + 1)
    add_record(raw, start=origin + 3, end=origin + 10)
    decoded = decode_trace(raw.view(np.int64), rank=0, clock_mhz=1000)
    rows = events(decoded)
    assert [row["ts"] for row in rows] == [0, 0.003]
    assert [row["dur"] for row in rows] == [0.001, 0.007]
    assert decoded["metadata"]["origin_tick"] == str(origin)
    assert decoded["metadata"]["envelope_ticks"] == 10


def test_each_rank_is_zeroed_independently():
    first, second = trace_buffer(), trace_buffer()
    add_record(first, start=1234, end=1244)
    add_record(second, start=999999, end=1000009)
    a, b = decode_trace(first, rank=0, clock_mhz=1000), decode_trace(second, rank=15, clock_mhz=1000)
    assert events(a)[0]["ts"] == events(b)[0]["ts"] == 0
    assert a["metadata"]["origin_tick"] != b["metadata"]["origin_tick"]
    assert "not synchronized" in a["metadata"]["clock_alignment"]


def test_uint64_bytes_and_signed_array_decode_identically():
    raw = trace_buffer()
    add_record(raw, start=(1 << 63) + 1, end=(1 << 63) + 17)
    expected = decode_trace(raw.view(np.int64), rank=0, clock_mhz=1000)
    assert decode_trace(raw.astype("<u8").tobytes(), rank=0, clock_mhz=1000) == expected


@pytest.mark.parametrize(
    "field,value,message",
    [
        (0, 2, "schema"),
        (1, 5, "capacity"),
        (2, 5, "overflow"),
        (3, 1, "overflow"),
        (4, 80, "engine id"),
        (5, 1, "reserved header"),
    ],
)
def test_reject_corrupt_engine_header(field, value, message):
    raw = trace_buffer()
    add_record(raw)
    raw[7, field] = value
    with pytest.raises(ValueError, match=message):
        decode_trace(raw, rank=0, clock_mhz=1000)


def test_overflow_lists_all_affected_engines_and_writes_no_partial_trace(tmp_path):
    raw = trace_buffer()
    add_record(raw)
    raw[0, 3], raw[83, 3] = 2, 9
    output = tmp_path / "trace.json"
    with pytest.raises(TraceOverflowError, match=r"engine=0.*engine=83"):
        export_trace(raw, output, rank=0, clock_mhz=1000)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        (1, 999, "ticks"),
        (0, UNKNOWN, "ticks"),
        (1, UNKNOWN, "ticks"),
        (2, len(EVENTS), "event id"),
        (3, 16, "peer"),
        (13, 1, "reserved record"),
    ],
)
def test_reject_invalid_record(field, value, message):
    raw = trace_buffer()
    record = add_record(raw)
    record[field] = value
    with pytest.raises(ValueError, match=message):
        decode_trace(raw, rank=0, clock_mhz=1000)


def test_zero_duration_is_valid_and_unused_records_are_ignored():
    raw = trace_buffer()
    add_record(raw, start=123, end=123)
    raw[2, HEADER_WORDS + RECORD_WORDS :] = UNKNOWN
    assert events(decode_trace(raw, rank=0, clock_mhz=1000))[0]["dur"] == 0


@pytest.mark.parametrize("second_start", [900, 1005])
def test_same_engine_records_cannot_overlap_or_go_backwards(second_start):
    raw = trace_buffer()
    add_record(raw, start=1000, end=1010)
    add_record(raw, start=second_start, end=1020)
    with pytest.raises(ValueError, match="overlapping or out-of-order"):
        decode_trace(raw, rank=0, clock_mhz=1000)


@pytest.mark.parametrize("capacity", [0, 65537])
def test_invalid_capacity_is_rejected_before_record_size_calculation(capacity):
    raw = trace_buffer()
    raw[0, 1] = capacity
    with pytest.raises(ValueError, match="capacity"):
        decode_trace(raw, rank=0, clock_mhz=1000)


@pytest.mark.parametrize(
    "raw,message",
    [
        (b"abc", "multiple of 8"),
        (np.zeros(8, dtype=np.uint64), "headers"),
        (np.zeros(100, dtype=np.float64), "integer"),
        (np.zeros(100, dtype=np.int32), "64-bit"),
        (trace_buffer().reshape(-1)[:-1], "size"),
        (trace_buffer(), "no recorded"),
    ],
)
def test_reject_bad_buffer(raw, message):
    with pytest.raises(ValueError, match=message):
        decode_trace(raw, rank=0, clock_mhz=1000)


@pytest.mark.parametrize("rank", [-1, 16, True, 0.5])
def test_reject_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        decode_trace(trace_buffer(), rank=rank, clock_mhz=1000)


@pytest.mark.parametrize("clock", [0, -1, float("inf"), float("nan"), True])
def test_reject_invalid_clock(clock):
    with pytest.raises(ValueError, match="clock"):
        decode_trace(trace_buffer(), rank=0, clock_mhz=clock)


def test_offline_cli_writes_chrome_trace_without_npu(tmp_path, capsys):
    raw = trace_buffer()
    add_record(raw)
    binary, output = tmp_path / "raw.bin", tmp_path / "trace.json"
    raw.astype("<u8").tofile(binary)
    main([str(binary), "--output", str(output), "--rank", "3", "--clock-mhz", "1000"])
    assert json.loads(capsys.readouterr().out)["rank"] == 3
    assert json.loads(output.read_text())["metadata"]["rank"] == 3


def test_offline_cli_overflow_exits_nonzero(tmp_path, capsys):
    raw = trace_buffer()
    raw[0, 3] = 1
    binary = tmp_path / "raw.bin"
    raw.astype("<u8").tofile(binary)
    with pytest.raises(SystemExit) as error:
        main([str(binary), "--output", str(tmp_path / "out.json"), "--rank", "0", "--clock-mhz", "1000"])
    assert error.value.code == 2 and "overflow" in capsys.readouterr().err


def test_benchmark_trace_implies_timer_and_validates_capacity(tmp_path):
    from deep_ep_experimental.megamoe.urma_benchmark import parse_args

    args = parse_args(["--output-dir", str(tmp_path), "--trace"])
    assert args.trace and args.timer and args.trace_capacity == 4096
    plain = parse_args(["--output-dir", str(tmp_path)])
    assert not plain.timer and not plain.trace
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--trace-capacity", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--trace-capacity", "65537"])


def test_local_overflow_is_exchanged_before_raising_and_keeps_raw(tmp_path):
    from deep_ep_experimental.megamoe.urma_benchmark import export_trace_collective

    raw = trace_buffer()
    raw[3, 3] = 9
    exchanged = []

    def gather(status, stage):
        exchanged.append((status, stage))
        return [status, dict(rank=1, error="other rank overflow")]

    group = SimpleNamespace(rank=lambda: 0, all_gather=gather)
    with pytest.raises(RuntimeError, match=r"rank=0:.*rank=1:"):
        export_trace_collective(group, raw, tmp_path, bs=4096, clock_mhz=1000)
    assert len(exchanged) == 1 and "overflow" in exchanged[0][0]["error"]
    assert (tmp_path / "urma_trace.bin").exists() and not (tmp_path / "urma_trace.json").exists()
    assert json.loads((tmp_path / "trace_metadata.json").read_text())["error"]


def test_peer_overflow_causes_valid_local_rank_to_fail_after_exchange(tmp_path):
    from deep_ep_experimental.megamoe.urma_benchmark import export_trace_collective

    raw = trace_buffer()
    add_record(raw)
    group = SimpleNamespace(
        rank=lambda: 0, all_gather=lambda status, stage: [status, dict(rank=7, error="trace overflow")]
    )
    with pytest.raises(RuntimeError, match="rank=7"):
        export_trace_collective(group, raw, tmp_path, bs=2048, clock_mhz=1000)
    assert (tmp_path / "urma_trace.json").exists()
