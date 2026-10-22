"""Trace display coalescing must preserve timing, identity and raw accounting."""

from copy import deepcopy
import json

import pytest

from deep_ep_experimental.megamoe.urma_trace import (
    coalesce_gmm_tiles,
    decode_trace,
    export_trace,
    filter_trace_waits,
)
from deep_ep_experimental.megamoe.urma_trace_merge import merge_trace_files
from test_urma_trace import add_record, events, trace_buffer


def make_trace(names, *, engine=0, origin=2**60, fragment=4):
    raw = trace_buffer(len(names))
    for index, name in enumerate(names):
        add_record(
            raw,
            engine=engine,
            event=name,
            start=origin + 100 * index,
            end=origin + 100 * index + 50,
            fragment=fragment,
            expert=2,
        )
    return raw


@pytest.mark.parametrize(
    "names,composite",
    [
        (["Gmm1Tile", "Gmm1TileDrain"], 33),
        (["Gmm1Tile", "Gmm1TileDrain", "Gmm1Drain"], 35),
        (["Gmm2Tile", "Gmm2Drain"], 36),
    ],
)
def test_tile_envelope_keeps_integer_ticks_and_provenance(names, composite):
    original = decode_trace(make_trace(names), rank=0, clock_mhz=1000)
    snapshot = deepcopy(original)
    combined = coalesce_gmm_tiles(original)
    rows = events(combined)
    assert len(rows) == 1
    assert rows[0]["ts"] == 0
    assert rows[0]["dur"] == (100 * (len(names) - 1) + 50) / 1000
    assert rows[0]["args"]["event_id"] == composite
    assert [
        {key: source[key] for key in ("event_id", "record_index", "start_tick", "end_tick")}
        for source in rows[0]["args"]["source_records"]
    ] == [
        {key: event["args"][key] for key in ("event_id", "record_index", "start_tick", "end_tick")}
        for event in events(original)
    ]
    assert combined["metadata"]["coalesced_record_count"] == len(names) - 1
    assert combined["metadata"]["record_count"] == len(names)
    assert combined["metadata"]["envelope_ticks"] == original["metadata"]["envelope_ticks"]
    assert coalesce_gmm_tiles(combined) == combined
    assert original == snapshot


@pytest.mark.parametrize(
    "names",
    [
        ["Gmm1Tile", "WaitActivation", "Gmm1TileDrain"],
        ["Gmm2Tile", "WaitActivation", "Gmm2Drain"],
        ["Gmm1Tile"],
    ],
)
def test_shared_drain_and_hidden_waits_are_not_misattributed(names):
    trace = filter_trace_waits(decode_trace(make_trace(names), rank=0, clock_mhz=1000))
    assert coalesce_gmm_tiles(trace) == trace


def test_multi_tile_gmm1_joins_shared_context_drain_once():
    names = ["Gmm1Tile", "Gmm1TileDrain", "Gmm1Tile", "Gmm1TileDrain", "Gmm1Drain"]
    trace = coalesce_gmm_tiles(decode_trace(make_trace(names), rank=0, clock_mhz=1000))
    assert [event["args"]["event_id"] for event in events(trace)] == [35]
    assert events(trace)[0]["args"]["tile_count"] == 2


@pytest.mark.parametrize("engine", [1, 2])
def test_aiv_scopes_are_untouched(engine):
    trace = decode_trace(make_trace(["Gmm1Tile", "Gmm1TileDrain"], engine=engine), rank=0, clock_mhz=1000)
    assert coalesce_gmm_tiles(trace) == trace


@pytest.mark.parametrize("field,value", [("fragment", 5), ("expert", 3), ("tile_n", 1)])
def test_different_tile_metadata_does_not_join(field, value):
    trace = decode_trace(make_trace(["Gmm1Tile", "Gmm1TileDrain"]), rank=0, clock_mhz=1000)
    events(trace)[1]["args"][field] = value
    assert coalesce_gmm_tiles(trace) == trace


def test_merge_old_and_new_rank_json_and_reject_lost_record(tmp_path):
    raw = make_trace(["Gmm1Tile", "Gmm1TileDrain", "Gmm1Drain", "WaitActivation", "RankBarrier"])
    paths = [tmp_path / f"rank_{rank}.json" for rank in range(2)]
    for rank, path in enumerate(paths):
        export_trace(raw, path, rank=rank, clock_mhz=1000, combine_gmm=bool(rank))
    output = tmp_path / "allrank.json"
    meta = merge_trace_files(paths, output, world=2)
    assert meta["record_count"] == 10
    assert meta["visible_record_count"] == 4
    assert meta["coalesced_record_count"] == 4
    assert meta["hidden_wait_count"] == 2
    rows = events(json.loads(output.read_text()))
    # The old rank has already filtered a wait: keep its provable tile
    # composite, rather than infer complete batches from incomplete lanes.
    assert [event["args"]["event_id"] for event in rows] == [33, 24, 35, 24]
    # A real missing record must not be excused by display coalescing.
    broken = json.loads(paths[1].read_text())
    broken["traceEvents"] = [event for event in broken["traceEvents"] if event.get("name") != "RankBarrier"]
    paths[1].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="record count mismatch"):
        merge_trace_files(paths, output, world=2)


def test_export_split_option_preserves_raw_scopes(tmp_path):
    path = tmp_path / "split.json"
    export_trace(make_trace(["Gmm2Tile", "Gmm2Drain"]), path, rank=0, clock_mhz=1000, combine_gmm=False)
    assert [row["args"]["event_id"] for row in events(json.loads(path.read_text()))] == [19, 21]


@pytest.mark.parametrize(
    "engine,submit,wait,drain,composite",
    [
        (0, "Gmm1Tile", "WaitInputScale", "Gmm1Drain", 35),
        (0, "Gmm2Tile", "WaitActivation", "Gmm2Drain", 36),
        (1, "Gmm1Dequant", None, "Gmm1Drain", 37),
        (1, "Gmm2Dequant", None, "Gmm2Drain", 38),
    ],
)
def test_expert_batch_preserves_scopes_waits_and_scalar_gaps(engine, submit, wait, drain, composite):
    names = [submit, *([wait] if wait else []), submit, drain]
    trace = decode_trace(make_trace(names, engine=engine), rank=7, clock_mhz=1000)
    source = events(trace)
    for record in source[1:]:
        record["args"]["fragment"] = 9
    original = deepcopy(trace)
    combined = coalesce_gmm_tiles(trace)
    rows = events(combined)
    assert len(rows) == 1
    args = rows[0]["args"]
    assert args["event_id"] == composite and args["expert"] == 2
    assert args["fragment"] is None and args["tile_n"] is None and args["rows"] is None
    assert args["fragments"] == [4, 9] and args["tile_count"] == 2
    assert args["source_records"] == [record["args"] for record in source]
    assert args["active_ticks"] == 150
    assert args["dependency_wait_ticks"] == (50 if wait else 0)
    assert args["unrecorded_gap_ticks"] == 50 * (len(names) - 1)
    assert args["elapsed_ticks"] == sum(
        args[key] for key in ("active_ticks", "dependency_wait_ticks", "unrecorded_gap_ticks")
    )
    assert combined["metadata"]["record_count"] == len(names)
    assert combined["metadata"]["coalesced_record_count"] == len(names) - 1
    assert filter_trace_waits(combined)["metadata"]["hidden_wait_count"] == 0
    assert coalesce_gmm_tiles(combined) == combined and trace == original


def test_consecutive_same_expert_batches_do_not_share_a_drain():
    trace = decode_trace(make_trace(["Gmm2Tile", "Gmm2Drain"] * 2), rank=0, clock_mhz=1000)
    rows = events(coalesce_gmm_tiles(trace))
    assert [row["args"]["event_id"] for row in rows] == [36, 36]
    assert [row["args"]["record_index"] for row in rows] == [0, 2]


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_missing_batch_record_never_coalesces_a_remaining_tail(missing):
    trace = decode_trace(make_trace(["Gmm2Tile", "WaitActivation", "Gmm2Tile", "Gmm2Drain"]), rank=0, clock_mhz=1000)
    removed = events(trace)[missing]
    trace["traceEvents"].remove(removed)
    assert all(row["args"]["event_id"] < 35 for row in events(coalesce_gmm_tiles(trace)))


def test_illegal_stage_does_not_restart_a_batch_at_its_tail():
    trace = decode_trace(make_trace(["Gmm2Tile", "Gmm1Tile", "Gmm2Tile", "Gmm2Drain"]), rank=0, clock_mhz=1000)
    assert coalesce_gmm_tiles(trace) == trace


def test_batch_cannot_cross_experts_or_engines():
    trace = decode_trace(make_trace(["Gmm2Tile", "Gmm2Drain"]), rank=0, clock_mhz=1000)
    events(trace)[1]["args"]["expert"] = 3
    assert coalesce_gmm_tiles(trace) == trace
    events(trace)[1]["args"]["expert"] = 2
    events(trace)[1]["tid"] = 3
    assert coalesce_gmm_tiles(trace) == trace


def test_legacy_composite_can_join_batch_without_losing_or_double_counting_records():
    from deep_ep_experimental.megamoe.urma_trace import _coalesce_tile_drains

    names = ["Gmm1Tile", "Gmm1TileDrain", "WaitInputScale", "Gmm1Tile", "Gmm1TileDrain", "Gmm1Drain"]
    trace = decode_trace(make_trace(names), rank=0, clock_mhz=1000)
    for row in events(trace)[2:]:
        row["args"]["fragment"] = 9
    legacy = _coalesce_tile_drains(trace)
    assert [row["args"]["event_id"] for row in events(legacy)] == [33, 29, 33]
    combined = coalesce_gmm_tiles(legacy)
    row = events(combined)[0]
    assert row["args"]["event_id"] == 35
    assert len(row["args"]["source_records"]) == 6
    assert [part["record_index"] for part in row["args"]["source_records"]] == list(range(6))
    assert combined["metadata"]["record_count"] == 6
    assert combined["metadata"]["coalesced_record_count"] == 5
    assert coalesce_gmm_tiles(combined) == combined
