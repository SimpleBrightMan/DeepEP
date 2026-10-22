"""Trace display coalescing must preserve timing, identity and raw accounting."""

from copy import deepcopy
import json

import pytest

from deep_ep_experimental.megamoe.urma_trace import (
    TRACE_FLAG_DEFERRED_COMPLETION,
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
        (["Gmm1Tile", "Gmm1TileDrain", "Gmm1Drain"], 33),
        (["Gmm2Tile", "Gmm2Drain"], 34),
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
    assert rows[0]["args"]["source_records"] == [
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
        ["Gmm2Tile", "Gmm2Tile", "Gmm2Drain"],
        ["Gmm1Tile", "WaitActivation", "Gmm1TileDrain"],
        ["Gmm2Tile", "WaitActivation", "Gmm2Drain"],
        ["Gmm1Tile"],
    ],
)
def test_shared_drain_and_hidden_waits_are_not_misattributed(names):
    trace = filter_trace_waits(decode_trace(make_trace(names), rank=0, clock_mhz=1000))
    assert coalesce_gmm_tiles(trace) == trace


def test_multi_tile_gmm1_joins_tile_drains_but_keeps_group_drain():
    names = ["Gmm1Tile", "Gmm1TileDrain", "Gmm1Tile", "Gmm1TileDrain", "Gmm1Drain"]
    trace = coalesce_gmm_tiles(decode_trace(make_trace(names), rank=0, clock_mhz=1000))
    assert [event["args"]["event_id"] for event in events(trace)] == [33, 33, 27]


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
    assert [event["args"]["event_id"] for event in rows] == [33, 24, 33, 24]
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


@pytest.mark.parametrize("tile,drain", [("Gmm1Tile", "Gmm1TileDrain"), ("Gmm2Tile", "Gmm2Drain")])
def test_deferred_tile_is_never_joined_to_historical_drain(tile, drain):
    raw = make_trace([tile, drain])
    raw[0, 16 + 13] = TRACE_FLAG_DEFERRED_COMPLETION
    decoded = decode_trace(raw, rank=0, clock_mhz=1000)
    assert coalesce_gmm_tiles(decoded) == decoded


def test_legacy_pair_is_coalesced_in_raw_order_despite_deferred_start_reordering():
    raw = trace_buffer(3)
    add_record(raw, engine=0, event="Gmm1Tile", start=1000, end=1010, fragment=0, expert=0)
    add_record(raw, engine=0, event="Gmm1TileDrain", start=1010, end=1020, fragment=0, expert=0)
    add_record(raw, engine=0, event="Gmm2Tile", start=1005, end=1030, fragment=1, expert=0, flags=1)
    decoded = decode_trace(raw, rank=0, clock_mhz=1000)
    assert [row["args"]["record_index"] for row in events(decoded)] == [0, 2, 1]
    combined = coalesce_gmm_tiles(decoded)
    assert [row["args"]["event_id"] for row in events(combined)] == [33, 19]
    assert events(combined)[0]["dur"] == 0.02
    assert events(combined)[1]["args"]["completion_deferred"]
    assert combined["metadata"]["record_count"] == 3
    assert combined["metadata"]["coalesced_record_count"] == 1
    assert coalesce_gmm_tiles(combined) == combined


def test_merge_legacy_and_deferred_rank_traces_keeps_independent_semantics(tmp_path):
    paths = [tmp_path / f"rank_{rank}.json" for rank in range(2)]
    export_trace(make_trace(["Gmm1Tile", "Gmm1TileDrain"]), paths[0], rank=0, clock_mhz=1000)
    raw = trace_buffer(2)
    add_record(raw, engine=0, event="Gmm1Tile", start=1000, end=1040, fragment=0, flags=1)
    add_record(raw, engine=0, event="Gmm2Tile", start=1030, end=1080, fragment=0, flags=1)
    export_trace(raw, paths[1], rank=1, clock_mhz=1000)
    output = tmp_path / "allrank.json"
    meta = merge_trace_files(paths, output, world=2)
    rows = events(json.loads(output.read_text()))
    assert [(row["pid"], row["args"]["event_id"]) for row in rows] == [(0, 33), (1, 15), (1, 19)]
    assert all(row["args"].get("completion_deferred") for row in rows[1:])
    assert meta["record_count"] == 4 and meta["visible_record_count"] == 3
    assert meta["coalesced_record_count"] == 1
