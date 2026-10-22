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
