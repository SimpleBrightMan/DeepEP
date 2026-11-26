# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU unit tests for TP MegaMoE public API, timer protocol, breakdown, and soft-sync wait bubbles."""

from pathlib import Path
import struct

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe.timer import protocol as p
from deep_ep_experimental.megamoe.timer.breakdown import compact_events, rank_breakdown
from deep_ep_experimental.megamoe.timer.parse_timer_csv import decode
from deep_ep_experimental.megamoe.timer.trace_parser import (
    first_start_cycle,
    raw_events_for_rank,
    trace_events,
)


def test_tp_timer_events_and_wait_bubbles(tmp_path: Path):
    raw_path = tmp_path / "rank_0_npu_time.bin"
    with raw_path.open("wb") as stream:
        stream.truncate(p.TOTAL_BUFFER_SIZE * 8)

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        # Fixed anchor
        sync_idx = p.FIXED_NAMES.index("AIV_INPUT_CROSS_RANK_SYNC")
        put(1 * p.COUNTERS_PER_CORE + sync_idx * 2, 10_000, 12_000)

        # Dynamic TP events across AIC (core 0), AIV0 (core 1), AIV1 (core 2), and Route AIV (core 67)
        samples = [
            # Phase 1: AIC Shared GMM1 + AIV0 AllGather + AIV1 RouteTable split by AIV_ROUTE_WAIT
            (0, "AIC_SHARED_GMM1", 12_000, 30_000, -1, (1 << 32) | 0, (1 << 32) | 256),
            (1, "AIV_ALLGATHER", 12_500, 35_000, -1, 0, 0),
            (67, "AIV_ROUTE_TABLE", 12_500, 18_000, -1, 0, 0),
            (67, "AIV_ROUTE_WAIT", 18_000, 21_000, -1, 0, 0),
            (67, "AIV_ROUTE_TABLE", 21_000, 28_000, -1, 0, 0),
            # Phase 2: Wave 0 Permute (WAVE granularity), AIV0 GMM1/GMM2 Prologue (tile), AIC GMM1/GMM2, AIV1 SwiGLU
            (2, "AIV_PERMUTE", 28_500, 40_000, 0, 0, 0),
            (1, "AIV_GMM1_PROLOGUE", 36_000, 42_000, 0, (1 << 32) | 0, (1 << 32) | 256),
            (0, "AIC_GMM1_WAIT", 36_000, 40_000, 0, (1 << 32) | 0, (1 << 32) | 256),
            (0, "AIC_GMM1_ISSUE", 40_000, 48_000, 0, (1 << 32) | 0, (1 << 32) | 256),
            (2, "AIV_SWIGLU_ISSUE", 48_500, 53_000, 0, (1 << 32) | 0, (1 << 32) | 256),
            (1, "AIV_GMM2_PROLOGUE", 50_000, 58_000, 0, (1 << 32) | 0, (1 << 32) | 256),
            (0, "AIC_GMM2_WAIT", 50_000, 53_500, 0, (1 << 32) | 0, (1 << 32) | 256),
            (0, "AIC_GMM2_ISSUE", 53_500, 65_000, 0, (1 << 32) | 0, (1 << 32) | 256),
            # Phase 3: AIC Shared GMM2 + AIV Unpermute+Send + Recv+Reduce split by AIV_RECV_REDUCE_WAIT
            (0, "AIC_SHARED_GMM2", 66_000, 82_000, -1, (1 << 32) | 0, (1 << 32) | 256),
            (1, "AIV_UNPERMUTE_SEND", 66_000, 78_000, -1, 0, 0),
            (1, "AIV_RECV_REDUCE_WAIT", 78_000, 82_500, -1, 0, 0),
            (1, "AIV_RECV_REDUCE", 82_500, 90_000, -1, 0, 0),
        ]
        counts: dict[tuple[int, int], int] = {}
        for core, name, start, end, wave, m0, m1 in samples:
            fam = p.DYNAMIC_NAMES.index(name)
            slot = counts.get((core, fam), 0)
            counts[(core, fam)] = slot + 1
            logical = len(p.FIXED_NAMES) + fam * p.MAX_DYNAMIC_ITER + slot
            put(core * p.COUNTERS_PER_CORE + logical * 2, start, end)
            put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + fam, slot + 1)
            if m0 or m1:
                meta = p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + fam * p.MAX_DYNAMIC_ITER + slot) * 2
                put(meta, m0, m1)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, wave)

    parsed_csv = tmp_path / "rank_0_npu_time_parsed.csv"
    decode(raw_path, parsed_csv)
    events = raw_events_for_rank(parsed_csv, 0)
    trace = trace_events(events, "keep", first_start_cycle(events), include_wait=False, time_origin="rank")

    # Ensure zero WAIT bars are emitted when include_wait=False (soft-sync waits render as blank bubbles)
    assert not [ev for ev in trace if "WAIT" in ev["name"]]

    # Verify RouteTable was split into 2 segments with a 3.0 us blank bubble [18_000, 21_000]
    route_evs = [ev for ev in trace if ev["args"].get("event_family") == "AIV_ROUTE_TABLE"]
    assert len(route_evs) == 2
    assert route_evs[1]["ts"] - (route_evs[0]["ts"] + route_evs[0]["dur"]) == pytest.approx(3.0)

    # Verify Wave 0 Permute carries wave_0 label
    perm_evs = [ev for ev in trace if ev["args"].get("event_family") == "AIV_PERMUTE"]
    assert len(perm_evs) == 1 and perm_evs[0]["name"] == "wave_0_AIV_PERMUTE_0"

    # Verify TP breakdown succeeds
    raw_ints = np.fromfile(raw_path, dtype="<i8")
    bd = rank_breakdown(compact_events(raw_ints, 0), 0)
    assert bd["status"] == "ok" and bd["mode"] == "tp"
