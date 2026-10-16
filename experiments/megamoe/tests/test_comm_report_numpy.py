# Copyright (c) 2026, SimpleBright_Man

"""Exercise communication report serialization with the real NumPy timer dtype."""

import json

import numpy as np

from deep_ep._store import StoreGroup
from deep_ep_experimental.megamoe import comm_report
from deep_ep_experimental.megamoe.timer import protocol as p


class SparseNumpyTimer:
    """Avoid allocating a full timer buffer while preserving NumPy indexing types."""

    def __init__(self):
        self.values = {}

    def __len__(self):
        return p.TOTAL_BUFFER_SIZE

    def __getitem__(self, index):
        if isinstance(index, slice):
            return np.array([self.values.get(i, 0) for i in range(index.start, index.stop)], dtype=np.int64)
        return np.int64(self.values.get(index, 0))


class SingleRankStore:
    """Synchronous store for the actual StoreGroup JSON encoder/decoder path."""

    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.values[key]

    def check(self, keys):
        return all(key in self.values for key in keys)

    def wait(self, keys, timeout):
        assert self.check(keys)

    def add(self, key, value):
        result = int(self.values.get(key, b"0")) + value
        self.set(key, str(result))
        return result

    def delete_key(self, key):
        self.values.pop(key, None)


def numpy_timer():
    raw = SparseNumpyTimer()
    for family in comm_report.FAMILIES:
        operation, kind = family.removeprefix("AIV_").split("_")
        split = 10 if operation == "DISPATCH" else 12
        core = 2 if kind == "INTRA" else split * 3 + 2
        index = p.DYNAMIC_NAMES.index(family)
        raw.values[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + index] = 2
        rows = 16 if kind == "INTRA" else 48
        for wave in (0, 1):
            logical = len(p.FIXED_NAMES) + index * p.MAX_DYNAMIC_ITER + wave
            offset = core * p.COUNTERS_PER_CORE + logical * 2
            raw.values[offset] = 1000 + wave * 100000
            raw.values[offset + 1] = 11000 + wave * 100000
            metadata = (
                p.TASK_METADATA_OFFSET
                + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + logical - len(p.FIXED_NAMES)) * 2
            )
            raw.values[metadata] = (1 << 32) | (wave * rows)
            raw.values[metadata + 1] = rows
            raw.values[p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical] = wave
    return raw


def test_numpy_timer_reports_survive_store_gather_and_json_summary():
    raw = numpy_timer()
    assert isinstance(raw[0], np.int64)
    assert raw[0:2].dtype == np.int64
    # One local Store participant exercises the same JSON path as all 16 workers;
    # the communication report itself still contains all 16 distinct rank reports.
    group = StoreGroup(SingleRankStore(), 0, 1, prefix="numpy-timer-report")
    records = []
    for rank in range(16):
        events = comm_report.read_events(raw, rank)
        assert len(events) == 8
        report = comm_report.rank_report(events, rank, 16, 3584, 8, 10, 12)
        records.extend(group.all_gather(report, f"communication report rank={rank}"))
        for event in events:
            for key in (
                "rank",
                "core",
                "wave",
                "start_cycle",
                "end_cycle",
                "expert_id",
                "row_start",
                "row_count",
                "m_tile",
            ):
                assert type(event[key]) is int, (key, type(event[key]))
    summary = comm_report.summarize(records)
    decoded = json.loads(json.dumps(summary, allow_nan=False))
    assert decoded == summary
    assert [record["rank"] for record in decoded["ranks"]] == list(range(16))
    for record in decoded["ranks"]:
        for operation, row_bytes in (("DISPATCH", 3712), ("COMBINE", 7168)):
            for kind, rows in (("INTRA", 32), ("CROSS", 96)):
                value = record["classes"][f"{operation}_{kind}"]
                assert type(value["rows"]) is int and value["rows"] == rows
                assert type(value["bytes"]) is int and value["bytes"] == rows * row_bytes
                assert value["summed_wave_envelopes_us"] == 20
                assert value["gbps"] == rows * row_bytes / 20 / 1000
    group.close()
