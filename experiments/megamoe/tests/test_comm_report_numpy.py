# Copyright (c) 2026, SimpleBright_Man

"""Exercise actual NumPy timer/audit dtype through the real TCPStore encoder."""

import json

import numpy as np

from deep_ep._store import StoreGroup
from deep_ep_experimental.megamoe import comm_report
from comm_report_fixtures import capture


class SingleRankStore:
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


def test_numpy_timer_audit_survive_store_gather_and_json_summary(tmp_path):
    group = StoreGroup(SingleRankStore(), 0, 1, prefix="numpy-timer-report")
    records = []
    for rank in range(16):
        spec, measured, audit, _ = capture(rank=rank, dispatch_intra=10, combine_intra=0)
        assert isinstance(measured[0], np.int64)
        assert measured[0:2].dtype == np.int64
        events = comm_report.read_events(measured, rank)
        transfers = comm_report.read_transfers(audit, rank)
        for event in events:
            for key in ("rank", "core", "wave", "start_cycle", "end_cycle", "expert_id", "row_start", "row_count"):
                assert type(event[key]) is int, (key, type(event[key]))
        for transfer in transfers:
            for key in ("rank", "core", "wave", "expert_id", "row_start", "source_rank", "source_token", "topk_slot"):
                assert type(transfer[key]) is int, (key, type(transfer[key]))
        report = comm_report.rank_report(events, transfers=transfers, **spec)
        compact = comm_report.write_rank_detail(report, tmp_path)
        records.extend(group.all_gather(compact, f"communication report rank={rank}"))
    summary = comm_report.summarize(records)
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert summary["schema_version"] == 2
    assert set(summary["operations"]) == {"DISPATCH", "COMBINE"}
    assert [r["rank"] for r in summary["ranks"]] == list(range(16))
    for report in records:
        assert report["operations"]["DISPATCH"]["bytes"] == 128 * 3712
        assert report["operations"]["COMBINE"]["bytes"] == 128 * 7168
        assert report["validation"]["strict_wave_core"]
    group.close()
