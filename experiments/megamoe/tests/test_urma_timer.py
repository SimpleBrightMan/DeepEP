import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_timer import TIMER_NUMEL, communication_bandwidth


def timer_with_records():
    value = np.zeros((28, 3, 8, 16), dtype=np.int64)
    # rank0 owner0 -> rank1 INTRA; owner3 -> rank4 CROSS; owner15 -> self.
    value[0, 2, 1, :7] = [100, 200, 100, 1000, 1, 0, 1]
    value[15, 2, 1, :7] = [150, 300, 150, 2000, 1, 0, 1]
    value[3, 2, 1, :7] = [120, 520, 200, 8000, 2, 0, 1]
    value[3, 2, 5, :7] = [600, 1400, 500, 16000, 3, 0, 1]
    return value.reshape(-1)


def test_bandwidth_uses_envelope_and_separates_stages_and_classes():
    report = communication_bandwidth(timer_with_records(), 0, clock_mhz=1000)
    rows = {(r["phase"], r["communication_class"]): r for r in report["classes"]}
    intra = rows["Dispatch", "INTRA"]
    assert intra["payload_bytes"] == 3000
    assert intra["envelope_ticks"] == 200
    assert intra["bandwidth_gbps"] == 15
    assert intra["peers"] == [1, 0]
    assert rows["Dispatch", "CROSS"]["bandwidth_gbps"] == 20
    assert rows["Combine", "CROSS"]["bandwidth_gbps"] == 20
    assert rows["Combine", "INTRA"]["bandwidth_gbps"] is None


def test_raw_ticks_work_without_a_frequency_assumption():
    report = communication_bandwidth(timer_with_records(), 0, clock_mhz=None)
    assert all(row["bandwidth_gbps"] is None for row in report["classes"])
    assert report["classes"][0]["envelope_ticks"] == 200


def test_reject_incomplete_timer_schema():
    raw = timer_with_records().reshape(28, 3, 8, 16)
    raw[3, 2, 1, 6] = 0
    with pytest.raises(ValueError, match="schema"):
        communication_bandwidth(raw, 0)


@pytest.mark.parametrize("raw", [np.zeros(12, dtype=np.int64), np.zeros(TIMER_NUMEL)])
def test_reject_incompatible_timer_buffer(raw):
    with pytest.raises(ValueError):
        communication_bandwidth(raw, 0)
