import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_timer import TIMER_NUMEL, communication_bandwidth, summarize_bandwidth


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
    assert intra["includes_self_copy"]
    assert report["includes_self_copy_in_intra"]
    assert rows["Dispatch", "CROSS"]["bandwidth_gbps"] == 20
    assert not rows["Dispatch", "CROSS"]["includes_self_copy"]
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


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
def test_direct_self_pack_and_return_do_not_count_as_transport(bs):
    reports = []
    hidden, topk = 3584, 8
    peer_rows = bs * topk // 16
    for rank in range(16):
        raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
        for block in range(28):
            for subblock in range(2):
                worker = 2 * block + subblock
                pack_rows = len(range(worker, bs * topk, 56))
                reduce_rows = len(range(worker, bs, 56))
                raw[block, subblock + 1, 0, :7] = [
                    1,
                    101,
                    100,
                    pack_rows * (hidden + hidden // 32),
                    1,
                    0,
                    1,
                ]
                raw[block, subblock + 1, 7, :7] = [900, 1100, 200, reduce_rows * hidden * 2, 1, 0, 1]
        for owner in range(15):
            raw[owner, 2, 1, :7] = [200, 400, 200, peer_rows * (hidden + hidden // 32), 4, 0, 1]
            raw[owner, 2, 5, :7] = [500, 800, 300, peer_rows * hidden * 2, 4, 0, 1]
        report = communication_bandwidth(raw, rank)
        assert not report["includes_self_copy_in_intra"]
        for record in report["classes"]:
            peers = 3 if record["communication_class"] == "INTRA" else 12
            row_bytes = hidden + hidden // 32 if record["phase"] == "Dispatch" else hidden * 2
            assert record["payload_bytes"] == peers * peer_rows * row_bytes
            assert len(record["peers"]) == peers and rank not in record["peers"]
            assert not record["includes_self_copy"]
        assert raw[:, :, 0, 3].sum() == bs * topk * (hidden + hidden // 32)
        assert raw[:, :, 7, 3].sum() == bs * hidden * 2
        reports.append(report)
    summary = summarize_bandwidth(reports)
    assert all(not row["includes_self_copy"] and row["nonempty_ranks"] == 16 for row in summary)


def test_summary_retains_self_copy_label_for_older_timer_payloads():
    reports = [communication_bandwidth(timer_with_records(), rank) for rank in range(16)]
    summary = {(row["phase"], row["communication_class"]): row for row in summarize_bandwidth(reports)}
    assert summary["Dispatch", "INTRA"]["includes_self_copy"]
    assert not summary["Dispatch", "CROSS"]["includes_self_copy"]
    assert not summary["Combine", "INTRA"]["includes_self_copy"]


@pytest.mark.parametrize("rank", range(16))
def test_early_combine_helpers_count_only_their_disjoint_payload(rank):
    raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
    cross_peers = [peer for peer in range(16) if peer // 4 != rank // 4]
    expected = {
        ("Dispatch", "INTRA"): 0,
        ("Dispatch", "CROSS"): 0,
        ("Combine", "INTRA"): 0,
        ("Combine", "CROSS"): 0,
    }
    for peer in range(16):
        if peer == rank:
            continue
        owner = peer if peer < rank else peer - 1
        # Empty peers, uneven tails, and odd CV counts must not change accounting.
        rows = (peer % 5) * 33
        kind = "INTRA" if peer // 4 == rank // 4 else "CROSS"
        dispatch_bytes, combine_bytes = rows * 3696, rows * 7168
        raw[owner, 2, 1, :7] = [10, 50, 40, dispatch_bytes, 1, peer + 1, 1]
        helper_rows = 0
        if kind == "CROSS":
            chunks = (rows + 31) // 32
            helper_rows = min(rows, ((chunks + 1) // 2) * 32)
            helper_owner = 15 + cross_peers.index(peer)
            raw[helper_owner, 2, 5, :7] = [100, 200, 100, helper_rows * 7168, 1, peer + 1, 1]
        raw[owner, 2, 5, :7] = [300, 500, 200, (rows - helper_rows) * 7168, 1, peer + 1, 1]
        expected["Dispatch", kind] += dispatch_bytes
        expected["Combine", kind] += combine_bytes
    report = communication_bandwidth(raw, rank)
    classes = {(row["phase"], row["communication_class"]): row for row in report["classes"]}
    assert {key: row["payload_bytes"] for key, row in classes.items()} == expected
    assert not report["includes_self_copy_in_intra"]
    cross = classes["Combine", "CROSS"]
    assert cross["first_tick"] == 100 and cross["last_tick"] == 500
    assert cross["envelope_ticks"] == 400
    assert cross["bandwidth_gbps"] == expected["Combine", "CROSS"] / 400
    assert any(owner >= 15 for owner in cross["workers"])
    assert all(peer // 4 != rank // 4 for peer in cross["peers"])
    assert all(owner < 15 for owner in classes["Combine", "INTRA"]["workers"])


@pytest.mark.parametrize(
    "owner,phase,marker,message",
    [
        (16, 1, 6, "non-transport"),
        (27, 5, 5, "non-transport"),
        (16, 5, 0, "explicit peer marker"),
        (15, 5, 1, "static assignment"),
        (0, 5, 3, "static assignment"),
        (3, 5, 17, "peer marker"),
    ],
)
def test_reject_unmapped_or_inconsistent_transport_workers(owner, phase, marker, message):
    raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
    raw[owner, 2, phase, :7] = [10, 50, 40, 64, 1, marker, 1]
    with pytest.raises(ValueError, match=message):
        communication_bandwidth(raw, 0)
