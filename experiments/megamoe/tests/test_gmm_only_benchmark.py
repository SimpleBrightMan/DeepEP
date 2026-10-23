"""Work accounting and profiler evidence for the isolated CATLASS replay."""

import csv
import math

import pytest

from deep_ep_experimental.megamoe.gmm_only_benchmark import read_samples, route_counts, summarize, workload


def test_original_balanced_workload_all_ranks():
    counts = route_counts()
    assert counts.shape == (16, 384)
    assert (counts.sum(axis=1) == 2048 * 8).all()
    descriptions = [workload(counts, rank) for rank in range(16)]
    assert sum(row["receive_rows"] for row in descriptions) == 16 * 2048 * 8
    for row in descriptions:
        assert row["receive_rows"] == 16384
        assert sorted(set(row["expert_rows"])) == [682, 683]
        assert row["m_groups"] == 72
        assert row["gmm1_tiles"] == 360  # Each logical GMM1 tile computes gate AND up.
        assert row["gmm2_tiles"] == 1008
        assert row["gmm1_flops"] + row["gmm2_flops"] == 6 * 16384 * 3584 * 1280


def test_mfu_uses_effective_rows_and_one_device_elapsed_time():
    value = summarize([100, 200], 150 * 1e6 * 765, 765)
    assert value["mean_us"] == 150
    assert value["effective_tflops"] == 765
    assert value["mfu_percent"] == 100
    for durations in ([], [0], [float("nan")], [-1], [float("inf")]):
        with pytest.raises(ValueError):
            summarize(durations, 100, 765)
    with pytest.raises(ValueError):
        summarize([1], 100, math.inf)


def write_profile(tmp_path, rows):
    path = tmp_path / "kernel_details.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Start Time(us)", "Duration(us)", "Stream ID", "Device_id"])
        writer.writerows(rows)


def test_profiler_exact_kernel_selection_and_decimal_order(tmp_path):
    write_profile(
        tmp_path,
        [
            ["mega_moe_gmm_only_kernel_do", "0", "999", "1", "0"],
            ["mega_moe_urma_kernel", "1", "999", "1", "0"],
            ["_Z24mega_moe_gmm_only_kernelN11MegaMoeUrma9ArgumentsEj", "123456789012345.002", "12.5", "1", "0"],
            ["mega_moe_gmm_only_kernel(MegaMoeUrma::Arguments, unsigned int)", "123456789012345.001", "10.5", "1", "0"],
        ],
    )
    assert [s["duration_us"] for s in read_samples(tmp_path, 2)] == [10.5, 12.5]
    with pytest.raises(ValueError, match="Expected"):
        read_samples(tmp_path, 3)


@pytest.mark.parametrize(
    "row",
    [
        ["mega_moe_gmm_only_kernel", "1", "0", "1", "0"],
        ["mega_moe_gmm_only_kernel", "nan", "1", "1", "0"],
    ],
)
def test_invalid_profile_samples_rejected(tmp_path, row):
    write_profile(tmp_path, [row])
    with pytest.raises(ValueError):
        read_samples(tmp_path, 1)


def test_duplicate_or_multiple_device_profile_rejected(tmp_path):
    row = ["mega_moe_gmm_only_kernel", "1", "10", "1", "0"]
    write_profile(tmp_path, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        read_samples(tmp_path, 2)
    write_profile(tmp_path, [row, [*row[:4], "1"]])
    with pytest.raises(ValueError, match="multi-device"):
        read_samples(tmp_path, 2)
