"""Work accounting and profiler evidence for the isolated CATLASS replay."""

import csv
import math
import sys
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.gmm_only_benchmark import read_samples, route_counts, summarize, workload
from deep_ep_experimental.megamoe.gmm_only_benchmark import load_gmm_extension


def test_standalone_loader_does_not_require_core_native(monkeypatch):
    from deep_ep_experimental.megamoe import api

    def unavailable(*args):
        raise NotImplementedError("MegaMoE is not implemented in DeepEP")

    monkeypatch.setattr(api, "require_native", unavailable)
    extension = SimpleNamespace(gmm_only_protocol=2, GmmOnlyPlan=object)
    monkeypatch.setitem(sys.modules, "deep_ep_experimental.megamoe._C", extension)
    assert load_gmm_extension() is extension


def test_standalone_loader_reports_missing_or_old_extension(monkeypatch):
    monkeypatch.setitem(sys.modules, "deep_ep_experimental.megamoe._C", None)
    with pytest.raises(ImportError, match="Native loader error") as error:
        load_gmm_extension()
    assert sys.executable in str(error.value)
    monkeypatch.setitem(sys.modules, "deep_ep_experimental.megamoe._C", SimpleNamespace())
    with pytest.raises(RuntimeError, match="Python/native versions differ"):
        load_gmm_extension()


@pytest.mark.parametrize("bs,groups", [(2048, 72), (4096, 144), (8192, 264)])
def test_original_balanced_workload_all_ranks(bs, groups):
    counts = route_counts(bs)
    assert counts.shape == (16, 384)
    assert (counts.sum(axis=1) == bs * 8).all()
    descriptions = [workload(counts, rank, bs) for rank in range(16)]
    assert sum(row["receive_rows"] for row in descriptions) == 16 * bs * 8
    for row in descriptions:
        assert row["bs"] == bs
        assert row["receive_rows"] == bs * 8
        assert sorted(set(row["expert_rows"])) == [bs * 8 // 24, bs * 8 // 24 + 1]
        assert row["m_groups"] == groups
        assert row["gmm1_tiles"] == groups * 5  # Each logical GMM1 tile computes gate AND up.
        assert row["gmm2_tiles"] == groups * 14
        assert row["gmm1_flops"] + row["gmm2_flops"] == 6 * bs * 8 * 3584 * 1280
    with pytest.raises(ValueError, match="routing counts"):
        workload(counts, 0, 4096 if bs == 2048 else 2048)


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
