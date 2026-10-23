"""CPU fixtures for interpreting native snapshots without importing torch/NPU."""

import gc
import json
import weakref

import numpy as np
import pytest

from deep_ep_experimental.megamoe import reference
from deep_ep_experimental.megamoe.urma_debug import (
    FRAGMENT_DTYPE,
    SEGMENT_DTYPE,
    save_snapshot,
    source_weight_prefixes,
)


def byte_view(value):
    return np.ascontiguousarray(value).view(np.uint8).reshape(-1)


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    return (value.view(np.uint32) >> 16).astype("<u2")


def fixture_snapshot():
    bs, topk, hidden, intermediate = 3, 2, 32, 32
    order = np.array([4, 0, 2, 5, 1, 3], dtype="<i8")
    inverse = np.argsort(order).astype("<i8").reshape(bs, topk)
    x = np.broadcast_to(np.array([56, 64, 72], dtype=np.uint8)[:, None], (bs, hidden)).copy()
    scales = np.array([[126], [127], [128]], dtype=np.uint8)
    received = x[order // topk]
    received_scale = scales[order // topk]
    gather_order = [0, 3, 1, 2, 4, 5]
    returned = np.broadcast_to(np.arange(2, 14, 2, dtype=np.float32)[:, None], (6, hidden)).copy()
    remote = returned.copy()
    remote[:3] = 0  # These self rows are intentionally unused in native GMM2 output.
    weights = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]], dtype="<f4")
    expected = np.broadcast_to(np.array([8.5, 9, 3.5], dtype=np.float32)[:, None], (bs, hidden)).copy()
    fragments = np.array([(0, 0, 0, 2, 2, 0, 0), (1, 2, 2, 4, 2, 1, 1)], dtype=FRAGMENT_DTYPE)
    segments = np.array(
        [(0, 0, 0, 0, 0, 1, 0, 0), (1, 0, 3, 3, 1, 1, 1, 0), (0, 0, 1, 1, 0, 2, 0, 0), (1, 0, 4, 4, 2, 2, 1, 0)],
        dtype=SEGMENT_DTYPE,
    )
    local_flags = np.zeros((16, 32), dtype="<i4")
    local_flags[:3, 0] = [15, 28, 16]
    raw = {
        "x": x,
        "x_scale": scales,
        "received_x": received,
        "received_scale": received_scale,
        "gathered_input_scale": received_scale[gather_order],
        "gmm1": bf16(np.ones((6, intermediate * 2))),
        "activation": np.full((6, intermediate), 56, dtype=np.uint8),
        "activation_scale": np.full((6, intermediate // 32), 127, dtype=np.uint8),
        "gmm2_remote": bf16(remote),
        "returned": bf16(returned),
        "output": bf16(expected),
        "weights": weights,
        "reduce_index": inverse,
        "order": order,
        "fragments": fragments,
        "segments": segments,
        "local_flags": local_flags,
        "weight1_prefix": np.array([0x22, 0xA2, 0x80], dtype=np.uint8),
        "weight_scale1_prefix": np.array([127, 128, 126], dtype=np.uint8),
    }
    return {key: byte_view(value) for key, value in raw.items()}


def save(raw, path, **kwargs):
    dimensions = dict(bs=3, hidden=32, intermediate=32, topk=2, received_rows=6, rank=0)
    dimensions.update(kwargs)
    return save_snapshot(raw, path, **dimensions)


def test_valid_pack_gather_and_weighted_reduce(tmp_path):
    report = save(fixture_snapshot(), tmp_path)
    assert not report["errors"]
    assert not report["findings"]
    assert report["checks"]["order_permutation"]
    assert report["checks"]["order_inverse_match"]
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    assert report["checks"]["cpu_reduce_vs_output"]["cpu_nonzero_elements"] == 96
    segments = report["checks"]["segments"]
    assert segments["expert_rows_covered_once"] and segments["receive_rows_covered_once"]
    assert segments["self_rows"] == 3
    assert segments["self_pack_x_mismatched_bytes"] == 0
    assert segments["self_pack_scale_mismatched_bytes"] == 0
    assert segments["gathered_scale_mismatched_bytes"] == 0
    assert report["checks"]["expert_major_received_x"]["nonzero_elements"] == 192
    remote = report["checks"]["remote_segment_output"]
    assert remote["elements"] == remote["nonzero_elements"] == 96
    assert report["flags"]["local_named"] == dict(dispatch_done=15, aic_done=28, incoming_done=16)
    assert json.loads((tmp_path / "debug_intermediates.json").read_text()) == report
    with np.load(tmp_path / "debug_samples.npz", allow_pickle=False) as samples:
        np.testing.assert_array_equal(samples["cpu_reduce"][:, 0], [8.5, 9.0, 3.5])
        np.testing.assert_array_equal(samples["weight1_prefix"], [1, 1, 1, -1, 0, -0.0])


def test_nonzero_return_zero_output_points_to_reduce(tmp_path):
    raw = fixture_snapshot()
    raw["output"][:] = 0
    report = save(raw, tmp_path)
    check = report["checks"]["cpu_reduce_vs_output"]
    assert not check["exact_match"]
    assert check["different_elements"] == 96
    assert check["max_abs_error"] == 9.0
    assert check["cpu_nonzero_elements"] == 96 and check["output_nonzero_elements"] == 0
    assert any(item.startswith("RETURN_NONZERO_OUTPUT_ZERO") for item in report["findings"])
    assert not any(item.startswith("RETURN_ALL_ZERO") for item in report["findings"])


def test_zeros_before_reduce_are_not_blamed_on_reduce(tmp_path):
    raw = fixture_snapshot()
    for name in ("gmm1", "activation", "gmm2_remote", "returned", "output"):
        raw[name][:] = 0
    report = save(raw, tmp_path)
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    assert any(
        item.startswith("ZERO_TRANSITION: received_x is nonzero, gmm1 is all zero") for item in report["findings"]
    )
    assert any(item.startswith("RETURN_ALL_ZERO") for item in report["findings"])
    assert not any(item.startswith("RETURN_NONZERO_OUTPUT_ZERO") for item in report["findings"])


def test_corrupt_pack_and_scale_gather_are_detected(tmp_path):
    raw = fixture_snapshot()
    raw["received_x"][0] ^= 1
    raw["gathered_input_scale"][4] ^= 1
    report = save(raw, tmp_path)
    assert report["checks"]["segments"]["self_pack_x_mismatched_bytes"] == 1
    assert report["checks"]["segments"]["gathered_scale_mismatched_bytes"] == 1


def test_invalid_inverse_does_not_index_or_hide_stats(tmp_path):
    raw = fixture_snapshot()
    raw["reduce_index"].view("<i8")[0] = 999
    report = save(raw, tmp_path)
    assert not report["checks"]["reduce_indices_in_range"]
    assert "cpu_reduce_vs_output" not in report["checks"]
    assert report["stages"]["output"]["nonzero_elements"] == 96


def test_bad_optional_buffers_preserve_other_diagnostics(tmp_path):
    raw = fixture_snapshot()
    raw["fragments"] = np.array([1], dtype=np.uint8)
    raw["gmm1_done"] = np.zeros(129, dtype=np.uint8)
    raw["gmm1"] = np.array([1], dtype=np.uint8)
    report = save(raw, tmp_path)
    assert len(report["errors"]) == 3
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]


def test_nonfinite_data_serializes_without_nan_json(tmp_path):
    raw = fixture_snapshot()
    raw["x"][0] = 127  # E4M3 NaN.
    raw["x_scale"][0] = 255  # E8M0 NaN.
    raw["output"].view("<u2")[0] = 0x7F80  # BF16 +inf.
    report = save(raw, tmp_path)
    assert report["stages"]["x"]["nonfinite_elements"] == 1
    assert report["stages"]["x_scale"]["nonfinite_elements"] == 1
    assert report["checks"]["cpu_reduce_vs_output"]["nonfinite_elements"] == 1
    json.loads((tmp_path / "debug_intermediates.json").read_text(), parse_constant=lambda value: pytest.fail(value))


def test_cpu_tensor_protocol_without_torch(tmp_path):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    report = save({key: Tensor(value) for key, value in fixture_snapshot().items()}, tmp_path)
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]


def test_reject_nonbyte_snapshot_and_invalid_dimensions(tmp_path):
    raw = fixture_snapshot()
    raw["gmm1"] = np.zeros((6, 64), dtype=np.float32)
    report = save(raw, tmp_path)
    assert any("snapshot must contain uint8" in error for error in report["errors"])
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    with pytest.raises(ValueError, match="scale widths"):
        save(fixture_snapshot(), tmp_path, hidden=31)


def test_reduce_larger_than_one_cpu_block(tmp_path):
    bs, hidden, topk = 137, 32, 2
    raw = {
        "returned": byte_view(bf16(np.ones((bs * topk, hidden)))),
        "output": byte_view(bf16(np.ones((bs, hidden)))),
        "weights": byte_view(np.full((bs, topk), 0.5, dtype="<f4")),
        "reduce_index": byte_view(np.arange(bs * topk, dtype="<i8").reshape(bs, topk)),
    }
    report = save(raw, tmp_path, bs=bs)
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    assert report["checks"]["cpu_reduce_vs_output"]["cpu_nonzero_elements"] == bs * hidden
    with np.load(tmp_path / "debug_samples.npz") as samples:
        assert samples["cpu_reduce"].shape == (4, hidden)


def test_source_weight_prefixes_preserve_raw_bytes_and_bound_each_copy():
    limit = 1024 * 1024
    raw = (
        (np.arange(limit + 38, dtype=np.uint8).reshape(2, -1), np.array([0, 127, 128, 255], dtype=np.uint8)),
        (np.arange(limit + 64, dtype=np.uint8), np.arange(limit + 32, dtype=np.uint8)),
    )
    prefixes = source_weight_prefixes(raw)
    assert set(prefixes) == {
        "source_weight1_prefix",
        "source_weight_scale1_prefix",
        "source_weight2_prefix",
        "source_weight_scale2_prefix",
    }
    for layer, pair in enumerate(raw, 1):
        for kind, source in zip(("weight", "weight_scale"), pair):
            prefix = prefixes[f"source_{kind}{layer}_prefix"]
            assert prefix.dtype == np.uint8 and prefix.ndim == 1
            assert prefix.size == min(source.size, limit)
            assert prefix.flags.c_contiguous and prefix.flags.owndata and prefix.base is None
            assert not np.shares_memory(prefix, source)
            np.testing.assert_array_equal(prefix, source.reshape(-1)[:limit])
    np.testing.assert_array_equal(prefixes["source_weight_scale1_prefix"], [0, 127, 128, 255])
    expected = {name: value.copy() for name, value in prefixes.items()}
    for pair in raw:
        for source in pair:
            source[:] = 42
    for name in prefixes:
        np.testing.assert_array_equal(prefixes[name], expected[name])


def test_source_prefixes_do_not_keep_original_allocations_alive():
    def create():
        arrays = [np.full(2048, value, dtype=np.uint8) for value in (1, 2, 3, 4)]
        refs = [weakref.ref(array) for array in arrays]
        prefixes = source_weight_prefixes(((arrays[0], arrays[1]), (arrays[2], arrays[3])), limit=7)
        return prefixes, refs

    prefixes, original = create()
    gc.collect()
    assert all(ref() is None for ref in original)
    for value, prefix in enumerate(prefixes.values(), 1):
        np.testing.assert_array_equal(prefix, np.full(7, value, dtype=np.uint8))
        assert prefix.nbytes == 7


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, np.int64(3)])
def test_source_prefixes_reject_invalid_limit(limit):
    raw = ((np.array([1], dtype=np.uint8), np.array([127], dtype=np.uint8)),)
    with pytest.raises(ValueError, match="positive integer"):
        source_weight_prefixes(raw, limit=limit)


@pytest.mark.parametrize("bad", [np.array([1], dtype=np.int8), np.array([1.0]), np.arange(12, dtype=np.uint8)[::2]])
def test_source_prefixes_reject_nonbyte_or_noncontiguous_source(bad):
    with pytest.raises(ValueError, match="contiguous CPU uint8"):
        source_weight_prefixes(((bad, np.array([127], dtype=np.uint8)),))


def test_source_prefixes_do_not_request_device_conversion():
    class DeviceNZ:
        def cpu(self):
            pytest.fail("source prefix capture must never request NPU NZ D2H")

        def __array__(self, *args, **kwargs):
            pytest.fail("source prefix capture must not coerce a device object")

    with pytest.raises(ValueError, match="contiguous CPU uint8"):
        source_weight_prefixes(((DeviceNZ(), np.array([127], dtype=np.uint8)),))


def test_source_prefix_report_distinguishes_cpu_nd_from_device_snapshot(tmp_path):
    raw = fixture_snapshot()
    del raw["weight1_prefix"], raw["weight_scale1_prefix"]
    sources = (
        (np.array([0x22, 0xA2, 0x80], dtype=np.uint8), np.array([127, 128, 126], dtype=np.uint8)),
        (np.array([0x43], dtype=np.uint8), np.array([129], dtype=np.uint8)),
    )
    prefixes = source_weight_prefixes(sources)
    raw.update(prefixes)
    report = save(raw, tmp_path)
    assert not report["errors"]
    for name, prefix in prefixes.items():
        stage = report["stages"][name]
        assert stage["source"] == "original_cpu_nd"
        assert stage["shape"] == [prefix.size]
        assert stage["dtype"] == ("e8m0" if "scale" in name else "packed_fp4_e2m1")
    assert "weight1_prefix" not in report["stages"]
    assert any("original CPU ND" in note and "not snapshots" in note for note in report["notes"])
    with np.load(tmp_path / "debug_samples.npz", allow_pickle=False) as samples:
        np.testing.assert_array_equal(samples["source_weight1_prefix"], [1, 1, 1, -1, 0, -0.0])
        np.testing.assert_array_equal(samples["source_weight_scale1_prefix"], [1, 2, 0.5])


def with_reference_activation(raw, rows=6):
    rng = np.random.default_rng(391)
    first = reference.bf16(rng.normal(0, 0.2, (rows, 64)).astype(np.float32))
    gate, up = np.split(first, 2, axis=-1)
    activated = reference.bf16(gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up)
    codes, scales = reference.quantize_input(activated)
    raw.update(gmm1=byte_view(bf16(first)), activation=byte_view(codes), activation_scale=byte_view(scales))
    return raw, first, codes, scales


def test_conditional_activation_matches_golden_from_real_gmm1(tmp_path):
    raw, _, _, _ = with_reference_activation(fixture_snapshot())
    report = save(raw, tmp_path)
    check = report["checks"]["cpu_activation_from_gmm1"]
    assert check["source"] == "actual_gmm1_conditioned_cpu_golden"
    assert check["diagnostic_only"] and not check["hardware_bitwise_emulation"]
    assert check["exact_byte_match"] and not check["first_mismatches"]
    assert check["code_elements"] == 192 and check["scale_elements"] == 6
    assert check["code_different_elements"] == check["scale_different_elements"] == 0
    assert check["dequantized_difference"]["different_elements"] == 0
    assert check["dequantized_difference"]["max_abs_error"] == 0.0
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    assert not report["errors"]


@pytest.mark.parametrize("corrupt", ["code", "scale"])
def test_conditional_activation_reports_corrupt_code_or_scale(tmp_path, corrupt):
    raw, first, codes, scales = with_reference_activation(fixture_snapshot())
    expected_codes, expected_scales = codes.copy(), scales.copy()
    if corrupt == "code":
        raw["activation"][32 + 5] ^= 1
    else:
        raw["activation_scale"][1] += 1
    report = save(raw, tmp_path)
    check = report["checks"]["cpu_activation_from_gmm1"]
    assert not check["exact_byte_match"]
    assert check["code_different_elements"] == (1 if corrupt == "code" else 0)
    assert check["scale_different_elements"] == (1 if corrupt == "scale" else 0)
    assert check["dequantized_difference"]["max_abs_error"] > 0
    assert len(check["first_mismatches"]) == (1 if corrupt == "code" else 8)
    example = check["first_mismatches"][0]
    channel = 5 if corrupt == "code" else 0
    assert example["row"] == 1 and example["channel"] == channel
    assert example["gate"] == float(first[1, channel])
    assert example["up"] == float(first[1, 32 + channel])
    assert example["expected_code"] == int(expected_codes[1, channel])
    assert example["expected_scale"] == int(expected_scales[1, 0])
    assert example["actual_code"] == int(raw["activation"][32 + channel])
    assert example["actual_scale"] == int(raw["activation_scale"][1])
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    assert "passed" not in check


@pytest.mark.parametrize("corrupt", ["gmm1", "activation", "activation_scale"])
def test_conditional_activation_nonfinite_is_only_a_diagnostic_error(tmp_path, corrupt):
    raw, _, _, _ = with_reference_activation(fixture_snapshot())
    if corrupt == "gmm1":
        raw[corrupt].view("<u2")[0] = 0x7F80
    else:
        raw[corrupt][0] = 127 if corrupt == "activation" else 255
    report = save(raw, tmp_path)
    assert "cpu_activation_from_gmm1" not in report["checks"]
    assert any("cpu_activation_from_gmm1: non-finite" in error for error in report["errors"])
    assert report["checks"]["cpu_reduce_vs_output"]["exact_match"]
    json.loads((tmp_path / "debug_intermediates.json").read_text(), parse_constant=lambda value: pytest.fail(value))


def test_conditional_activation_uses_bounded_32_row_blocks(tmp_path, monkeypatch):
    raw, _, _, _ = with_reference_activation({}, rows=70)
    original = reference.quantize_input
    sizes = []

    def checked(value):
        sizes.append(value.shape[0])
        assert value.shape[0] <= 32
        return original(value)

    monkeypatch.setattr(reference, "quantize_input", checked)
    report = save(raw, tmp_path, received_rows=70)
    assert sizes == [32, 32, 6]
    assert report["checks"]["cpu_activation_from_gmm1"]["exact_byte_match"]
