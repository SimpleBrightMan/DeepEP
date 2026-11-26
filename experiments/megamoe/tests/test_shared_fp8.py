"""Production FP8 byte-address/FIX-scale checks and independent FP8 reference values."""
from pathlib import Path
import numpy as np
import pytest
from test_device_schedule import compile_and_run
from deep_ep_experimental.megamoe.reference import FP4, fp8_decode, make_weights, unpack_weight


def test_production_fp8_nz_dma_plan(tmp_path):
    compile_and_run(Path(__file__).with_name("shared_fp8_copy_host.cpp"), tmp_path)


def test_fp8_shared_values_are_not_restricted_to_fp4():
    raw = make_weights(2, 128, 256, 41, weight_bits=8)
    for codes, scales in raw:
        values = fp8_decode(codes)
        assert np.any(~np.isin(values, FP4))
        expected = values * np.repeat(np.exp2(scales.astype(np.int16).reshape(*codes.shape[:-1], -1) - 127), 32, axis=-1)
        np.testing.assert_array_equal(unpack_weight((codes, scales), weight_bits=8), expected)
    assert raw[0][0].shape == (2, 512, 128)
    assert raw[1][0].shape == (2, 128, 256)


def test_fp8_shared_known_values_and_scale():
    codes = np.tile(np.array([0x00, 0x80, 0x01, 0x38, 0x39, 0x7e, 0xb9, 0xfe], dtype=np.uint8), 8).reshape(1, 64)
    scales = np.array([[[128, 125]]], dtype=np.uint8)
    expected = np.tile([0, -0.0, 2**-9, 1, 1.125, 448, -1.125, -448], 8).reshape(1, 64)
    expected[:, :32] *= 2
    expected[:, 32:] *= 0.25
    np.testing.assert_array_equal(unpack_weight((codes, scales), weight_bits=8), expected)
    with pytest.raises(ValueError, match="weight_bits"):
        make_weights(1, 64, 64, 1, weight_bits=16)
