"""Offline shared W8 fixtures must retain the independent W4 Golden values."""
import numpy as np
from deep_ep_experimental.megamoe.reference import (
    expand_fp4_to_fp8, unpack_weight, dequantize_input, make_weights, expert_forward,
)


def test_all_packed_codes_and_nonuniform_scales_preserved():
    packed = np.arange(256, dtype=np.uint8).reshape(1, 8, 32)
    scales = np.arange(115, 131, dtype=np.uint8).reshape(1, 8, 1, 2)
    codes, expanded_scales = expand_fp4_to_fp8((packed, scales))
    assert codes.shape == (1, 8, 64)
    assert codes.flags.c_contiguous
    assert not np.shares_memory(scales, expanded_scales)
    np.testing.assert_array_equal(expanded_scales, scales)
    actual = dequantize_input(codes, expanded_scales.reshape(1, 8, 2))
    np.testing.assert_array_equal(actual, unpack_weight((packed, scales)))


def test_nonzero_shared_ffn_matches_independent_fp4_reference():
    raw = make_weights(1, 128, 256, 20261109)
    w4 = [unpack_weight(pair)[0] for pair in raw]
    w8 = []
    for pair in raw:
        codes, scales = expand_fp4_to_fp8(pair)
        w8.append(dequantize_input(codes, scales.reshape(*codes.shape[:-1], codes.shape[-1] // 32))[0])
    x = np.random.default_rng(93).normal(size=(7, 128)).astype(np.float32)
    expected = expert_forward(x, *w4)
    actual = expert_forward(x, *w8)
    assert np.count_nonzero(expected) > expected.size // 2
    np.testing.assert_array_equal(actual, expected)
