"""CPU check of the production offline permutation and fused projection semantics."""

import numpy as np
import pytest

from deep_ep_experimental.megamoe.api import _interleave_gate_up_128


class ArrayTensor:
    """The small torch view interface used by the offline permutation, backed by NumPy."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def reshape(self, shape):
        return ArrayTensor(self.data.reshape(shape))

    def transpose(self, a, b):
        return ArrayTensor(self.data.swapaxes(a, b))

    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.data))


def test_weight_and_scale_permutation_preserves_exact_storage_bits():
    rng = np.random.default_rng(241)
    for tail in [(64,), (2, 2)]:  # packed FP4 bytes / paired E8M0 bytes
        shape = (3, 2560, *tail)
        data = rng.integers(0, 256, shape, dtype=np.uint8)
        fused = _interleave_gate_up_128(ArrayTensor(data)).data
        for tile in range(10):
            for half in range(2):
                begin = tile * 256 + half * 128
                old_begin = half * 1280 + tile * 128
                np.testing.assert_array_equal(fused[:, begin:begin + 128], data[:, old_begin:old_begin + 128])


@pytest.mark.parametrize("rows", [1, 85, 128, 171, 255, 256])
def test_one_fused_projection_replaces_two_planar_projections(rows):
    rng = np.random.default_rng(rows)
    x = rng.integers(-2, 3, (rows, 32), dtype=np.int32)
    weights = rng.integers(-2, 3, (1, 2560, 32), dtype=np.int32)
    fused = _interleave_gate_up_128(ArrayTensor(weights)).data[0]
    expected = x @ weights[0].T
    gate, up = np.empty((rows, 1280), np.int32), np.empty((rows, 1280), np.int32)
    for tile in range(10):
        physical = x @ fused[tile * 256:(tile + 1) * 256].T
        gate[:, tile * 128:(tile + 1) * 128] = physical[:, :128]
        up[:, tile * 128:(tile + 1) * 128] = physical[:, 128:]
    np.testing.assert_array_equal(gate, expected[:, :1280])
    np.testing.assert_array_equal(up, expected[:, 1280:])
