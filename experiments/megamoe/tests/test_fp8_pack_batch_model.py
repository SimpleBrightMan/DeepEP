"""Address/capacity model of FP8 batched DMA; device tests still required."""
import numpy as np
import pytest


def align(value, size):
    return (value + size - 1) // size * size


@pytest.mark.parametrize('hidden', [32, 64, 96, 128, 256, 1024, 7168, 8192])
@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 17, 65])
def test_pack_range_and_pingpong_preserve_bytes_and_guard_regions(hidden, count):
    scale_bytes = (hidden + 31) // 32
    scale_offset = align(hidden, 256)
    row_bytes = scale_offset + align(scale_bytes, 32)
    span = 2048 + 2 * row_bytes + 2 * align(hidden, 128) * 2
    rows = min(span // (2 * row_bytes), 65535)
    assert rows >= 1
    slot_bytes = rows * row_bytes
    assert 2 * slot_bytes <= span
    input_gap = (row_bytes - align(hidden, 32)) // 32
    scale_gap = (row_bytes - align(scale_bytes, 32)) // 32
    assert align(hidden + input_gap * 32, 32) == row_bytes
    assert align(scale_bytes + scale_gap * 32, 32) == row_bytes
    start = 7
    rng = np.random.default_rng(31)
    data = rng.integers(0, 256, (start + count + 3, hidden), dtype=np.uint8)
    scales = rng.integers(0, 256, (start + count + 3, scale_bytes), dtype=np.uint8)
    # Preserve the reset/route guards around the aliasable, zero-initialized span.
    storage = np.full(span + 64, 171, dtype=np.uint8)
    storage[32:32 + span] = 0
    result = np.full((start + count + 3, row_bytes), 211, dtype=np.uint8)
    for sequence, index in enumerate(range(0, count, rows)):
        copied = min(rows, count - index)
        slot = 32 + sequence % 2 * slot_bytes
        for r in range(copied):
            dst = slot + r * row_bytes
            storage[dst:dst + hidden] = data[start + index + r]
            storage[dst + scale_offset:dst + scale_offset + scale_bytes] = scales[start + index + r]
        result[start + index:start + index + copied] = storage[slot:slot + copied * row_bytes].reshape(copied, row_bytes)
    expected = np.zeros((count, row_bytes), dtype=np.uint8)
    expected[:, :hidden] = data[start:start + count]
    expected[:, scale_offset:scale_offset + scale_bytes] = scales[start:start + count]
    np.testing.assert_array_equal(result[start:start + count], expected)
    assert np.all(result[:start] == 211) and np.all(result[start + count:] == 211)
    assert np.all(storage[:32] == 171) and np.all(storage[32 + span:] == 171)
