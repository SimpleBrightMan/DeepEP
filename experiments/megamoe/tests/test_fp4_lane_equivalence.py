"""Exhaustive lane algebra check; device lane ordering still requires NPU accuracy."""


def arithmetic_right_2(byte):
    return (byte >> 2) | (0xC0 if byte & 0x80 else 0)


def test_all_packed_fp4_bytes():
    for packed in range(256):
        old_low = arithmetic_right_2((packed << 4) & 0xFF) & 0x9C
        old_high = arithmetic_right_2(packed) & 0x9C
        new = tuple(arithmetic_right_2((packed << shift) & 0xFF) & 0x9C for shift in (4, 0))
        assert new == (old_low, old_high), (packed, new, old_low, old_high)


if __name__ == "__main__":
    test_all_packed_fp4_bytes()
    print("All 256 packed bytes: both FP8 lane bit patterns match")
