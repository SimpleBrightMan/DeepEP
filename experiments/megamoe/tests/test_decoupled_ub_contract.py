"""Address/credit contract model, not a substitute for NPU precision testing."""
import unittest


class DecoupledUbContract(unittest.TestCase):
    def test_handoff_preserves_independent_ring_positions(self):
        # 覆盖零任务、两套环回绕及多次 GMM1/GMM2 交接；不能把输出游标冒充输入游标。
        for first in range(13):
            for second in range(13):
                for third in range(13):
                    owner = (0, 0, 0)
                    observed = []
                    for count in (first, second, third):
                        next_owner = tuple(owner)
                        for _ in range(count):
                            observed.append(next_owner)
                            b, src, dst = next_owner
                            next_owner = ((b + 1) % 2, (src + 1) % 3, (dst + 1) % 2)
                        owner = next_owner
                    self.assertEqual(observed, [(i % 2, i % 3, i % 2)
                                                for i in range(first + second + third)])

    def test_address_sets_are_disjoint_and_within_arch_limit(self):
        inputs = [set(range(i * 32768, (i + 1) * 32768)) for i in range(3)]
        outputs = [set(98304 + slot * 256 + vector * 512 + byte
                       for vector in range(256) for byte in range(256)) for slot in range(2)]
        all_slots = inputs + outputs
        for index, slot in enumerate(all_slots):
            self.assertEqual(len(slot), 32768 if index < 3 else 65536)
            for other in all_slots[index + 1:]:
                self.assertFalse(slot & other)
        combined = set().union(*all_slots)
        self.assertEqual(combined, set(range(224 * 1024)))
        self.assertLess(max(combined), 248 * 1024)

    def test_zero_short_and_wrapped_streams_balance_both_credit_rings(self):
        # 不同 HardEvent 管线有独立 event-id 空间；不用同一数组表示两种 ownership。
        for tiles in range(257):
            input_free, output_free = [1] * 3, [1] * 2
            for tile in range(tiles):
                src, dst = tile % 3, tile % 2
                self.assertEqual(input_free[src], 1)
                self.assertEqual(output_free[dst], 1)
                input_free[src] -= 1
                output_free[dst] -= 1
                # V 完成后归还 packed 输入；MTE3 完成后归还 FP8 输出。
                input_free[src] += 1
                output_free[dst] += 1
            self.assertEqual(input_free, [1] * 3)
            self.assertEqual(output_free, [1] * 2)
            for index in range(3):
                input_free[index] -= 1
            for index in range(2):
                output_free[index] -= 1
            self.assertFalse(any(input_free + output_free))


if __name__ == "__main__":
    unittest.main()
