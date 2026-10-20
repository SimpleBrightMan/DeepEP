"""Check the address/bit permutation used by the production DS reverse-layout variants."""
import unittest

class ScaleMapping(unittest.TestCase):
    def test_weight_block_transpose(self):
        for pairs in (48,112):
            n=256
            pair_major=[(p<<16)|c for p in range(pairs) for c in range(n)]
            n_major=[pair_major[p*n+c] for c in range(n) for p in range(pairs)]
            self.assertEqual(len(set(n_major)),pairs*n)
            for p in range(pairs):
                for c in (0,1,127,128,255):
                    self.assertEqual(pair_major[p*n+c],n_major[c*pairs+p])
    def test_activation_tiles_cover_once(self):
        pairs=48
        for valid_m in (1,31,64,127,128,129,256):
            seen=set()
            for stage in range(0,valid_m,128):
                for n in range(0,pairs,2):
                    for r in range(stage,min(stage+128,valid_m)):
                        for p in range(n,n+2):
                            old=n*256+r+(p-n)*256
                            new=r*pairs+p
                            self.assertNotIn(new,seen)
                            self.assertEqual((old//512*2+(old%512)//256,old%256),(p,r))
                            seen.add(new)
            self.assertEqual(seen,set(range(valid_m*pairs)))

if __name__=='__main__':unittest.main()
