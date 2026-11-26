#define __aicore__
#include <algorithm>
#include <cassert>
#include <vector>
#include "catlass_gmm/weight_copy_plan.hpp"
using namespace MegaMoeImpl::GmmKernel;
int main()
{
    assert(WeightFixScale(false) == 1 && WeightFixScale(true) == 64);
    for (uint32_t fullK : {64U, 128U, 384U, 512U, 768U, 3072U, 7168U}) {
        for (uint32_t fullN : {256U, 512U, 1024U}) {
            std::vector<uint32_t> physical;
            for (uint32_t kb = 0; kb < fullK; kb += 32)
                for (uint32_t row = 0; row < fullN; ++row)
                    for (uint32_t k = kb; k < kb + 32; ++k) physical.push_back(row * fullK + k);
            for (uint32_t tileN : {128U, 192U, 224U, 256U}) {
                const uint32_t tileK = tileN == 128 ? 512 : 256;
                for (uint32_t ns = 0; ns < fullN; ns += tileN)
                    for (uint32_t ks = 0; ks < fullK; ks += tileK) {
                        const auto n = std::min(tileN, fullN - ns), k = std::min(tileK, fullK - ks);
                        const auto plan = MakeFp8NzCopyPlan(fullN, ns, ks, k, n);
                        assert(plan.blocks * plan.blockBytes == k * n && k * n <= 64 * 1024);
                        uint64_t offset = plan.sourceOffset;
                        for (uint32_t block = 0; block < plan.blocks; ++block) {
                            for (uint32_t row = ns; row < ns + n; ++row)
                                for (uint32_t kk = ks + block * 32; kk < ks + (block + 1) * 32; ++kk)
                                    assert(physical.at(offset++) == row * fullK + kk);
                            offset += plan.sourceGapBytes;
                        }
                    }
            }
        }
    }
}
