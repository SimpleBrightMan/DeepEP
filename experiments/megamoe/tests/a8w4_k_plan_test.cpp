#include <cassert>
#include <initializer_list>
#define __aicore__
#include "../csrc/kernels/mega_moe/catlass_gmm/a8w4_k_plan.hpp"
#include "../csrc/kernels/mega_moe/catlass_gmm/a8w4_l1_layout.hpp"

template<unsigned Stages>
void CheckLayout()
{
    using Map = MegaMoeImpl::GmmKernel::A8W4L1Layout<Stages>;
    bool occupied[512]{};
    auto mark = [&](unsigned begin, unsigned bytes) {
        assert(begin % 1024 == 0 && bytes % 1024 == 0 && begin + bytes <= 512 * 1024);
        for (unsigned i = begin / 1024; i < (begin + bytes) / 1024; ++i) {
            assert(!occupied[i]);
            occupied[i] = true;
        }
    };
    for (unsigned i = 0; i < Stages; ++i) mark(Map::B(i), 64 * 1024);
    for (unsigned i = 0; i < 2; ++i) {
        mark(Map::A(i), Map::A_BYTES);
        mark(Map::SA(i), 32 * 1024);
        mark(Map::SB(i), 32 * 1024);
    }
    assert(Map::FULL_FLAG + Stages <= Map::EMPTY_FLAG);
    assert(Map::EMPTY_FLAG + Stages <= 16);
}

int main()
{
    using MegaMoeImpl::GmmKernel::MakeA8W4KPlan;
    CheckLayout<2>();
    CheckLayout<4>();
    assert(MakeA8W4KPlan(256, 256, false).ka == 256);
    assert(MakeA8W4KPlan(256, 256, false, 64 * 1024).ka == 256);
    assert(MakeA8W4KPlan(192, 128, true, 64 * 1024).kb == 256);
    for (unsigned m = 1; m <= 256; ++m) {
        for (unsigned n = 1; n <= 256; ++n) {
            for (bool dynamic : {false, true}) {
                auto after = MakeA8W4KPlan(m, n, dynamic, 64 * 1024);
                // AIC和AIV0共用同一计划；所有尾块须满足新的物理容量。
                assert(after.ka >= after.kb && after.ka % after.kb == 0);
                assert(((m + 15) / 16 * 16) * after.ka <= 64 * 1024);
                assert(n * after.kb <= 64 * 1024);
                for (unsigned k : {64, 128, 256, 320, 512, 3072, 7168}) {
                    unsigned released = 0;
                    for (unsigned offset = 0; offset < k; offset += after.kb) {
                        unsigned actual = after.kb < k - offset ? after.kb : k - offset;
                        unsigned start = offset / after.ka * after.ka;
                        unsigned window = after.ka < k - start ? after.ka : k - start;
                        assert(offset + actual <= start + window);
                        if (offset + actual == k || (offset + actual) % after.ka == 0)
                            ++released;
                    }
                    assert(released == (k + after.ka - 1) / after.ka);
                }
            }
        }
    }
}
