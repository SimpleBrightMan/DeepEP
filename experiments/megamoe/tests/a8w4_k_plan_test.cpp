#include <cassert>
#include <initializer_list>
#define __aicore__
#include "../csrc/kernels/mega_moe/catlass_gmm/a8w4_k_plan.hpp"

int main()
{
    using MegaMoeImpl::GmmKernel::MakeA8W4KPlan;
    assert(MakeA8W4KPlan(256, 256, false).ka == 256);
    assert(MakeA8W4KPlan(256, 256, false, true).ka == 512);
    for (unsigned m = 1; m <= 256; ++m) {
        for (unsigned n = 1; n <= 256; ++n) {
            for (bool dynamic : {false, true}) {
                auto before = MakeA8W4KPlan(m, n, dynamic);
                auto after = MakeA8W4KPlan(m, n, dynamic, true);
                // A窗口只增宽，不改变AIV0的B生产节拍；所有尾块仍须满足物理容量。
                assert(before.kb == after.kb);
                assert(after.ka >= after.kb && after.ka % after.kb == 0);
                assert(((m + 15) / 16 * 16) * after.ka <= 128 * 1024);
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
