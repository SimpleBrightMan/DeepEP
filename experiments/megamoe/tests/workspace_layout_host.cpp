// Execute the real host/device layout with math-only SDK stubs; no device allocation.
#include <cassert>
#include <cstdint>
#include <iostream>
#include "common/mega_moe_workspace.hpp"

using namespace MegaMoeImpl;

int main()
{
    for (uint32_t ep : {2U, 8U, 64U, 128U}) {
        for (uint32_t experts : {1U, 6U, 48U}) {
            for (uint32_t bs : {1U, 255U, 256U, 257U, 4096U, 4104U}) {
                for (bool w4 : {false, true}) {
                    for (uint32_t shared : {0U, 1U, 8U}) {
                        if (!w4 && shared != 0) continue; // Public A8W8 has no shared experts.
                        MegaMoeTilingData t{};
                        t.bs = t.numMaxTokensPerRank = bs;
                        t.epWorldSize = ep;
                        t.moeExpertPerRank = experts;
                        t.aicNum = 32;
                        t.blockAivNum = 64;
                        t.h = 7168;
                        t.hiddenDim = 6144;
                        t.topK = 6;
                        t.sharedExpertNum = shared;
                        t.maxOutputSize = bs * ep * (experts < t.topK ? experts : t.topK);
                        t.groupedMatmulMode = w4 ? GROUPED_MATMUL_MODE_A8W4 : GROUPED_MATMUL_MODE_GENERAL;
                        auto* base = reinterpret_cast<uint8_t*>(uintptr_t{0x100000000});
                        const WorkspaceInfo layout(base, &t);
                        const uint64_t groups = (uint64_t(t.maxOutputSize) + 255) / 256;
                        // Frozen MTE contract, independent of the production allocation sequence.
                        uint64_t slots = experts * (2 * groups + t.aicNum) + t.aicNum;
                        if (w4) slots += experts * groups + t.aicNum;
                        if (shared) slots += ((bs + 255) / 256) * shared + 4 * t.aicNum;
                        assert(uint64_t(layout.flagResetElementCount) == slots * INT_CACHELINE);
                        assert(layout.flagDispatchToGmm1Ptr - layout.flagActivationToGmm2Ptr ==
                               int64_t(experts * groups * INT_CACHELINE * sizeof(int32_t)));
                        assert(layout.flagSendCntCalToUpdParamsPtr - layout.flagDispatchToGmm1Ptr ==
                               int64_t(experts * groups * INT_CACHELINE * sizeof(int32_t)));
                        assert((layout.gmm2ReadyPtr != nullptr) == w4);
                        assert((layout.flagGmm2IssuedPtr != nullptr) == w4);
                        if (w4) {
                            assert(layout.flagGmm2IssuedPtr == layout.gmm2ReadyPtr + experts * groups * INT_CACHELINE * sizeof(int32_t));
                            assert(layout.flagGmm2IssuedPtr + t.aicNum * INT_CACHELINE * sizeof(int32_t) <= layout.flagActivationToGmm2Ptr + layout.flagResetElementCount * sizeof(int32_t));
                        }
                        assert((layout.gmm1MmadResPtr != nullptr) == w4);
                        assert((layout.cumsumInfoPtr != nullptr) == w4);
                        const auto flagsEnd = layout.flagActivationToGmm2Ptr +
                                              layout.flagResetElementCount * sizeof(int32_t);
                        assert(flagsEnd <= layout.gmm2MmadResPtr);
                        assert(layout.gmm2MmadResPtr + uint64_t(t.maxOutputSize) * t.h * 2 <=
                               base + layout.workspaceSize);
                        assert((uintptr_t(layout.gmm2MmadResPtr) - uintptr_t(base)) % 512 == 0);
                        // Record full allocation identity for frozen-baseline comparisons.
                        std::cout << ep << ',' << experts << ',' << bs << ',' << w4 << ',' << shared << ','
                                  << layout.workspaceSize << ',' << layout.flagResetElementCount << ','
                                  << layout.gmm2MmadResPtr - base << '\n';
                    }
                }
            }
        }
    }
}
