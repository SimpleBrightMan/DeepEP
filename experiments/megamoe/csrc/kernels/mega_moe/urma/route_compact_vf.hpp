// Stable CSR compaction, derived from the Reg idioms in
// lueelu/DeepEP main 89e19399260c16c483b3614f57d2e3c90eb9fc2e:
// experiments/megamoe/csrc/kernels/mega_moe/stage/mega_moe_route_compact.hpp
// RoutePrepare design and validation: experiments/megamoe/URMA_DEVICE_ROUTING.md.
#pragma once

#include "kernel_operator.h"

namespace MegaMoeUrma {

// Stable local CSR in one VF. Inputs and outputs are disjoint UB regions.
//   ids:    32B-aligned int32 expert IDs in [0,384), padded to full vector.
//   csr:    32B-aligned int32 storage, at least round_up(itemCount, 8) items.
//   counts: 32B-aligned int32[384].
// Each output CSR element is a FLAT ROUTE ID, not a token ID. Duplicate expert
// choices remain separate entries; no unique-topk or balance assumption.
// Count/prefix consumers must wait V_S (or V_MTE3 for a direct DMA publication)
// after this VF. Do not reuse its UB until the dependent DMA has completed.
__simd_vf__ inline void RouteCompactCsrVF(__ubuf__ int32_t* ids,
                                        __ubuf__ int32_t* csr,
                                        __ubuf__ int32_t* counts,
                                        uint32_t itemCount,
                                        uint32_t globalBegin)
{
    constexpr uint32_t experts = 384U;
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(int32_t);
    static_assert(experts % lanes == 0U, "dense count stores require complete register blocks");

    auto full = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<int32_t> idsReg, flatReg, compactReg;
    AscendC::Reg::RegTensor<int32_t> laneCounts, countReg, oneReg;
    AscendC::Reg::RegTensor<int32_t> countPacked, countBroadcast, laneIndex;
    AscendC::Reg::RegTensor<uint32_t> zeroIndex;
    AscendC::Reg::Duplicate(oneReg, 1, full);
    AscendC::Reg::Duplicate(zeroIndex, 0U, full);
    AscendC::Reg::Duplicate(countPacked, 0, full);
    AscendC::Reg::Arange(laneIndex, 0);

    // Preserve BOTH the append cursor and the unaligned tail across experts.
    // Resetting AR or finalizing/reinitializing the tail at each expert would
    // turn this back into independent expert slots or corrupt a short bucket.
    __ubuf__ int32_t* routeDst = csr;
    AscendC::Reg::UnalignRegForStore routeStore;
    AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    for (uint32_t expert = 0U; expert < experts; ++expert) {
        AscendC::Reg::Duplicate(laneCounts, 0, full);
        for (uint32_t offset = 0U; offset < itemCount; offset += lanes) {
            uint32_t remaining = itemCount - offset;
            auto active = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::LoadAlign(idsReg, ids + offset);
            AscendC::Reg::Arange(flatReg, static_cast<int32_t>(globalBegin + offset));
            AscendC::Reg::MaskReg match;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(
                match, idsReg, static_cast<int32_t>(expert), active);

            // STORE_REG squeeze and its AR-consuming StoreUnAlign must be
            // paired. No unrelated AR access is allowed between the pair.
            AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(
                compactReg, flatReg, match);
            AscendC::Reg::StoreUnAlign(routeDst, compactReg, routeStore);
            AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(
                laneCounts, laneCounts, oneReg, match);
        }

        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(countReg, laneCounts, full);
        // Build one dense count vector per 'lanes' experts. Use aligned stores
        // rather than a second StoreUnAlign stream while route AR is live.
        AscendC::Reg::Gather(countBroadcast, countReg, zeroIndex);
        AscendC::Reg::MaskReg countLane;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(
            countLane, laneIndex, static_cast<int32_t>(expert % lanes), full);
        AscendC::Reg::Select(countPacked, countBroadcast, countPacked, countLane);
        if ((expert + 1U) % lanes == 0U) {
            AscendC::Reg::StoreAlign(counts + expert + 1U - lanes, countPacked, full);
        }
    }
    if (itemCount != 0U) {
        AscendC::Reg::StoreUnAlignPost(routeDst, routeStore);
    }
}

} // namespace MegaMoeUrma
