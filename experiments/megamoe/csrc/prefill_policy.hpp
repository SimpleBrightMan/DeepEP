// Copyright (c) 2026, SimpleBright_Man
#pragma once

#include <cstdint>

namespace deepep::megamoe {

struct PrefillProfileInputs {
    bool requested;
    bool isA8W4;
    bool combineNoQuant;
    bool topkWeightsPrefetch;
    uint32_t world;
    uint32_t aic;
    uint32_t aiv;
    uint32_t bs;
    uint32_t hidden;
    uint32_t sharedExperts;
    uint32_t localExperts;
    uint32_t ubBytes;
    uint32_t dispatchSlots;
    uint32_t dispatchRouteItems;
    uint32_t dispatchCopyBytes;
};

inline bool UsePrefillProfile(const PrefillProfileInputs& p)
{
    if (!p.requested || !p.isA8W4 || !p.combineNoQuant || p.topkWeightsPrefetch || p.world != 16U || p.aic != 28U ||
        p.aiv != 56U || p.bs <= 256U || p.sharedExperts != 0U || p.hidden == 0U || p.localExperts == 0U ||
        p.dispatchSlots != 6U || p.dispatchRouteItems == 0U || p.dispatchCopyBytes == 0U) {
        return false;
    }
    const auto align32 = [](uint64_t bytes) { return (bytes + 31U) / 32U * 32U; };
    const uint64_t dispatchBytes = align32(static_cast<uint64_t>(p.world) * p.localExperts * 4U) +
                                   align32(static_cast<uint64_t>(p.localExperts) * 4U) +
                                   align32(static_cast<uint64_t>(p.dispatchRouteItems) * 4U) +
                                   6U * (static_cast<uint64_t>(p.dispatchCopyBytes) + 32U);
    // The row ring shares the existing Combine UB allocation [64 KiB,184 KiB).
    const uint64_t combineBytes = align32(static_cast<uint64_t>(p.hidden) * 2U) * 8U;
    // The existing metadata region extends to 248 KiB, beyond the row ring.
    return dispatchBytes <= p.ubBytes && p.ubBytes >= 248U * 1024U && combineBytes <= 120U * 1024U;
}

inline void FillPrefillMetadata(int32_t* values, uint32_t world, uint32_t bs, uint32_t topk)
{
    for (uint32_t rank = 0U; rank < world; ++rank) {
        for (uint32_t token = 0U; token < bs; ++token) {
            for (uint32_t slot = 0U; slot < topk; ++slot) {
                const uint64_t row = (static_cast<uint64_t>(rank) * bs + token) * topk + slot;
                int32_t* record = values + row * 8U;
                record[0] = static_cast<int32_t>(rank);
                record[1] = static_cast<int32_t>(token);
                record[2] = static_cast<int32_t>(slot);
                for (uint32_t field = 3U; field < 8U; ++field) record[field] = 0;
            }
        }
    }
}

}  // namespace deepep::megamoe
