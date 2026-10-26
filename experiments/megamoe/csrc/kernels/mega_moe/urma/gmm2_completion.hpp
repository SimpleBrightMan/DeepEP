// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"

namespace MegaMoeUrma {

#if defined(__NPU_DEVICE__)
#define MEGAMOE_COMPLETION_INLINE __aicore__ inline
#else
#define MEGAMOE_COMPLETION_INLINE inline
#endif

// One expert's contributions owned by one AIC. The last group may be a short
// M tail. All groups retain the same N width and global ownership rotation.
struct Gmm2CompletionRecord {
    uint32_t batchBegin = 0U;
    uint32_t groupCount = 0U;
    uint32_t tilesPerGroup = 0U;
    uint64_t globalTileBegin = 0U;
    uint32_t lastFragmentIndex = 0U;
    Fragment lastFragment{};
};

// Backend::Fence captures the FIX completion boundary NOW. Retire later waits
// for exactly that boundary and publishes the saved expert's contributions.
// Submit the successor before Flush, but flush before waiting for unavailable
// inputs, changing incompatible hardware contexts, or returning from compute.
template <class Backend>
class DeferredGmm2Completion {
public:
    MEGAMOE_COMPLETION_INLINE explicit DeferredGmm2Completion(Backend& backend) : backend_(backend) {}

    MEGAMOE_COMPLETION_INLINE bool Pending() const
    {
        return pending_;
    }

    MEGAMOE_COMPLETION_INLINE void Arm(const Gmm2CompletionRecord& record)
    {
        // Never reuse an outstanding event or overwrite its expert identity.
        Flush();
        record_ = record;
        backend_.Fence();
        pending_ = true;
    }

    MEGAMOE_COMPLETION_INLINE void Flush()
    {
        if (!pending_) return;
        backend_.Retire(record_);
        pending_ = false;
    }

private:
    Backend& backend_;
    Gmm2CompletionRecord record_{};
    bool pending_ = false;
};

#undef MEGAMOE_COMPLETION_INLINE
}  // namespace MegaMoeUrma
