// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "timer.hpp"

namespace MegaMoeUrma {

// One outstanding notification per AIC. The marker is queued immediately
// after its tile's FIX stores. Submit the next independent tile before
// Publish(), so waiting for this marker never drains the newer tile.
// FIX_S 1 is private to this queue; the matmul context teardown uses FIX_S 0.
class DeferredGmmCompletion {
public:
    __aicore__ inline void Publish(Timer& timer)
    {
        if constexpr (g_coreType == AscendC::AIC) {
            if (!pending_) return;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kCompletionEvent);
            if (event_ == TraceEvent::Gmm1Tile) {
                AscendC::WriteGmByPassDCache(flag_, 1);
            } else {
                // One contribution per completed GMM2 N tile. Its marker
                // covers every segmented FIX write, including self output.
                AscendC::AtomicAdd(flag_, 1);
            }
            timer.TraceEnd(event_, start_, meta_, kTraceFlagDeferredCompletion);
            pending_ = false;
        }
    }

    // Call after Compute and after publishing any previous notification.
    __aicore__ inline void Arm(__gm__ int32_t* flag, TraceEvent event, uint64_t start, const TraceMeta& meta)
    {
        if constexpr (g_coreType == AscendC::AIC) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kCompletionEvent);
            flag_ = flag;
            event_ = event;
            start_ = start;
            meta_ = meta;
            pending_ = true;
        }
    }

    __aicore__ inline void WaitInput(__gm__ int32_t* flag, int32_t expected, Timer& timer)
    {
        if constexpr (g_coreType == AscendC::AIC) {
            if (AscendC::ReadGmByPassDCache(flag) < expected) {
                // The activation owner may need our pending GMM1 flag in
                // order to satisfy this wait. Never hold it while blocking.
                Publish(timer);
                while (AscendC::ReadGmByPassDCache(flag) < expected) {
                }
            }
        }
    }

private:
    static constexpr uint16_t kCompletionEvent = 1U;
    __gm__ int32_t* flag_ = nullptr;
    TraceEvent event_ = TraceEvent::Gmm1Tile;
    uint64_t start_ = 0U;
    TraceMeta meta_{};
    bool pending_ = false;
};

}  // namespace MegaMoeUrma
