// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_REORDER_PROGRESS_INLINE __aicore__ inline
#else
#define URMA_REORDER_PROGRESS_INLINE inline
#endif

// Consume exactly this physical AIC's GMM2 task subsequence. GMM1 lookahead,
// empty experts and batch boundaries do not change ownership. The AIC's
// DeferredOutput publishes these flags in submission order, so a not-ready
// head implies every later owned tile is still unpublished. One flag probe
// suffices; unrelated AICs need not finish their earlier tiles first.
template <class Backend>
class OwnedReorderProgress {
public:
    URMA_REORDER_PROGRESS_INLINE OwnedReorderProgress(Backend& backend, uint32_t block)
        : backend_(backend), block_(block)
    {
        PrepareNext();
    }

    URMA_REORDER_PROGRESS_INLINE bool RunOneReady()
    {
        if (!Ready()) return false;
        // Run returns only after this tile's final-layout stores and counter.
        backend_.Run(fragment_, fragmentIndex_, localTile_);
        ready_ = false;
        tile_ += kAic;
        PrepareNext();
        return true;
    }

    URMA_REORDER_PROGRESS_INLINE bool DrainReady()
    {
        bool progress = false;
        while (RunOneReady()) progress = true;
        return progress;
    }

    URMA_REORDER_PROGRESS_INLINE bool Done() const
    {
        return done_;
    }
    URMA_REORDER_PROGRESS_INLINE bool Ready()
    {
        // These per-invocation flags only move from zero to one. Remember a
        // successful trace-boundary probe until RunOneReady consumes it.
        if (!done_ && !ready_) ready_ = backend_.Ready(fragment_, localTile_);
        return !done_ && ready_;
    }
    URMA_REORDER_PROGRESS_INLINE uint32_t FragmentIndex() const
    {
        return fragmentIndex_;
    }

private:
    URMA_REORDER_PROGRESS_INLINE void PrepareNext()
    {
        while (tile_ >= tileLimit_) {
            if (step_ >= backend_.TaskCount()) {
                if (!done_) backend_.Complete();
                done_ = true;
                return;
            }
            const auto task = backend_.TaskAt(step_++);
            if (task.stage != ComputeBatchStage::Gmm2) continue;
            range_ = backend_.BatchAt(task.batchIndex);
            if (range_.begin == range_.end) continue;
            const auto first = backend_.FragmentAt(range_.begin);
            tilesPerGroup_ = backend_.TileCount(first);
            tile_ = FirstOwnedComputeTile(first.gmm2TileBegin, block_, kAic);
            tileLimit_ = static_cast<uint64_t>(range_.end - range_.begin) * tilesPerGroup_;
        }
        const auto owned = OwnedComputeGroupAt(tile_, tilesPerGroup_, kAic);
        fragmentIndex_ = range_.begin + owned.group;
        localTile_ = owned.firstTile;
        fragment_ = backend_.FragmentAt(fragmentIndex_);
    }

    Backend& backend_;
    uint32_t block_;
    uint32_t step_ = 0U;
    ComputeBatchRange range_{};
    uint32_t tilesPerGroup_ = 0U;
    uint64_t tile_ = 0U;
    uint64_t tileLimit_ = 0U;
    uint32_t fragmentIndex_ = 0U;
    uint32_t localTile_ = 0U;
    Fragment fragment_{};
    bool ready_ = false;
    bool done_ = false;
};

#undef URMA_REORDER_PROGRESS_INLINE
}  // namespace MegaMoeUrma
