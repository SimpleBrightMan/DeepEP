// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"

namespace MegaMoeUrma {

#if defined(__NPU_DEVICE__)
#define MEGAMOE_WORK_INLINE __aicore__ inline
#else
#define MEGAMOE_WORK_INLINE inline
#endif

// One owned M-group, not a new scheduling/communication unit. The cursor only
// reads immutable plan data: preparing a successor must never wait for input,
// publish completion, or touch the active context's events/buffers.
struct ComputeWork {
    ComputeBatchStage stage = ComputeBatchStage::None;
    uint32_t fragmentIndex = 0U;
    uint32_t batchBegin = 0U;
    uint32_t firstTile = 0U;
    uint32_t tilesPerGroup = 0U;
    bool endBatch = false;
    Fragment fragment{};
};

MEGAMOE_WORK_INLINE constexpr bool ContinueComputeContext(const ComputeWork& current, const ComputeWork& next)
{
    return current.stage != ComputeBatchStage::None && current.stage == next.stage;
}

// Plan supplies TaskCount(), Task(index), Batch(index), Group(index), and
// TileCount(stage, fragment). Both engines use the same cursor independently;
// empty/unowned batches are skipped without introducing a handshake or claim.
template <class Plan>
class ComputeWorkCursor {
public:
    MEGAMOE_WORK_INLINE ComputeWorkCursor(const Plan& plan, uint32_t worker, uint32_t workers = kAic)
        : plan_(plan), worker_(worker), workers_(workers)
    {
    }

    MEGAMOE_WORK_INLINE bool Next(ComputeWork& result)
    {
        Fragment first{};
        bool loadedFirst = false;
        while (ownedTile_ >= tileLimit_) {
            if (step_ >= plan_.TaskCount()) {
                result = {};
                return false;
            }
            const auto task = plan_.Task(step_++);
            const auto batch = plan_.Batch(task.batchIndex);
            if (batch.fragmentCount == 0U) continue;
            first = plan_.Group(batch.fragmentBegin);
            const uint32_t width = plan_.TileCount(task.stage, first);
            const uint64_t prefix = task.stage == ComputeBatchStage::Gmm1 ? first.gmm1TileBegin : first.gmm2TileBegin;
            const uint64_t begin = FirstOwnedComputeTile(prefix, worker_, workers_);
            const uint64_t limit = static_cast<uint64_t>(batch.fragmentCount) * width;
            if (begin >= limit) continue;
            stage_ = task.stage;
            batchBegin_ = batch.fragmentBegin;
            width_ = width;
            ownedTile_ = begin;
            tileLimit_ = limit;
            loadedFirst = true;
        }
        const auto owned = OwnedComputeGroupAt(ownedTile_, width_, workers_);
        ownedTile_ += static_cast<uint64_t>(owned.tileCount) * workers_;
        const uint32_t index = batchBegin_ + owned.group;
        result.stage = stage_;
        result.fragmentIndex = index;
        result.batchBegin = batchBegin_;
        result.firstTile = owned.firstTile;
        result.tilesPerGroup = width_;
        result.endBatch = ownedTile_ >= tileLimit_;
        result.fragment = loadedFirst && owned.group == 0U ? first : plan_.Group(index);
        return true;
    }

private:
    const Plan& plan_;
    uint32_t worker_;
    uint32_t workers_;
    uint32_t step_ = 0U;
    ComputeBatchStage stage_ = ComputeBatchStage::None;
    uint32_t batchBegin_ = 0U;
    uint32_t width_ = 0U;
    uint64_t ownedTile_ = 0U;
    uint64_t tileLimit_ = 0U;
};

#undef MEGAMOE_WORK_INLINE
}  // namespace MegaMoeUrma
