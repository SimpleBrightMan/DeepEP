// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {
enum Phase : uint32_t { Pack = 0, Dispatch, Gmm1, Activation, Gmm2, Combine, Barrier, ReducePhase };

// Each engine/phase owns a complete 128B cache line. Values are raw system
// counter ticks: first start, last end, sum of measured intervals, bytes,
// interval count, reserved, schema. No cross-rank clock subtraction is valid.
class Timer {
public:
    __aicore__ inline void Init(uint64_t address)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        address_ = address;
        if constexpr (g_coreType == AscendC::AIV) {
            core_ = (AscendC::GetBlockIdx() / AscendC::GetTaskRation()) * 3U + AscendC::GetSubBlockIdx() + 1U;
        } else {
            core_ = AscendC::GetBlockIdx() * 3U;
        }
#else
        (void)address;
#endif
    }

    __aicore__ inline void Begin(Phase phase)
    {
        if (address_ == 0U) return;
        start_[phase] = AscendC::GetSystemCycle();
        if (count_[phase] == 0U) first_[phase] = start_[phase];
    }

    __aicore__ inline void End(Phase phase, uint64_t bytes = 0U)
    {
        if (address_ == 0U) return;
        const uint64_t end = AscendC::GetSystemCycle();
        total_[phase] += end - start_[phase];
        bytes_[phase] += bytes;
        ++count_[phase];
        auto* record = reinterpret_cast<__gm__ uint64_t*>(address_) +
                       (static_cast<uint64_t>(core_) * kTimerPhases + phase) * kTimerRecordElements;
        record[0] = first_[phase];
        record[1] = end;
        record[2] = total_[phase];
        record[3] = bytes_[phase];
        record[4] = count_[phase];
        record[5] = 0;
        record[6] = kProtocol;
        AscendC::GlobalTensor<uint64_t> tensor;
        tensor.SetGlobalBuffer(record);
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }

private:
    uint64_t address_ = 0U;
    uint32_t core_ = 0U;
    uint64_t start_[kTimerPhases]{};
    uint64_t first_[kTimerPhases]{};
    uint64_t total_[kTimerPhases]{};
    uint64_t bytes_[kTimerPhases]{};
    uint64_t count_[kTimerPhases]{};
};
}  // namespace MegaMoeUrma
