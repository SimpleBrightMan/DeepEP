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
    __aicore__ inline void Init(uint64_t address, uint64_t traceAddress = 0U, uint64_t traceCapacity = 0U)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        address_ = address;
        traceAddress_ = traceAddress;
        traceCapacity_ = traceCapacity;
        if constexpr (g_coreType == AscendC::AIV) {
            core_ = (AscendC::GetBlockIdx() / AscendC::GetTaskRation()) * 3U + AscendC::GetSubBlockIdx() + 1U;
        } else {
            core_ = AscendC::GetBlockIdx() * 3U;
        }
#else
        (void)address;
        (void)traceAddress;
        (void)traceCapacity;
#endif
    }

    __aicore__ inline uint64_t TraceBegin() const
    {
        return traceAddress_ == 0U ? 0U : AscendC::GetSystemCycle();
    }

    // This interval measures the instrumented call, not a synthesized DMA or
    // Cube hardware duration. Deferred tiles end at completion-flag publication
    // and can overlap other records. Recording adds no pipeline drain.
    __aicore__ inline void TraceEnd(TraceEvent event, uint64_t start, const TraceMeta& meta = TraceMeta{},
                                    uint64_t flags = 0U)
    {
        if (traceAddress_ == 0U) return;
        const uint64_t end = AscendC::GetSystemCycle();
        if (traceCount_ >= traceCapacity_) {
            ++traceDropped_;
            return;
        }
        auto* record = TraceHeader() + kTraceHeaderElements + traceCount_ * kTraceRecordElements;
        record[0] = start;
        record[1] = end;
        record[2] = static_cast<uint64_t>(event);
        record[3] = meta.peer;
        record[4] = meta.cv;
        record[5] = meta.fragment;
        record[6] = meta.expert;
        record[7] = meta.row;
        record[8] = meta.rows;
        record[9] = meta.tileM;
        record[10] = meta.tileN;
        record[11] = meta.columns;
        record[12] = meta.bytes;
        record[13] = flags;
        record[14] = 0U;
        record[15] = 0U;
        FlushTraceLine(record);
        ++traceCount_;
    }

    // Called once by every AIC/AIV engine, including engines with zero events.
    // An unfinished kernel leaves a zero header and is rejected by the reader.
    __aicore__ inline void FinishTrace()
    {
        if (traceAddress_ == 0U) return;
        auto* header = TraceHeader();
        header[0] = kTraceSchema;
        header[1] = traceCapacity_;
        header[2] = traceCount_;
        header[3] = traceDropped_;
        header[4] = core_;
        FlushTraceLine(header);
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
    __aicore__ inline __gm__ uint64_t* TraceHeader() const
    {
        return reinterpret_cast<__gm__ uint64_t*>(traceAddress_) +
               static_cast<uint64_t>(core_) * (kTraceHeaderElements + traceCapacity_ * kTraceRecordElements);
    }

    __aicore__ inline void FlushTraceLine(__gm__ uint64_t* record) const
    {
        AscendC::GlobalTensor<uint64_t> tensor;
        tensor.SetGlobalBuffer(record);
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }

    uint64_t address_ = 0U;
    uint64_t traceAddress_ = 0U;
    uint64_t traceCapacity_ = 0U;
    uint64_t traceCount_ = 0U;
    uint64_t traceDropped_ = 0U;
    uint32_t core_ = 0U;
    uint64_t start_[kTimerPhases]{};
    uint64_t first_[kTimerPhases]{};
    uint64_t total_[kTimerPhases]{};
    uint64_t bytes_[kTimerPhases]{};
    uint64_t count_[kTimerPhases]{};
};
}  // namespace MegaMoeUrma
