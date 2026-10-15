// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "kernel_launch.hpp"

using namespace AscendC;

#if defined(__NPU_DEVICE__)
#include "kernel_operator.h"

namespace deepep::dispatch_bench {
namespace {

using namespace AscendC;

constexpr uint32_t kTaskWords = sizeof(CopyTask) / sizeof(uint32_t);
constexpr uint32_t kWorkerWords = sizeof(Worker) / sizeof(uint32_t);
constexpr uint32_t kMetaBytes = kMetaWords * sizeof(uint32_t);

// The six ring slots use the same event IDs and dependencies as MegaMoE's
// Fetch/StoreDispatchTokenAndMetaInfo. No event or UB slot survives Run().
template <MetadataMode Mode>
class DispatchBandwidthCopy {
public:
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR peerBases, GM_ADDR tasks, GM_ADDR metadataInput, GM_ADDR workers,
                                GM_ADDR outputTokens, GM_ADDR outputScales, GM_ADDR outputMetadata, GM_ADDR timer,
                                KernelConfig config, uint32_t workerId)
    {
        config_ = config;
        workerId_ = workerId;
        tasks_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tasks));
        if constexpr (Mode != MetadataMode::Scalar) {
            metadataInput_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataInput));
        }
        outputTokens_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outputTokens));
        outputScales_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outputScales));
        outputMetadata_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputMetadata));
        timer_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(timer));

        GlobalTensor<uint32_t> workerWords;
        workerWords.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workers));
        const uint32_t workerOffset = workerId * kWorkerWords;
        worker_.traffic = workerWords.GetValue(workerOffset);
        worker_.begin = workerWords.GetValue(workerOffset + 1U);
        worker_.count = workerWords.GetValue(workerOffset + 2U);

        GlobalTensor<uint64_t> bases;
        bases.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(peerBases));
        for (uint32_t peer = 0U; peer < config_.world; ++peer) {
            peerBases_[peer] = bases.GetValue(peer);
        }

        // Packed source records include scale padding; the local scale output
        // stores only scale_elements bytes, exactly as the production Dispatch.
        scaleOffset_ = (config_.hidden + 255U) / 256U * 256U;
        const uint32_t rowStorageBytes = config_.ring_slots * config_.row_bytes;
        uint32_t metadataBytes = config_.ring_slots * kMetaBytes;
        if constexpr (Mode == MetadataMode::Preload) {
            // Empty workers still require one timer record after Run().
            metadataBytes = worker_.count == 0U ? kMetaBytes : worker_.count * kMetaBytes;
        }
        pipe.InitBuffer(storage_, rowStorageBytes + metadataBytes);
        rows_ = storage_.Get<uint8_t>();
        metadata_ = rows_[rowStorageBytes].ReinterpretCast<uint32_t>();
        if constexpr (Mode == MetadataMode::Scalar) {
            for (uint32_t word = 0U; word < config_.ring_slots * kMetaWords; ++word) {
                metadata_.SetValue(word, 0U);
            }
        }
    }

    __aicore__ inline void Run()
    {
        uint32_t initializedSlots = 0U;
        uint32_t nextSlot = 0U;
        uint32_t pendingSlot = 0U;
        uint32_t pendingDestination = 0U;
        uint32_t pendingMetadataIndex = 0U;
        bool hasPending = false;
        const uint64_t start = GetSystemCycle();
        if constexpr (Mode == MetadataMode::Preload) {
            if (worker_.count != 0U) {
                // Timed once per invocation. The first row's MTE2->MTE3
                // notification also covers this earlier read on the same pipe.
                DataCopy(metadata_, metadataInput_[static_cast<uint64_t>(worker_.begin) * kMetaWords],
                         worker_.count * kMetaWords);
            }
        }

        // Keep the ring live across peers and repeats. The next remote read is
        // submitted before the previous row's local writes, just like Dispatch.
        for (uint32_t repeat = 0U; repeat < config_.repeats; ++repeat) {
            for (uint32_t row = 0U; row < worker_.count; ++row) {
                const uint32_t taskIndex = worker_.begin + row;
                const CopyTask task = ReadTask(taskIndex);
                const bool reuse = initializedSlots == config_.ring_slots;
                Fetch(nextSlot, taskIndex, task, reuse);
                if (hasPending) {
                    Store(pendingSlot, pendingDestination, pendingMetadataIndex);
                }
                pendingSlot = nextSlot;
                pendingDestination = task.destination;
                // Preloaded records are indexed within this worker and reused
                // from row zero on every repeat; data slots continue rotating.
                pendingMetadataIndex = Mode == MetadataMode::Preload ? row : nextSlot;
                hasPending = true;
                if (!reuse) {
                    ++initializedSlots;
                }
                ++nextSlot;
                if (nextSlot == config_.ring_slots) {
                    nextSlot = 0U;
                }
            }
        }

        if (hasPending) {
            Store(pendingSlot, pendingDestination, pendingMetadataIndex);
        }
        for (uint32_t slot = 0U; slot < initializedSlots; ++slot) {
            const TEventID event = static_cast<TEventID>(slot);
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            if constexpr (Mode == MetadataMode::Scalar) {
                WaitFlag<HardEvent::MTE3_S>(event);
            }
        }
        if constexpr (Mode != MetadataMode::Scalar) {
            // One final Scalar completion fence replaces all per-row fences.
            // It also protects the UB record reused by WriteTimer, including
            // when this worker owns no rows. Event 6 is outside the row ring.
            const TEventID event = static_cast<TEventID>(6U);
            SetFlag<HardEvent::MTE3_S>(event);
            WaitFlag<HardEvent::MTE3_S>(event);
        }
        const uint64_t end = GetSystemCycle();
        WriteTimer(start, end);
    }

private:
    __aicore__ inline CopyTask ReadTask(uint32_t index)
    {
        const uint32_t offset = index * kTaskWords;
        return {tasks_.GetValue(offset), tasks_.GetValue(offset + 1U), tasks_.GetValue(offset + 2U),
                tasks_.GetValue(offset + 3U)};
    }

    __aicore__ inline void Fetch(uint32_t slot, uint32_t taskIndex, const CopyTask& task, bool reuse)
    {
        const TEventID event = static_cast<TEventID>(slot);
        if (reuse) {
            WaitFlag<HardEvent::MTE3_MTE2>(event);
        }
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(peerBases_[task.peer]));
        DataCopy(rows_[slot * config_.row_bytes], source[static_cast<uint64_t>(task.token) * config_.row_bytes],
                 config_.row_bytes);
        if constexpr (Mode == MetadataMode::Dma) {
            // The same MTE3->MTE2 reuse dependency protects both UB buffers.
            // Publish readiness only after the remote row and local metadata.
            DataCopy(metadata_[slot * kMetaWords], metadataInput_[static_cast<uint64_t>(taskIndex) * kMetaWords],
                     kMetaWords);
        }
        SetFlag<HardEvent::MTE2_MTE3>(event);

        if constexpr (Mode == MetadataMode::Scalar) {
            if (reuse) {
                WaitFlag<HardEvent::MTE3_S>(event);
            }
            const uint32_t metadataOffset = slot * kMetaWords;
            metadata_.SetValue(metadataOffset, task.peer);
            metadata_.SetValue(metadataOffset + 1U, task.token);
            metadata_.SetValue(metadataOffset + 2U, task.topk_slot);
            SetFlag<HardEvent::S_MTE3>(event);
        }
    }

    __aicore__ inline void Store(uint32_t slot, uint32_t destination, uint32_t metadataIndex)
    {
        const TEventID event = static_cast<TEventID>(slot);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        const LocalTensor<uint8_t> row = rows_[slot * config_.row_bytes];
        DataCopyPad(outputTokens_[static_cast<uint64_t>(destination) * config_.hidden], row,
                    {1U, config_.hidden, 0U, 0U, 0U});
        DataCopyPad(outputScales_[static_cast<uint64_t>(destination) * config_.scale_elements], row[scaleOffset_],
                    {1U, config_.scale_elements, 0U, 0U, 0U});
        if constexpr (Mode == MetadataMode::Scalar) {
            WaitFlag<HardEvent::S_MTE3>(event);
        }
        DataCopy(outputMetadata_[static_cast<uint64_t>(destination) * kMetaWords],
                 metadata_[metadataIndex * kMetaWords], kMetaWords);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        if constexpr (Mode == MetadataMode::Scalar) {
            SetFlag<HardEvent::MTE3_S>(event);
        }
    }

    __aicore__ inline void WriteTimer(uint64_t start, uint64_t end)
    {
        // Every row DMA has completed, so the first 32B metadata slot is free.
        // Use MTE3 for the record as well, including a completion wait before
        // the pipe/UB is destroyed. Event 6 is outside the data ring's IDs 0..5.
        LocalTensor<uint64_t> record = metadata_.ReinterpretCast<uint64_t>();
        record.SetValue(0U, start);
        record.SetValue(1U, end);
        record.SetValue(2U, static_cast<uint64_t>(worker_.traffic));
        record.SetValue(3U, static_cast<uint64_t>(worker_.count) * config_.repeats);
        const TEventID event = static_cast<TEventID>(6U);
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        DataCopy(timer_[static_cast<uint64_t>(workerId_) * 4U], record, 4U);
        SetFlag<HardEvent::MTE3_S>(event);
        WaitFlag<HardEvent::MTE3_S>(event);
    }

    KernelConfig config_{};
    Worker worker_{};
    uint32_t workerId_ = 0U;
    uint32_t scaleOffset_ = 0U;
    uint64_t peerBases_[kMaxRanks]{};
    TBuf<TPosition::VECCALC> storage_;
    LocalTensor<uint8_t> rows_;
    LocalTensor<uint32_t> metadata_;
    GlobalTensor<uint32_t> tasks_;
    GlobalTensor<uint32_t> metadataInput_;
    GlobalTensor<uint8_t> outputTokens_;
    GlobalTensor<uint8_t> outputScales_;
    GlobalTensor<uint32_t> outputMetadata_;
    GlobalTensor<uint64_t> timer_;
};

}  // namespace
}  // namespace deepep::dispatch_bench
#endif

// The 28 active lanes occupy AIV1 across the same 28 mixed groups as MegaMoE.
// AIC/AIV0 are idle, and no all-core barrier depends on their participation.
template <deepep::dispatch_bench::MetadataMode Mode>
__schedmode__(1) __global__
    __mix__(1, 2) void dispatch_bandwidth_kernel(GM_ADDR peerBases, GM_ADDR tasks, GM_ADDR metadataInput,
                                                 GM_ADDR workers, GM_ADDR outputTokens, GM_ADDR outputScales,
                                                 GM_ADDR outputMetadata, GM_ADDR timer,
                                                 deepep::dispatch_bench::KernelConfig config)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (AscendC::GetSubBlockIdx() != 1U) {
        return;
    }
    const uint32_t worker = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    if (worker >= config.cores) {
        return;
    }
    AscendC::TPipe pipe;
    deepep::dispatch_bench::DispatchBandwidthCopy<Mode> copy;
    copy.Init(pipe, peerBases, tasks, metadataInput, workers, outputTokens, outputScales, outputMetadata, timer, config,
              worker);
    copy.Run();
#endif
}

extern "C" void dispatch_bandwidth_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream, uint8_t* peer_bases,
                                             uint8_t* tasks, uint8_t* metadata_input, uint8_t* workers,
                                             uint8_t* output_tokens, uint8_t* output_scales, uint8_t* output_metadata,
                                             uint8_t* timer, deepep::dispatch_bench::KernelConfig config)
{
    if (config.metadata_mode == static_cast<uint32_t>(deepep::dispatch_bench::MetadataMode::Dma)) {
        dispatch_bandwidth_kernel<deepep::dispatch_bench::MetadataMode::Dma><<<block_dim, ub_bytes, stream>>>(
            peer_bases, tasks, metadata_input, workers, output_tokens, output_scales, output_metadata, timer, config);
    } else if (config.metadata_mode == static_cast<uint32_t>(deepep::dispatch_bench::MetadataMode::Preload)) {
        dispatch_bandwidth_kernel<deepep::dispatch_bench::MetadataMode::Preload><<<block_dim, ub_bytes, stream>>>(
            peer_bases, tasks, metadata_input, workers, output_tokens, output_scales, output_metadata, timer, config);
    } else {
        dispatch_bandwidth_kernel<deepep::dispatch_bench::MetadataMode::Scalar><<<block_dim, ub_bytes, stream>>>(
            peer_bases, tasks, metadata_input, workers, output_tokens, output_scales, output_metadata, timer, config);
    }
}
