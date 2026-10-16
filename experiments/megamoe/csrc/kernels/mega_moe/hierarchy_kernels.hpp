// Copyright (c) 2026, SimpleBright_Man
#pragma once
#include "hierarchy_types.hpp"

// These phase kernels deliberately use a completed-kernel + rank barrier protocol.
// No worker spins waiting for a token whose producer might not be scheduled.
__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_hierarchy_copy_kernel(GM_ADDR tasks, uint32_t count, uint64_t ffts)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(ffts);
    if constexpr (g_coreType == AIC) return;
    if (GetSubBlockIdx() != 1U) return;
    constexpr uint32_t slots = 6;
    constexpr uint32_t stride = 16384;
    auto* table = reinterpret_cast<__gm__ HierarchyCopyTask*>(tasks);
    uint32_t issued = 0;
    for (uint32_t index = GetBlockIdx() / GetTaskRation(); index < count; index += GetBlockNum()) {
        HierarchyCopyTask task{table[index].source, table[index].destination, table[index].bytes, 0};
        auto event = static_cast<TEventID>(issued % slots);
        if (issued >= slots) WaitFlag<HardEvent::MTE3_MTE2>(event);
        LocalTensor<uint8_t> buffer(TPosition::VECCALC, (issued % slots) * stride, stride);
        GlobalTensor<uint8_t> source, destination;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(task.source));
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(task.destination));
        DataCopy(buffer, source, task.bytes);
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        DataCopy(destination, buffer, task.bytes);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        ++issued;
    }
    for (uint32_t slot = 0; slot < slots && slot < issued; ++slot) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_hierarchy_reduce_kernel(GM_ADDR rows, GM_ADDR inputs, uint32_t count, uint32_t hidden,
                                                        uint32_t mode, GM_ADDR output, uint64_t ffts)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(ffts);
    if constexpr (g_coreType == AIC) return;
    if (GetSubBlockIdx() != 1U) return;
    auto* rowTable = reinterpret_cast<__gm__ HierarchyReduceRow*>(rows);
    auto* inputTable = reinterpret_cast<__gm__ HierarchyReduceInput*>(inputs);
    LocalTensor<float> accumulator(TPosition::VECCALC, 0, hidden);
    LocalTensor<float> value(TPosition::VECCALC, hidden * sizeof(float), hidden);
    LocalTensor<bfloat16_t> bf16(TPosition::VECCALC, hidden * sizeof(float) * 2, hidden);
    for (uint32_t index = GetBlockIdx() / GetTaskRation(); index < count; index += GetBlockNum()) {
        HierarchyReduceRow row{rowTable[index].destination, rowTable[index].begin, rowTable[index].count};
        Duplicate(accumulator, 0.0f, hidden);
        PipeBarrier<PIPE_V>();
        for (uint32_t item = row.begin; item < row.begin + row.count; ++item) {
            HierarchyReduceInput input{inputTable[item].source, inputTable[item].weight};
            if (mode == 1U) {
                GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input.source));
                DataCopy(bf16, source, hidden);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(value, bf16, RoundMode::CAST_NONE, hidden);
                PipeBarrier<PIPE_V>();
                union WeightBits {
                    uint32_t bits;
                    float value;
                } weight;
                weight.bits = AscendC::ReadGmByPassDCache(reinterpret_cast<__gm__ uint32_t*>(input.weight));
                Muls(value, value, weight.value, hidden);
                PipeBarrier<PIPE_V>();
            } else {
                GlobalTensor<float> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input.source));
                DataCopy(value, source, hidden);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            Add(accumulator, accumulator, value, hidden);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        if (mode == 3U) {
            GlobalTensor<bfloat16_t> destination;
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output + row.destination));
            Cast(bf16, accumulator, RoundMode::CAST_RINT, hidden);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(destination, bf16, hidden);
        } else {
            GlobalTensor<float> destination;
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(row.destination));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(destination, accumulator, hidden);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_hierarchy_copy_do(uint32_t blocks, uint32_t ub,
                                                                                  void* stream, uint8_t* tasks,
                                                                                  uint32_t count, uint64_t ffts)
{
    mega_moe_hierarchy_copy_kernel<<<blocks, ub, stream>>>(tasks, count, ffts);
}
extern "C" __attribute__((visibility("default"))) void mega_moe_hierarchy_reduce_do(uint32_t blocks, uint32_t ub,
                                                                                    void* stream, uint8_t* rows,
                                                                                    uint8_t* inputs, uint32_t count,
                                                                                    uint32_t hidden, uint32_t mode,
                                                                                    uint8_t* output, uint64_t ffts)
{
    mega_moe_hierarchy_reduce_kernel<<<blocks, ub, stream>>>(rows, inputs, count, hidden, mode, output, ffts);
}
