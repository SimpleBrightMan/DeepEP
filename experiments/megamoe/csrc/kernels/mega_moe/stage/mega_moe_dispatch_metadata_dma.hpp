// Copyright (c) 2026, SimpleBright_Man

#ifndef MEGA_MOE_DISPATCH_METADATA_DMA_H
#define MEGA_MOE_DISPATCH_METADATA_DMA_H

// Included after the stage's AscendC types/constants. The metadata table is
// immutable local GM, indexed by (source rank, source token, top-k slot).
// Token data and metadata share the existing ring slot and MTE completion
// events: no scalar access may reuse its metadata UB while MTE3 consumes it.
namespace MegaMoeImpl {

using namespace AscendC;

template <bool IsBufferReuse, typename ActivationType>
__aicore__ inline void FetchDispatchTokenWithDmaMetadata(LocalTensor<ActivationType> copy,
                                                         GlobalTensor<ActivationType> remote, uint64_t remoteOffset,
                                                         uint32_t rowBytes, LocalTensor<int32_t> metaSlot,
                                                         GlobalTensor<int32_t> metaInput, uint64_t metadataRow,
                                                         int32_t bufferIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    if constexpr (IsBufferReuse) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    DataCopy(copy, remote[remoteOffset], rowBytes);
    DataCopy(metaSlot, metaInput[metadataRow * INT32_PER_256B], INT32_PER_256B);
    // Both reads must finish before either payload or metadata is written out.
    SetFlag<HardEvent::MTE2_MTE3>(eventId);
}

template <typename ActivationType, typename QuantScaleType>
__aicore__ inline void StoreDispatchTokenWithDmaMetadata(
    GlobalTensor<ActivationType> tokenOut, GlobalTensor<QuantScaleType> scaleOut, GlobalTensor<int32_t> metaOut,
    LocalTensor<ActivationType> copy, LocalTensor<int32_t> metaSlot, uint32_t tokenAlignedBytes, int64_t tokenElems,
    int64_t scaleElems, int32_t dstRow, int32_t bufferIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<QuantScaleType> scale = copy[tokenAlignedBytes].template ReinterpretCast<QuantScaleType>();
    DataCopyPad(tokenOut[static_cast<int64_t>(dstRow) * tokenElems], copy,
                {1, static_cast<uint16_t>(tokenElems * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleOut[static_cast<int64_t>(dstRow) * scaleElems], scale,
                {1, static_cast<uint16_t>(scaleElems * sizeof(QuantScaleType)), 0U, 0U, 0U});
    DataCopy(metaOut[static_cast<int64_t>(dstRow) * INT32_PER_256B], metaSlot, INT32_PER_256B);
    // Reuse protects all three writes, including the metadata slot.
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

__aicore__ inline void DrainDispatchMetadataDmaSlot(int32_t bufferIdx)
{
    // Consume the final slot event. This is an MTE2 dependency, not a scalar
    // completion wait: callers retain PublishGmm1TileReady's MTE3_S fence.
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(bufferIdx));
}

}  // namespace MegaMoeImpl
#endif
