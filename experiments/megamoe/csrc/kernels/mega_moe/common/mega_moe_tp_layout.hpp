// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_tp_layout.hpp
 * \brief MegaMoE TP 模式 Tiling ABI、跨卡 Peermem 对称窗口、本地 GM Workspace 与 AIV1 常驻 UB 布局。
 *        Host 准备校验与 Device 地址绑定共享同一套公式，禁止两侧各写一份偏移。
 */

#ifndef MEGA_MOE_TP_LAYOUT_HPP
#define MEGA_MOE_TP_LAYOUT_HPP

#include <cstdint>
#include "mega_moe_constants.hpp"
#include "mega_moe_aiv1_ub_layout.hpp"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#define MEGAMOE_TP_DEFINED_GM_ADDR 1
#endif

namespace MegaMoeImpl {

constexpr int64_t TP_PEERMEM_SYNC_BYTES = 60LL * 1024LL;

enum class SharedPrecisionMode : uint8_t {
    NONE = 0U,
    FP8_MX = 1U,
    BF16 = 2U,
};

constexpr uint32_t TP_DEFAULT_M_GROUPS_PER_WAVE = 22U;
constexpr uint32_t TP_DEFAULT_AG_CORES_PER_PEER = 11U;
constexpr uint32_t TP_DEFAULT_RS_TOKENS_PER_CHUNK = 64U;
constexpr uint32_t TP_PERMUTE_RING_SLOTS = 6U;

struct alignas(8) MegaMoeTpTilingData {
    uint32_t bs;                     // 本卡本次实际输入 token 数 (e.g. 4096)
    uint32_t numMaxTokensPerRank;    // 全卡一致的单卡最大 token 容量上界
    uint32_t tpWorldSize;            // TP 卡数 (e.g. 4)
    uint32_t numExperts;             // 路由专家总数 (每卡持有全部路由专家, e.g. 352)
    uint32_t topK;                   // 每 token 选中的路由专家数 (e.g. 8)
    uint32_t h;                      // token hidden 维 (e.g. 4096)
    uint32_t routedGmm1OutDim;       // 本卡路由 GMM1 输出维 = 2 * (I_global / TP) (e.g. 768)
    uint32_t sharedExpertNum;        // 共享专家数 (不做 TP, e.g. 1)
    uint32_t sharedGmm1OutDim;       // 共享专家 GMM1 输出维 = 2 * I_shared (e.g. 3072)
    uint32_t aicNum;                 // 32
    uint32_t aivNum;                 // 64
    uint32_t mGroupsPerWave;         // 每 Wave 消费的 256-row M-group 预算 (默认 22)
    uint32_t agCoresPerPeer;         // AllGather 分配给每个远端 rank 的 AIV 数 (默认 11)
    uint32_t rsTokensPerChunk;       // ReduceScatter 分核/同步粒度 (默认 64 tokens/chunk)
    float clampLimit;                // SwiGLU 截断阈值
    float activationAlpha;           // 默认 1.0f
    float activationBeta;            // 默认 1.0f
    uint8_t sharedPrecisionMode;     // SharedPrecisionMode: 0=无, 1=FP8_MX, 2=BF16
    uint8_t routeWeightBeforeQuant;  // 0=Unpermute FP32 加权, 1=SwiGLU 量化前乘权重
    uint8_t actMode;                 // MegaMoeActMode::SWIGLU
    uint8_t actSubMode;              // MegaMoeActSubMode::DEFAULT
};
static_assert(sizeof(MegaMoeTpTilingData) == 72, "MegaMoeTpTilingData ABI size changed");

#if defined(__CCE_AICORE__)
#define MEGAMOE_TP_HOST_DEVICE __host__ __aicore__
#else
#define MEGAMOE_TP_HOST_DEVICE
#endif

MEGAMOE_TP_HOST_DEVICE constexpr int64_t TpCeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

MEGAMOE_TP_HOST_DEVICE constexpr int64_t TpCeilAlign(int64_t a, int64_t align)
{
    return TpCeilDiv(a, align) * align;
}

MEGAMOE_TP_HOST_DEVICE inline GM_ADDR TpAllocAligned(GM_ADDR base, int64_t& offset, int64_t bytes)
{
    GM_ADDR ptr = base != nullptr ? (base + offset) : nullptr;
    offset += TpCeilAlign(bytes, ALIGN_512);
    return ptr;
}

MEGAMOE_TP_HOST_DEVICE inline GM_ADDR TpAllocFlagInts(GM_ADDR base, int64_t& offset, int64_t int32Count)
{
    GM_ADDR ptr = base != nullptr ? (base + offset) : nullptr;
    offset += int32Count * static_cast<int64_t>(sizeof(int32_t));
    return ptr;
}

MEGAMOE_TP_HOST_DEVICE constexpr int64_t CalcTpMxScaleStrideBytes(int64_t cols)
{
    return TpCeilDiv(cols, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

/*
 * TP 模式对称窗口布局：
 *   [0, 60KiB)          rankSyncInWorld（复用跨卡握手与每 AIV 单调递增 epoch 计数槽）
 *   agSrcData           [M_max, H] FP8
 *   agSrcScale          [M_max, S_h] E8M0
 *   agSrcTopkIds        [M_max, topK] int32
 *   agSrcTopkWeights    [M_max, topK] float32
 *   rsRecvBuf           [tpWorldSize, M_max, H] BF16
 *   rsChunkReady        [tpWorldSize, maxChunksPerRank] * 64B cacheline (存单调递增 epoch，免清零)
 */
struct TpPeermemLayout {
    int64_t rankSyncOffset;
    int64_t agSrcDataOffset;
    int64_t agSrcScaleOffset;
    int64_t agSrcTopkIdsOffset;
    int64_t agSrcTopkWeightsOffset;
    int64_t rsRecvBufOffset;
    int64_t rsChunkReadyOffset;
    int64_t totalBytes;
};

MEGAMOE_TP_HOST_DEVICE inline TpPeermemLayout BuildTpPeermemLayout(
    int64_t maxTokensPerRank, int64_t tpWorldSize, int64_t topK, int64_t h,
    int64_t rsTokensPerChunk = TP_DEFAULT_RS_TOKENS_PER_CHUNK)
{
    TpPeermemLayout layout{};
    int64_t offset = 0;
    layout.rankSyncOffset = offset;
    offset += TP_PEERMEM_SYNC_BYTES;

    const int64_t scaleH = CalcTpMxScaleStrideBytes(h);
    layout.agSrcDataOffset = offset;
    offset += TpCeilAlign(maxTokensPerRank * h, ALIGN_512);

    layout.agSrcScaleOffset = offset;
    offset += TpCeilAlign(maxTokensPerRank * scaleH, ALIGN_512);

    layout.agSrcTopkIdsOffset = offset;
    offset += TpCeilAlign(maxTokensPerRank * topK * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);

    layout.agSrcTopkWeightsOffset = offset;
    offset += TpCeilAlign(maxTokensPerRank * topK * static_cast<int64_t>(sizeof(float)), ALIGN_512);

    layout.rsRecvBufOffset = offset;
    offset += TpCeilAlign(tpWorldSize * maxTokensPerRank * h * static_cast<int64_t>(sizeof(uint16_t)), ALIGN_512);

    const int64_t chunkGranularity = rsTokensPerChunk > 0 ? rsTokensPerChunk : TP_DEFAULT_RS_TOKENS_PER_CHUNK;
    const int64_t maxChunksPerRank = TpCeilDiv(maxTokensPerRank, chunkGranularity);
    layout.rsChunkReadyOffset = offset;
    offset += TpCeilAlign(
        tpWorldSize * maxChunksPerRank * static_cast<int64_t>(INT_CACHELINE * sizeof(int32_t)), ALIGN_512);

    layout.totalBytes = offset;
    return layout;
}

MEGAMOE_TP_HOST_DEVICE inline int64_t CalcTpPeermemLeastSize(
    int64_t maxTokensPerRank, int64_t tpWorldSize, int64_t topK, int64_t h,
    int64_t rsTokensPerChunk = TP_DEFAULT_RS_TOKENS_PER_CHUNK)
{
    return BuildTpPeermemLayout(maxTokensPerRank, tpWorldSize, topK, h, rsTokensPerChunk).totalBytes;
}

struct TpPeermemInfo {
    GM_ADDR rankSyncInWorldPtr{nullptr};
    GM_ADDR agSrcDataPtr{nullptr};
    GM_ADDR agSrcScalePtr{nullptr};
    GM_ADDR agSrcTopkIdsPtr{nullptr};
    GM_ADDR agSrcTopkWeightsPtr{nullptr};
    GM_ADDR rsRecvBufPtr{nullptr};
    GM_ADDR rsChunkReadyPtr{nullptr};
    TpPeermemLayout layout{};

    MEGAMOE_TP_HOST_DEVICE TpPeermemInfo() = default;
    MEGAMOE_TP_HOST_DEVICE TpPeermemInfo(GM_ADDR base, const MegaMoeTpTilingData* t)
    {
        layout = BuildTpPeermemLayout(t->numMaxTokensPerRank, t->tpWorldSize, t->topK, t->h, t->rsTokensPerChunk);
        if (base != nullptr) {
            rankSyncInWorldPtr = base + layout.rankSyncOffset;
            agSrcDataPtr = base + layout.agSrcDataOffset;
            agSrcScalePtr = base + layout.agSrcScaleOffset;
            agSrcTopkIdsPtr = base + layout.agSrcTopkIdsOffset;
            agSrcTopkWeightsPtr = base + layout.agSrcTopkWeightsOffset;
            rsRecvBufPtr = base + layout.rsRecvBufOffset;
            rsChunkReadyPtr = base + layout.rsChunkReadyOffset;
        }
    }
};

struct TpWorkspaceInfo {
    // AllGather 落地与全局路由表区域
    GM_ADDR agXDataPtr{nullptr};
    GM_ADDR agXScalePtr{nullptr};
    GM_ADDR agTopkIdsPtr{nullptr};
    GM_ADDR agTopkWeightsPtr{nullptr};
    GM_ADDR routePartialCountsPtr{nullptr};
    GM_ADDR routePartialOffsetsPtr{nullptr};
    GM_ADDR expertCountsPtr{nullptr};
    GM_ADDR expertOffsetsPtr{nullptr};
    GM_ADDR expertRevTokenNumsPtr{nullptr};
    GM_ADDR permRowToTokenPtr{nullptr};
    GM_ADDR tokenToPermRowPtr{nullptr};
    GM_ADDR metaInfoPtr{nullptr};

    // 路由专家计算流水区域
    GM_ADDR permutedXPtr{nullptr};
    GM_ADDR permutedScalePtr{nullptr};
    GM_ADDR gmm1OutPtr{nullptr};
    GM_ADDR actQuantDataPtr{nullptr};
    GM_ADDR actQuantScalePtr{nullptr};
    GM_ADDR gmm2OutPtr{nullptr};

    // 共享专家独立计算区域 (本卡 bs)
    GM_ADDR sharedBf16InputPtr{nullptr};
    GM_ADDR sharedGmm1OutPtr{nullptr};
    GM_ADDR sharedActDataPtr{nullptr};
    GM_ADDR sharedActScalePtr{nullptr};
    GM_ADDR sharedGmm2OutPtr{nullptr};

    // 连续软同步通知区（每次调用在 Phase 0 一次性清零）
    GM_ADDR flagRegionBeginPtr{nullptr};
    GM_ADDR flagLocalPrepDonePtr{nullptr};
    GM_ADDR flagRouteGroupSyncPtr{nullptr};
    GM_ADDR flagRouteReadyPtr{nullptr};
    GM_ADDR flagPermuteToGmm1Ptr{nullptr};
    GM_ADDR flagGmm1IssuedPtr{nullptr};
    GM_ADDR flagActToGmm2Ptr{nullptr};
    GM_ADDR flagRoutedGmm2DonePtr{nullptr};
    GM_ADDR flagSharedGmm1DonePtr{nullptr};
    GM_ADDR flagSharedActDonePtr{nullptr};
    GM_ADDR sharedGmm2TileCounterPtr{nullptr};
    int64_t flagResetElementCount{0};
    int32_t mGroupFlagSlotsPerExpert{0};

    int64_t workspaceSize{0};

    MEGAMOE_TP_HOST_DEVICE TpWorkspaceInfo() = default;
    MEGAMOE_TP_HOST_DEVICE TpWorkspaceInfo(GM_ADDR base, const MegaMoeTpTilingData* t)
    {
        Initialize(base, t);
    }

private:
    MEGAMOE_TP_HOST_DEVICE void Initialize(GM_ADDR base, const MegaMoeTpTilingData* t)
    {
        const int64_t maxLocalTokens = static_cast<int64_t>(t->numMaxTokensPerRank);
        const int64_t world = static_cast<int64_t>(t->tpWorldSize);
        const int64_t maxGlobalTokens = maxLocalTokens * world;
        const int64_t topK = static_cast<int64_t>(t->topK);
        const int64_t maxRoutedRows = maxGlobalTokens * topK;
        const int64_t experts = static_cast<int64_t>(t->numExperts);
        const int64_t h = static_cast<int64_t>(t->h);
        const int64_t scaleH = CalcTpMxScaleStrideBytes(h);
        const int64_t routedN1 = static_cast<int64_t>(t->routedGmm1OutDim);
        const int64_t routedInter = routedN1 / ACTIVATION_N_HALF;
        const int64_t scaleRoutedInter = CalcTpMxScaleStrideBytes(routedInter);
        const int64_t aicNum = static_cast<int64_t>(t->aicNum);
        const int64_t aivNum = static_cast<int64_t>(t->aivNum);
        const int64_t expertStride = TpCeilAlign(experts, static_cast<int64_t>(INT_CACHELINE));

        int64_t offset = 0;

        agXDataPtr = TpAllocAligned(base, offset, maxGlobalTokens * h);
        agXScalePtr = TpAllocAligned(base, offset, maxGlobalTokens * scaleH);
        agTopkIdsPtr = TpAllocAligned(base, offset, maxGlobalTokens * topK * static_cast<int64_t>(sizeof(int32_t)));
        agTopkWeightsPtr = TpAllocAligned(base, offset, maxGlobalTokens * topK * static_cast<int64_t>(sizeof(float)));

        routePartialCountsPtr =
            TpAllocAligned(base, offset, aivNum * expertStride * static_cast<int64_t>(sizeof(int32_t)));
        routePartialOffsetsPtr =
            TpAllocAligned(base, offset, aivNum * expertStride * static_cast<int64_t>(sizeof(int32_t)));
        expertCountsPtr = TpAllocAligned(base, offset, expertStride * static_cast<int64_t>(sizeof(int32_t)));
        expertOffsetsPtr = TpAllocAligned(
            base, offset, TpCeilAlign(experts + 1, static_cast<int64_t>(INT_CACHELINE)) * static_cast<int64_t>(sizeof(int32_t)));
        expertRevTokenNumsPtr =
            TpAllocAligned(base, offset, aicNum * expertStride * static_cast<int64_t>(sizeof(int32_t)));
        permRowToTokenPtr = TpAllocAligned(base, offset, maxRoutedRows * static_cast<int64_t>(sizeof(int32_t)));
        tokenToPermRowPtr = TpAllocAligned(base, offset, maxRoutedRows * static_cast<int64_t>(sizeof(int32_t)));
        metaInfoPtr =
            TpAllocAligned(base, offset, maxRoutedRows * static_cast<int64_t>(META_INFO_SIZE * sizeof(int32_t)));

        // 连续 flag 通知区：每个逻辑槽独占 64B cache line，支持 ResetWorkspaceRegion 一次清零。
        // 单专家最多接收 maxGlobalTokens 行（同 token 的 topK 专家互异）。
        const int64_t maxMGroupsPerExpert = TpCeilDiv(maxGlobalTokens, static_cast<int64_t>(L1_TILE_M_256));
        mGroupFlagSlotsPerExpert = static_cast<int32_t>(maxMGroupsPerExpert * INT_CACHELINE);

        const int64_t flagBegin = offset;
        flagRegionBeginPtr = base != nullptr ? (base + offset) : nullptr;

        flagLocalPrepDonePtr = TpAllocFlagInts(base, offset, INT_CACHELINE);
        flagRouteGroupSyncPtr = TpAllocFlagInts(base, offset, aivNum * INT_CACHELINE);
        flagRouteReadyPtr = TpAllocFlagInts(base, offset, INT_CACHELINE);
        flagPermuteToGmm1Ptr = TpAllocFlagInts(base, offset, experts * mGroupFlagSlotsPerExpert);
        flagGmm1IssuedPtr = TpAllocFlagInts(base, offset, aicNum * INT_CACHELINE);
        flagActToGmm2Ptr = TpAllocFlagInts(base, offset, experts * mGroupFlagSlotsPerExpert);
        flagRoutedGmm2DonePtr = TpAllocFlagInts(base, offset, aicNum * INT_CACHELINE);
        if (t->sharedExpertNum > 0U) {
            flagSharedGmm1DonePtr = TpAllocFlagInts(base, offset, aicNum * INT_CACHELINE);
            flagSharedActDonePtr = TpAllocFlagInts(base, offset, aicNum * INT_CACHELINE);
            const int64_t sharedGroups = TpCeilDiv(maxLocalTokens, static_cast<int64_t>(L1_TILE_M_256));
            sharedGmm2TileCounterPtr =
                TpAllocFlagInts(base, offset, static_cast<int64_t>(t->sharedExpertNum) * sharedGroups * INT_CACHELINE);
        }
        offset = TpCeilAlign(offset, ALIGN_512);
        flagResetElementCount = (offset - flagBegin) / static_cast<int64_t>(sizeof(int32_t));

        permutedXPtr = TpAllocAligned(base, offset, maxRoutedRows * h);
        permutedScalePtr = TpAllocAligned(base, offset, maxRoutedRows * scaleH);
        gmm1OutPtr = TpAllocAligned(base, offset, maxRoutedRows * routedN1 * static_cast<int64_t>(sizeof(uint16_t)));
        actQuantDataPtr = TpAllocAligned(base, offset, maxRoutedRows * routedInter);
        actQuantScalePtr = TpAllocAligned(base, offset, maxRoutedRows * scaleRoutedInter);
        gmm2OutPtr = TpAllocAligned(base, offset, maxRoutedRows * h * static_cast<int64_t>(sizeof(uint16_t)));

        if (t->sharedExpertNum > 0U) {
            const int64_t sharedCount = static_cast<int64_t>(t->sharedExpertNum);
            const int64_t sharedN1 = static_cast<int64_t>(t->sharedGmm1OutDim);
            const int64_t sharedInter = sharedN1 / ACTIVATION_N_HALF;
            const int64_t scaleSharedInter = CalcTpMxScaleStrideBytes(sharedInter);
            sharedBf16InputPtr =
                TpAllocAligned(base, offset, maxLocalTokens * h * static_cast<int64_t>(sizeof(uint16_t)));
            sharedGmm1OutPtr =
                TpAllocAligned(base, offset, sharedCount * maxLocalTokens * sharedN1 * static_cast<int64_t>(sizeof(uint16_t)));
            // 预留 BF16 字节数以同时兼容 FP8_MX (1B/elem) 与 BF16 (2B/elem) 两种共享专家精度。
            sharedActDataPtr = TpAllocAligned(
                base, offset, sharedCount * maxLocalTokens * sharedInter * static_cast<int64_t>(sizeof(uint16_t)));
            sharedActScalePtr = TpAllocAligned(base, offset, sharedCount * maxLocalTokens * scaleSharedInter);
            sharedGmm2OutPtr =
                TpAllocAligned(base, offset, sharedCount * maxLocalTokens * h * static_cast<int64_t>(sizeof(uint16_t)));
        }

        workspaceSize = offset;
    }
};

/*
 * Phase 2 AIV1 常驻 UB 分区：
 *   [0, ActivationBytes)                     SwiGLU + MXFP8 量化工作区（~66.5 KiB，Routed 与 Shared 串行复用）
 *   [permDataBase, permScaleBase)            Permute 6 槽 token 数据环（6 * H 字节，H=4096 时 24 KiB）
 *   [permScaleBase, permIdxBase)             Permute 6 槽 scale 环（6 * Align32(scaleH) 字节）
 *   [permIdxBase, permEnd)                   Permute 批量索引缓存（4 KiB，单批最多 1024 个 permRowToToken 索引）
 */
constexpr uint32_t TP_PERMUTE_IDX_BATCH_CAP = 1024U;

struct TpAiv1ResidentUbLayout {
    uint32_t activationBytes;
    uint32_t permDataBase;
    uint32_t permScaleBase;
    uint32_t permIdxBase;
    uint32_t permRowDataBytes;
    uint32_t permRowScaleBytes;
    uint32_t end;
    bool valid;
};

MEGAMOE_TP_HOST_DEVICE constexpr TpAiv1ResidentUbLayout MakeTpAiv1ResidentUbLayout(uint32_t h)
{
    const uint32_t actBytes =
        static_cast<uint32_t>(AlignResidentUb(Aiv1ResidentUbLayout::ActivationBytes));
    const uint32_t rowDataBytes = static_cast<uint32_t>(AlignResidentUb(h));
    const uint32_t scaleCols = ((h + 63U) / 64U) * 2U;
    const uint32_t rowScaleBytes = static_cast<uint32_t>(AlignResidentUb(scaleCols));
    const uint32_t permDataBase = actBytes;
    const uint32_t permScaleBase = permDataBase + TP_PERMUTE_RING_SLOTS * rowDataBytes;
    const uint32_t permIdxBase = permScaleBase + TP_PERMUTE_RING_SLOTS * rowScaleBytes;
    const uint32_t end = permIdxBase + TP_PERMUTE_IDX_BATCH_CAP * static_cast<uint32_t>(sizeof(int32_t));
    return {actBytes, permDataBase, permScaleBase, permIdxBase, rowDataBytes, rowScaleBytes, end,
            h > 0U && h % 64U == 0U && end <= 256U * 1024U};
}

#undef MEGAMOE_TP_HOST_DEVICE

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_TP_LAYOUT_HPP
