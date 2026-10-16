// Execute production orchestration with recording stage stubs; not a device/DMA simulator.
#include "common/mega_moe_shared_policy.hpp"
#include <cassert>
#include <string>
#include <vector>

using namespace MegaMoeImpl;
enum { AIC, AIV, PIPE_ALL, OVERFLOW_MODE_CTRL };
namespace MegaMoeImpl {
enum class Stage { INPUT_PREPARE, SHARED_EXPERT_GMM1, CROSS_RANK_SYNC_INPUT,
                   SHARED_EXPERT_GMM2, CROSS_RANK_SYNC_OUTPUT, UNPERMUTE, COMPLETE };
}
namespace AscendTimer {
enum { KERNEL_TIMING, AIC_WAIT_INPUT_PREP, AIV_ROUTE_MASK, AIV_INPUT_CROSS_RANK_SYNC,
       AIV_EXPORT_SYNC, AIV_FINAL_CROSS_RANK_WAIT, AIV_UNPERMUTE, AIV_RESET };
}
struct Timer {
    void Start(int) {} void End(int) {} void FlushDynamicCounts() {}
};
struct ExceptionDump { void UpdateStage(Stage) {} };
struct AivJobContext { unsigned jobIndex, totalJobs; };
using MegaMoeUnpermuteBufferConfig = int;

template <bool W4, int Role>
struct Pipeline {
    static constexpr bool ENABLE_A8W4 = W4;
    static constexpr int g_coreType = Role, CombineQuantMode = 0;
    static constexpr bool TopkWeightsPrefetch = false;
    bool computeReplay_ = false;
    int resetBatchElementCount_ = 0, resetTensor_ = 0;
    void SendAndQuantBuffInit() { events.push_back("buffers"); }
    template <bool, class... T> void ResetSyncStatus(T...) { events.push_back("reset"); }
    static constexpr unsigned GMM1_TILE_M = 256;
    using TopkWeightsType = int;
    unsigned sharedExpertNum_, subBlock;
    InputPreparationMode inputMode_;
    uint32_t blockIdx_ = 0, blockNum_ = 32, rankId_ = 0, worldSize_ = 8, aivCoreIdx_ = 0;
    int aivJob_ = 0, commonConfig_ = 0, countWorkspace_ = 0, tokenDispatchScratch_ = 0;
    int g_winRankAddr_ = 0, sendMaskConfig_ = 0, sendMaskScratch_ = 0;
    int tokenUnpermuteConfig_ = 0, tokenUnpermuteScratch_ = 0;
    struct {
        struct { int rankSyncInWorldPtr = 0; } peermemInfo;
        struct { int flagTopkValidIndexSyncPtr = 0; } workspaceInfo;
    } params_;
    Timer timer_;
    ExceptionDump exceptionDump_;
    std::vector<std::string> events;
    Pipeline(unsigned shared, unsigned bs, unsigned sub)
        : sharedExpertNum_(shared), subBlock(sub), inputMode_(SelectInputPreparationMode(W4, shared, bs)) {}
    unsigned GetSubBlockIdx() { return subBlock; }
    template <int, int> int GetCtrlSpr() { return 0; }
    template <int, int> void SetCtrlSpr(int) {}
    template <int> void PipeBarrier() { events.push_back("drain"); }
    template <bool VectorOnly> void SyncAll() { events.push_back(VectorOnly ? "aiv-sync" : "local-sync"); }
    template <bool Aiv1Only = false, class... T> void CrossRankSyncInWorldSize(T...) {
        events.push_back(Aiv1Only ? "aiv1-handshake" : "rank-handshake");
    }
    template <class... T> void GatherAndSendExpertCompactRoutes(T...) { events.push_back("route"); }
    template <class... T> void ExportCompactExpertTokenCounts(T...) { events.push_back("export"); }
    template <int, class, bool, unsigned, class... T> void UnpermuteTokens(T...) { events.push_back("unpermute"); }
    int InitTokenUnpermuteBuffers() { return 0; }
    void ProcessInputPreparationStage() { events.push_back("input"); }
    void ProcessSharedExpertGmm1(int32_t&) { events.push_back("shared-g1"); }
    void ProcessSharedExpertGmm2() { events.push_back("shared-g2"); }
    void ProcessSharedExpertTail() { events.push_back("shared-tail"); }
    void ProcessMoeExpertStages(int32_t&) { events.push_back("routed-w4"); }
    void ProcessMoeExpertStages() { events.push_back("routed-w8"); }

    // Bodies are taken verbatim from the current production header by the Python test.
    #include "process_wave_methods.inc"
};

template <bool W4, int Role>
static void CheckOrder(unsigned shared, unsigned bs, unsigned sub)
{
    Pipeline<W4, Role> p(shared, bs, sub);
    p.ProcessWave(p);
    // Pre-refactor contract (8ffafeb): same per-role call order, with independent input booleans.
    std::vector<std::string> expected{"input"};
    if (Role == AIV) expected.push_back("drain");
    expected.push_back("local-sync");
    if (shared) expected.push_back("shared-g1");
    const bool oldDecode = W4 && bs <= 256;
    const bool oldRouteAiv1 = W4 && shared && bs > 256;
    if (oldRouteAiv1) {
        if (Role == AIV && sub == 1) {
            expected.insert(expected.end(), {"route", "drain", "aiv1-handshake"});
        }
    } else if (!oldDecode) {
        expected.push_back("rank-handshake");
    }
    expected.push_back(W4 ? "routed-w4" : "routed-w8");
    if (W4 && shared) expected.push_back("shared-tail");
    if (Role == AIV) expected.insert(expected.end(), {"export", "drain", "aiv-sync"});
    if (!W4 && shared) expected.push_back("shared-g2");
    if (Role == AIV) expected.insert(expected.end(), {"rank-handshake", "unpermute"});
    assert(p.events == expected);
}

int main()
{
    // 诊断 replay 只能走计算；重构公共入口时不能意外恢复通信或输出归约。
    {
        Pipeline<true, AIC> p(0, 4096, 0);
        p.computeReplay_ = true;
        p.ProcessWave(p);
        assert(p.events == std::vector<std::string>({"buffers", "reset", "local-sync", "routed-w4", "drain", "local-sync"}));
    }
    {
        Pipeline<true, AIV> p(0, 4096, 1);
        p.computeReplay_ = true;
        p.ProcessWave(p);
        assert(p.events == std::vector<std::string>({"buffers", "reset", "drain", "local-sync", "routed-w4", "drain", "local-sync"}));
    }
    for (unsigned shared : {0U, 1U, 8U}) {
        for (unsigned bs : {1U, 96U, 255U, 256U, 257U, 4096U}) {
            CheckOrder<true, AIC>(shared, bs, 0);
            CheckOrder<false, AIC>(shared, bs, 0);
            for (unsigned sub : {0U, 1U}) {
                CheckOrder<true, AIV>(shared, bs, sub);
                CheckOrder<false, AIV>(shared, bs, sub);
            }
        }
    }
}
