// 执行实际角色入口；不提供Linear叶子，防止CV再次绕开Wave调度。
#include <cassert>
#include <cstdint>
#include <vector>
#define __gm__
constexpr int AIC = 0, AIV = 1;
constexpr unsigned INT_CACHELINE = 16U;
namespace AscendC { inline int32_t ReadGmByPassDCache(const int32_t* p) { return *p; } }
namespace MegaMoeImpl { enum class Stage { MOE_GMM1_ACTIVATION }; }
struct Aiv1ActivationUb { static void Init() {} static void Finish() {} };
std::vector<int> events;
template <bool, bool = false, class Cfg, class Ws, class Params>
void PrepareMoeExpertTokenCountTable(const Cfg&, const Ws&, const Params&, int) { events.push_back(2); }
template <class Ptr, class Ws>
void WaitForMoeExpertTokenCountReady(Ptr, const Ws&, unsigned) { events.push_back(3); }
struct Dump { void UpdateStage(MegaMoeImpl::Stage) { events.push_back(0); } };
template <int Core> struct Harness {
    static constexpr int g_coreType = Core;
    struct SharedActivationCursor { bool done = false; };
    Dump exceptionDump_;
    bool computeReplay_;
    unsigned sub;
    int commonConfig_=0, tokenDispatchScratch_=0;
    unsigned sharedExpertNum_=0;
    struct { unsigned blockIdx = 0; } countWorkspace_{};
    int32_t flagSlots_[64]{2};
    struct WsInfo { int32_t* flagSendCntCalToUpdParamsPtr; };
    struct ParamsWrap { WsInfo workspaceInfo; } params_{WsInfo{flagSlots_}};
    unsigned GetSubBlockIdx() { return sub; }
    void DispatchBuffInit() { events.push_back(1); }
    void TryRunSharedExpertActivation(SharedActivationCursor& s) { s.done = true; }
    void ProcessAiv1PriorityStages(int32_t& ready) { events.push_back(4); ++ready; }
    void ProcessGmmWaves(int32_t& ready) { events.push_back(5); ++ready; }
#include "wave_entry.inc"
};
template <int Core> void Check(unsigned sub, bool replay) {
    events.clear();
    Harness<Core> run{Dump{},replay,sub};
    int32_t ready=7;
    run.ProcessMoeExpertStages(ready);
    std::vector<int> expected{0,1};
    if (!replay) { expected.push_back(2); expected.push_back(3); }
    expected.push_back(Core==AIV && sub==1 ? 4 : 5);
    assert(events==expected && ready==8);
}
int main() {
    for (bool replay : {false,true}) for (unsigned sub : {0U,1U}) {
        Check<AIC>(sub,replay);
        Check<AIV>(sub,replay);
    }
}
