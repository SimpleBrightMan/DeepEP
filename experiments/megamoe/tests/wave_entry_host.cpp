// 执行实际角色入口；不提供Linear叶子，防止CV再次绕开Wave调度。
#include <cassert>
#include <cstdint>
#include <vector>
constexpr int AIC = 0, AIV = 1;
namespace MegaMoeImpl { enum class Stage { MOE_GMM1_ACTIVATION }; }
std::vector<int> events;
template <bool, class Params> void PrepareMoeExpertTokenCountTable(int,int,const Params&,int) { events.push_back(2); }
void WaitForMoeExpertTokenCountReady(int,int,int) { events.push_back(3); }
struct Dump { void UpdateStage(MegaMoeImpl::Stage) { events.push_back(0); } };
template <int Core, bool Cv> struct Harness {
    static constexpr int g_coreType = Core;
    static constexpr bool UseCv = Cv;
    Dump exceptionDump_;
    bool computeReplay_;
    unsigned sub;
    int commonConfig_=0, countWorkspace_=0, tokenDispatchScratch_=0;
    struct { struct { int flagSendCntCalToUpdParamsPtr=0; } workspaceInfo; } params_;
    unsigned GetSubBlockIdx() { return sub; }
    void DispatchBuffInit() { events.push_back(1); }
    void ProcessAiv1PriorityStages(int32_t& ready) { events.push_back(4); ++ready; }
    void ProcessGmmWaves(int32_t& ready) { events.push_back(5); ++ready; }
#include "wave_entry.inc"
};
template <int Core, bool Cv> void Check(unsigned sub, bool replay) {
    events.clear();
    Harness<Core,Cv> run{Dump{},replay,sub};
    int32_t ready=7;
    run.ProcessMoeExpertStages(ready);
    std::vector<int> expected{0,1};
    if (!replay) { expected.push_back(2); if (sub==0) expected.push_back(3); }
    expected.push_back(Core==AIV && sub==1 ? 4 : 5);
    assert(events==expected && ready==8);
}
int main() {
    for (bool replay : {false,true}) for (unsigned sub : {0U,1U}) {
        Check<AIC,false>(sub,replay); Check<AIC,true>(sub,replay);
        Check<AIV,false>(sub,replay); Check<AIV,true>(sub,replay);
    }
}
