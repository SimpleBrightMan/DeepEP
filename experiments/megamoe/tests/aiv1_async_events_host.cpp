// Not a performance simulator: independent engine queues detect premature UB reuse and publication.
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#define __aicore__
#define __gm__
using TEventID = unsigned;
namespace AscendC {
enum class HardEvent { V_MTE2, MTE3_V, MTE2_V, V_MTE3, MTE3_S, MTE2_MTE3, MTE3_MTE2 };
using H = HardEvent;
constexpr int V=0, M2=1, M3=2, S=3;
std::array<std::deque<std::function<bool()>>,3> queue;
std::array<std::array<bool,8>,7> flag{};
int source(H h) { return h==H::V_MTE2||h==H::V_MTE3 ? V : h==H::MTE2_V||h==H::MTE2_MTE3 ? M2 : M3; }
int target(H h) { return h==H::MTE3_S ? S : h==H::V_MTE2||h==H::MTE3_MTE2 ? M2 : h==H::V_MTE3||h==H::MTE2_MTE3 ? M3 : V; }
void run_until(const std::function<bool()>& done) {
    for (unsigned step=0;!done();++step) {
        if(step>10000) throw std::logic_error("event dependency did not terminate");
        bool progress=false;
        for(auto& q:queue) if(!q.empty()&&q.front()()) { q.pop_front(); progress=true; }
        if(!progress) throw std::logic_error("event deadlock / missing credit");
    }
}
template <H h> void SetFlag(TEventID id) {
    assert(id<8);
    queue[source(h)].push_back([id]{ auto& f=flag[unsigned(h)][id]; assert(!f); f=true; return true; });
}
template <H h> void WaitFlag(TEventID id) {
    assert(id<8);
    if(target(h)==S) {
        run_until([id]{ return flag[unsigned(h)][id]; });
        flag[unsigned(h)][id]=false;
    } else queue[target(h)].push_back([id]{ auto& f=flag[unsigned(h)][id]; if(!f)return false; f=false; return true; });
}
unsigned inputVersion=0, completedVersion=0;
void AtomicAdd(int32_t* p,int32_t n) { assert(completedVersion==unsigned(*p)+1); *p+=n; }
}
#include "common/aiv1_activation_ub.hpp"
using namespace AscendC;
using MegaMoeImpl::Aiv1ActivationUb;
using MegaMoeImpl::Aiv1AsyncEvents;
namespace AscendTimer { constexpr int NO_WAVE=-1, AIV_SWIGLU_SYNC=0; }
struct Consumer {
    int32_t* activationReadyFlag_;
    bool activationPending_=false;
    struct { void SetWave(int) {} int BeginBoundary(int){return 0;} void EndBoundary(int,int){} } timer_;
#include "activation_retire.inc"
};
#include "combine_drain.inc"
int main() {
    // Missing ready must not be silently accepted by the model.
    bool rejected=false;
    try { WaitFlag<H::MTE3_S>(6); } catch(const std::logic_error&) { rejected=true; }
    assert(rejected);
    Aiv1ActivationUb::Init();
    int32_t ready=0;
    Consumer consumer{&ready};
    uint32_t sends=0;
    for(unsigned version=1;version<=64;++version) {
        Aiv1ActivationUb::Acquire();
        queue[M2].push_back([version]{ assert(completedVersion==version-1); inputVersion=version; return true; });
        SetFlag<H::MTE2_V>(0); WaitFlag<H::MTE2_V>(0);
        queue[V].push_back([version]{ assert(inputVersion==version); return true; });
        SetFlag<H::V_MTE3>(0); WaitFlag<H::V_MTE3>(0);
        queue[M3].push_back([version]{ assert(inputVersion==version); completedVersion=version; return true; });
        Aiv1ActivationUb::Release();
        SetFlag<H::MTE3_S>(Aiv1AsyncEvents::ActivationDone);
        consumer.activationPending_=true;
        assert(ready==int32_t(version-1)); // issue is not completion/publication
        // Independent resident Combine rows may remain in flight across activation retirement.
        const unsigned slot=Aiv1AsyncEvents::CombineBase+sends%Aiv1AsyncEvents::CombineSlots;
        if(sends>=Aiv1AsyncEvents::CombineSlots) WaitFlag<H::MTE3_MTE2>(slot);
        SetFlag<H::MTE2_MTE3>(slot); WaitFlag<H::MTE2_MTE3>(slot);
        SetFlag<H::MTE3_MTE2>(slot);
        ++sends;
        assert(consumer.RetireActivation() && ready==int32_t(version));
        assert(!consumer.RetireActivation());
    }
    Aiv1ActivationUb::Finish();
    DrainCombineRowBuffers(sends,Aiv1AsyncEvents::CombineSlots,Aiv1AsyncEvents::CombineBase);
    assert(sends==0);
    run_until([]{for(auto& q:queue)if(!q.empty())return false; return true;});
    for(auto& domain:flag) for(bool f:domain) assert(!f);
    // Zero-activation invocation also returns its one initial credit.
    Aiv1ActivationUb::Init(); Aiv1ActivationUb::Finish();
    run_until([]{for(auto& q:queue)if(!q.empty())return false; return true;});
    for(auto& domain:flag) for(bool f:domain) assert(!f);
}
