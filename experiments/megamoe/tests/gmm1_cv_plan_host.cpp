// Production task stepping/layout; include tails, cross-tile slot parity and two independent engines.
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include "common/gmm1_cv_plan.hpp"
#include "common/mega_moe_aiv1_ub_layout.hpp"
using namespace MegaMoeImpl;
#define __aicore__
namespace AscendC {
enum class HardEvent { MTE3_V, V_MTE3 };
std::array<std::deque<std::function<bool()>>,2> queues;
std::array<bool,2> flags{};
template <HardEvent H> void SetFlag(unsigned id) {
    assert(id==0);
    queues[H==HardEvent::MTE3_V?1:0].push_back([]{ assert(!flags[unsigned(H)]); flags[unsigned(H)]=true; return true; });
}
template <HardEvent H> void WaitFlag(unsigned id) {
    assert(id==0);
    queues[H==HardEvent::MTE3_V?0:1].push_back([]{ if(!flags[unsigned(H)])return false; flags[unsigned(H)]=false; return true; });
}
void drain() {
    for(unsigned steps=0;;++steps) {
        if(queues[0].empty()&&queues[1].empty())return;
        bool progress=false;
        for(auto& q:queues)if(!q.empty()&&q.front()()){q.pop_front();progress=true;}
        if(!progress||steps>10000)throw std::logic_error("credit deadlock");
    }
}
}
#include "common/gmm1_cv_scratch.hpp"
int main() {
    unsigned sequence=0;
    for(unsigned rows=1;rows<=256;++rows) {
        Gmm1CvTask task{0,0,rows,128,0};
        unsigned consumed=0, stageCount=0, steps=0;
        const unsigned first=sequence;
        do {
            const unsigned count=task.StepCount();
            assert(count>=1&&count<=32);
            const unsigned offset=Gmm1CvPlan::SlotOffset(sequence)+task.nextRow%128*512;
            assert(offset+count*512<=Gmm1CvPlan::SlotOffset(sequence)+Gmm1CvPlan::SlotBytes);
            if(task.StageBegin())++stageCount;
            consumed+=count; ++steps;
            if(task.Advance(count))++sequence;
        } while(!task.Complete());
        assert(consumed==rows&&stageCount==Gmm1CvPlan::ChunkCount(rows));
        assert(sequence-first==stageCount&&steps==(rows+31)/32);
    }
    for(unsigned ep:{8,32,64,128})for(unsigned bs:{32,96,256,512,1024,2048,4096}) {
        const auto p=MakeAiv1ResidentUbLayout(7168,ep,384/ep,(bs+255)/256*256,2,7392,4);
        assert(p.valid&&p.dispatchBase==Gmm1CvPlan::End&&p.end<=248*1024);
    }
    // Queued V writes must not overtake MTE3 reads, even when all steps are submitted by Scalar.
    unsigned written=0, read=0;
    Gmm1CvScratch::Init();
    for(unsigned step=1;step<=64;++step) {
        Gmm1CvScratch::Acquire();
        AscendC::queues[0].push_back([&,step]{assert(read==step-1);written=step;return true;});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::queues[1].push_back([&,step]{assert(written==step);read=step;return true;});
        Gmm1CvScratch::Release();
    }
    Gmm1CvScratch::Finish(); AscendC::drain();
    assert(read==64&&!AscendC::flags[0]&&!AscendC::flags[1]);
    Gmm1CvScratch::Init();Gmm1CvScratch::Finish();AscendC::drain();
    assert(!AscendC::flags[0]);
}
