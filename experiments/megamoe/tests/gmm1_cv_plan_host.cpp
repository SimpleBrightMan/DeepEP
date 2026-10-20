// Execute production CV stepping/address helpers: tails, slot parity and in-place prefix safety.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "common/gmm1_cv_plan.hpp"
#include "common/mega_moe_aiv1_ub_layout.hpp"
using namespace MegaMoeImpl;
int main() {
    unsigned sequence=0;
    for(unsigned rows=1;rows<=256;++rows) {
        Gmm1CvTask task{0,0,rows,128,0};
        unsigned consumed=0, stageCount=0, steps=0, stores=0;
        const unsigned first=sequence;
        do {
            const unsigned count=task.StepCount();
            assert(count>=1&&count<=32);
            const unsigned row=task.nextRow%128, begin=task.nextRow/128*128;
            const Gmm1CvStep step{sequence,row,count,std::min(128U,rows-begin)};
            const unsigned slot=Gmm1CvPlan::SlotOffset(sequence);
            assert(step.InputOffset()+count*512<=slot+Gmm1CvPlan::SlotBytes);
            assert(step.OutputOffset()+count*128<=step.InputOffset()+count*512);
            assert(step.ScaleOffset()>=Gmm1CvPlan::ScaleBase&&step.ScaleOffset()+count*4<=Gmm1CvPlan::End);
            assert(step.ScaleOffset()+count*4<=Gmm1CvPlan::ScaleOffset(sequence)+Gmm1CvPlan::ScaleBytes);
            if(task.StageBegin())++stageCount;
            consumed+=count; ++steps;
            if(step.Last())++stores;
            assert(task.Advance(count)==step.Last());
            if(step.Last())++sequence;
        } while(!task.Complete());
        assert(consumed==rows&&stageCount==Gmm1CvPlan::ChunkCount(rows));
        assert(stores==stageCount&&sequence-first==stageCount&&steps==(rows+31)/32);
    }
    // Future stage input remains intact after every prefix write; not a numeric VF simulation.
    for(unsigned rows=1;rows<=128;++rows)for(unsigned seq:{0U,1U,2U,15U}) {
        std::vector<unsigned char> ub(Gmm1CvPlan::End+32,0xa5);
        for(unsigned row=0;row<rows;row+=32) {
            Gmm1CvStep s{seq,row,std::min(32U,rows-row),rows};
            const auto unread=s.InputOffset()+s.rows*512;
            std::fill(ub.begin()+s.OutputOffset(),ub.begin()+s.OutputOffset()+s.rows*128,0x31);
            assert(std::all_of(ub.begin()+unread,ub.begin()+Gmm1CvPlan::SlotOffset(seq)+rows*512,[](auto x){return x==0xa5;}));
        }
        assert(std::all_of(ub.begin()+Gmm1CvPlan::End,ub.end(),[](auto x){return x==0xa5;}));
    }
    for(unsigned ep:{8,32,64,128})for(unsigned bs:{32,96,256,512,1024,2048,4096}) {
        const auto p=MakeAiv1ResidentUbLayout(7168,ep,384/ep,(bs+255)/256*256,2,7392,4);
        assert(p.valid&&p.dispatchBase==Gmm1CvPlan::End&&p.end<=248*1024);
    }
}
