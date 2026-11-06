#ifdef NDEBUG
#undef NDEBUG
#endif
#include "completion_batch.hpp"
#include <cassert>
#include <vector>
#include <iostream>
using namespace MegaMoeUrma;

struct FakeTimer {};
struct Consumer {
    std::vector<bool> gmm1;
    uint32_t issued=0,consumed=0,polls=0,waits=0;
    std::vector<uint32_t> completed,reordered,released;
    bool grow=false;
    bool workDone=false;
    void ReleaseCompleted(){assert(workDone);released.push_back(consumed-1);workDone=false;}
    uint32_t IssuedCount(){++polls;return issued-consumed;}
    bool NextIsGmm1(){return gmm1.at(consumed);}
    void CompleteIssuedOne(RoutedCompletionItem& item,FakeTimer&) {
        assert(consumed<issued);
        ++waits; // model real hardware completion, NOT just issue detection
        item.gmm1=gmm1.at(consumed);item.logicalTile=consumed;
        item.group=consumed/3;item.tile=consumed%3;item.fragment.rowBegin=1000+consumed;
        completed.push_back(consumed);
        ++consumed; // RX ownership is held until the work finishes
        if(grow)issued=gmm1.size(); // new arrivals cannot grow the captured batch
    }
};
struct Work {
    Consumer& c;uint32_t closes=0;
    void Before(){++closes;}
    void Activate(const RoutedCompletionItem& item,FakeTimer&) {
        assert(item.gmm1 && c.completed.back()==item.logicalTile);
        assert(c.released.size()==item.logicalTile);c.workDone=true;
    }
    void Reorder(const RoutedCompletionItem& item,FakeTimer&) {
        assert(!item.gmm1 && c.completed.back()==item.logicalTile);
        assert(c.released.size()==item.logicalTile);c.workDone=true;
        assert(item.fragment.rowBegin==1000+item.logicalTile); // descriptor forwarded directly
        c.reordered.push_back(item.logicalTile);
    }
};
int main() {
    FakeTimer timer;uint32_t scenarios=0;
    for(uint32_t mask=0;mask<256;++mask)for(uint32_t initially=0;initially<=8;++initially) {
        Consumer c;for(uint32_t i=0;i<8;++i)c.gmm1.push_back((mask>>i)&1);
        c.issued=initially;Work work{c};
        const uint32_t count=ConsumeIssuedReorders(c,work,timer);
        assert(c.polls==1 && c.waits==count && count<=kDirectReorderBatch);
        assert(work.closes==(initially!=0));
        const uint32_t expected=initially?1U:0U;
        assert(count==expected);
        // Drain mixed streams across many bounded turns; no missing/duplicate tiles.
        c.issued=8;
        while(c.consumed<8)assert(ConsumeIssuedReorders(c,work,timer)>0);
        assert(c.waits==8 && c.released.size()==8);
        std::vector<uint32_t> expectedReorder;
        for(uint32_t i=0;i<8;++i){assert(c.completed[i]==i);if(!c.gmm1[i])expectedReorder.push_back(i);}
        assert(c.reordered==expectedReorder);++scenarios;
    }
    Consumer c{{false,false,false,false},1};c.grow=true;Work work{c};
    assert(ConsumeIssuedReorders(c,work,timer)==1 && c.issued==4 && c.polls==1);
    assert(ConsumeIssuedReorders(c,work,timer)==1 && c.consumed==2);
    assert(ConsumeIssuedReorders(c,work,timer)==1 && c.consumed==3);
    assert(ConsumeIssuedReorders(c,work,timer)==1 && c.consumed==4);
    assert(ConsumeIssuedReorders(c,work,timer)==0);
    std::cout<<scenarios+1<<" completion batch cases passed\n";
}
