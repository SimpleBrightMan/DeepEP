#ifdef NDEBUG
#undef NDEBUG
#endif
#include "owned_compute_cursor.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
using namespace MegaMoeUrma;
struct Plan {
    uint32_t n1,n2;std::vector<ComputeBatch> batches;std::vector<ComputeBatchTask> tasks;
    uint32_t TaskCount()const{return tasks.size();}
    ComputeBatchTask Task(uint32_t i)const{return tasks[i];}
    ComputeBatch Batch(uint32_t i)const{return batches[i];}
    uint32_t Tiles(bool g1)const{return g1?n1:n2;}
    Fragment Group(uint32_t i)const{return {i/3,i,uint64_t(i)*256,256,0,uint64_t(i)*n1,uint64_t(i)*n2};}
};
int main(){
    std::mt19937 rng(20261105);size_t cases=0,empty=0;
    using Key=std::tuple<bool,uint32_t,uint32_t,uint64_t>;
    for(uint32_t round=0;round<500;++round){
        Plan p{1U+rng()%64U,1U+rng()%32U,{},{}};
        uint32_t group=0;
        for(uint32_t i=0;i<1U+round%16U;++i){uint32_t n=round<16?round%2:rng()%20;p.batches.push_back({group,n});group+=n;}
        // Two GMM1 batches ahead, plus zero-work and late stage ownership cases.
        uint32_t first=0;
        for(uint32_t g2=0;g2<p.batches.size();++g2){while(first<p.batches.size()&&first<g2+2U)p.tasks.push_back({ComputeBatchStage::Gmm1,first++});p.tasks.push_back({ComputeBatchStage::Gmm2,g2});}
        for(uint32_t core=0;core<kAic;++core){
            std::vector<Key> expected,actual;
            for(auto task:p.tasks){auto b=p.batches[task.batchIndex];bool g1=task.stage==ComputeBatchStage::Gmm1;uint32_t nt=p.Tiles(g1);
                for(uint32_t g=b.fragmentBegin;g<b.fragmentBegin+b.fragmentCount;++g)for(uint32_t t=0;t<nt;++t){uint64_t logical=uint64_t(g)*nt+t;if((g1?kAic-1U-logical%kAic:logical%kAic)==core)expected.emplace_back(g1,g,t,logical);}}
            OwnedTileSequence<Plan> cursor(p,core);RoutedCompletionItem item;
            while(cursor.Next(item))actual.emplace_back(item.gmm1,item.group,item.tile,item.logicalTile);
            assert(actual==expected);assert(!cursor.Next(item));if(actual.empty())++empty;++cases;
        }
    }
    assert(empty>0);std::cout<<"owned cursor cases="<<cases<<" empty="<<empty<<" passed\n";
}
