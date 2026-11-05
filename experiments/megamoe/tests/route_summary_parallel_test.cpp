#include "route_summary_parallel.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
using namespace MegaMoeUrma;

static void check(uint32_t ep,uint32_t experts,uint32_t bs,uint32_t topk,uint32_t cv,uint32_t rank,bool shared,uint32_t kind) {
    Config c{};c.world=ep;c.localExperts=experts/ep;c.bs=bs;c.topk=topk;c.rank=rank;
    c.hidden=5120;c.intermediate=2304;c.cvRows=cv;c.sourceRowCapacity=uint64_t(bs)*topk;
    c.maxPeerChunks=(c.sourceRowCapacity+cv-1U)/cv;c.gmm1AheadMGroups=12U;
    DeviceRouteInfo source{};
    std::mt19937 rng(519+bs+experts+kind);
    uint32_t local=0U;
    for(uint32_t e=0;e<experts;++e){source.outgoing[e]=local;local+=(e+rank)%7U;}
    source.outgoing[experts]=local;
    for(uint32_t s=0;s<ep;++s) {
        source.returnBase[s]=uint64_t(s)*331U;
        uint32_t total=0U;
        for(uint32_t e=0;e<c.localExperts;++e) {
            source.incoming[s][e]=total;
            const uint32_t n=kind==0U?(bs*topk)/(ep*c.localExperts):
                kind==1U?(e==0U && s==ep-1U?bs*topk:0U):rng()%613U;
            total+=n;
        }
        source.incoming[s][c.localExperts]=total;
    }
    auto expected=source;auto expectedConfig=c;
    const bool wave12=ep==4U && (!MEGAMOE_URMA_ROUTE_REORDER || shared);
    BuildRouteSummary(expectedConfig,expected,wave12);

    DeviceRouteInfo actual{};
    std::copy(source.outgoing,source.outgoing+experts+1U,actual.outgoing);
    // Source workers may complete in any order. Each owns only its row/base.
    for(uint32_t s=ep;s--;) {
        std::copy(source.incoming[s],source.incoming[s]+c.localExperts+1U,actual.incoming[s]);
        actual.returnBase[s]=source.returnBase[s];
    }
    std::vector<uint32_t> workers(kAic);
    for(uint32_t i=0;i<kAic;++i)workers[i]=i;
    std::shuffle(workers.begin(),workers.end(),rng);
    for(uint32_t worker:workers) {
        const uint32_t begin=uint64_t(c.localExperts)*worker/kAic,end=uint64_t(c.localExperts)*(worker+1U)/kAic;
        // The real stage only loads incoming into UB; poison all unrelated
        // fields so a hidden dependency on another worker's output is caught.
        DeviceRouteInfo scratch{};
        for(uint32_t s=0;s<ep;++s)std::copy(actual.incoming[s],actual.incoming[s]+c.localExperts+1U,scratch.incoming[s]);
        std::fill(scratch.groupBase,scratch.groupBase+kMaxLocalExperts+1U,0xBADU);
        std::fill(scratch.compactBase,scratch.compactBase+kMaxLocalExperts+1U,0xBADU);
        std::fill(scratch.gmm1Base,scratch.gmm1Base+kMaxLocalExperts+1U,0xBADU);
        SummarizeRouteExperts(c,scratch,begin,end);
        for(uint32_t e=begin;e<end;++e) {
            std::copy(scratch.sources[e],scratch.sources[e]+kWorld+1U,actual.sources[e]);
            actual.groupBase[e+1U]=scratch.groupBase[e+1U];actual.compactBase[e+1U]=scratch.compactBase[e+1U];
            actual.gmm1Base[e+1U]=scratch.gmm1Base[e+1U];
        }
    }
    FinishRouteSummaryPrefix(c,actual,wave12);
    for(uint32_t e=0;e<=c.localExperts;++e) {
        assert(actual.groupBase[e]==expected.groupBase[e]);assert(actual.compactBase[e]==expected.compactBase[e]);
        assert(actual.gmm1Base[e]==expected.gmm1Base[e]);
        if(e<c.localExperts)for(uint32_t s=0;s<=ep;++s)assert(actual.sources[e][s]==expected.sources[e][s]);
    }
    for(uint32_t s=0;s<=ep;++s) {
        assert(actual.sendCvBase[s]==expected.sendCvBase[s]);assert(actual.recvCvBase[s]==expected.recvCvBase[s]);
    }
    assert(c.fragmentCount==expectedConfig.fragmentCount && c.receiveRows==expectedConfig.receiveRows);
    assert(c.gmm1Tiles==expectedConfig.gmm1Tiles && c.gmm2Tiles==expectedConfig.gmm2Tiles);
    assert(c.selfSendRow==expectedConfig.selfSendRow && c.selfRecvRow==expectedConfig.selfRecvRow && c.selfRows==expectedConfig.selfRows);
    assert(c.sendChunkCount==expectedConfig.sendChunkCount && c.recvChunkCount==expectedConfig.recvChunkCount);
    assert(c.packRoundCount==expectedConfig.packRoundCount && c.computeTaskCount==expectedConfig.computeTaskCount);
}
int main() {
    uint32_t count=0;
    for(auto ep:{4U,8U,16U})for(auto experts:{ep,64U,128U,384U,512U})
      for(auto bs:{1U,1024U,8192U})for(auto k:{1U,6U,8U})for(auto cv:{128U,256U,384U})
      for(bool shared:{false,true})for(uint32_t kind=0;kind<3U;++kind)for(auto rank:{0U,ep-1U}) {
        if(experts>kExperts || experts%ep || k>experts)continue;
        check(ep,experts,bs,k,cv,rank,shared,kind);++count;
      }
    std::cout<<"parallel AIV0 route summary cases passed="<<count<<"\n";
}
