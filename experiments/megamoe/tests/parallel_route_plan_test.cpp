#include "parallel_route_plan.hpp"
#include "expert_plan.hpp"
#if MEGAMOE_URMA_ROUTE_REORDER
#include "peer_reorder_plan.hpp"
#endif
#include <cassert>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
using namespace MegaMoeUrma;
struct Output {
    ExpertPlan p;
    std::vector<std::tuple<uint64_t,uint64_t,uint32_t,uint32_t>> spans;
    std::vector<std::pair<uint32_t,uint32_t>> ranges;
    template<class T>static void set(std::vector<T>& v,uint32_t i,T x){if(v.size()<=i)v.resize(i+1);v[i]=x;}
    uint32_t Gmm1Tiles(uint32_t rows,const Config& c){
#if MEGAMOE_URMA_ROUTE_REORDER
        return (c.intermediate+127U)/128U;
#else
        return Gmm1ComputeTileCount(rows,c);
#endif
    }
    void FragmentAt(uint32_t i,Fragment v){set(p.fragments,i,v);}
    void Segment(uint32_t i,RowSegment v){set(p.segments,i,v);}
    void Send(uint32_t i,Chunk v){set(p.sendChunks,i,v);}
    void Receive(uint32_t i,Chunk v){set(p.recvChunks,i,v);}
    void Link(uint32_t i,uint32_t v){set(p.chunkGroupIndices,i,v);}
    void Batch(uint32_t i,ComputeBatch v){set(p.computeBatches,i,v);}
    void Task(uint32_t i,ComputeBatchTask v){set(p.computeTasks,i,v);}
    void PackChunk(uint32_t cv,uint32_t peer,Chunk c){if(p.packRounds.size()<=cv)p.packRounds.resize(cv+1);p.packRounds[cv].chunks[peer]=c;}
    void PackRows(uint32_t cv,uint64_t n){p.packRounds[cv].rows=n;}
    void ReorderSpan(uint32_t i,uint64_t src,uint64_t dst,uint32_t g,uint32_t rows){set(spans,i,std::make_tuple(src,dst,g,rows));}
    void ReorderRange(uint32_t i,uint32_t begin,uint32_t count){set(ranges,i,std::make_pair(begin,count));}
};
static void check(uint32_t ep,uint32_t experts,uint32_t bs,uint32_t k,uint32_t cv,uint32_t distribution, bool shared) {
    std::mt19937 rng(91+distribution+bs);
    std::vector<int64_t> counts(ep*experts,0);
    for(uint32_t s=0;s<ep;++s)for(uint32_t i=0;i<bs*k;++i) {
        const uint32_t expert=distribution==0?(i+s*19)%experts:distribution==1?rng()%experts:
            distribution==2?0:distribution==3?(s%2?experts-1U:1U):rng()%7;
        ++counts[s*experts+expert];
    }
    for(uint32_t rank=0;rank<ep;++rank) {
        Config c{};c.bs=bs;c.world=ep;c.rank=rank;c.topk=k;c.hidden=5120;c.intermediate=2304;
        c.localExperts=experts/ep;c.cvRows=cv;
        // Deliberately allocate beyond this invocation's BS; sparse layout must
        // use capacity, never actual counts. Include changed-BS reuse semantics.
        c.sourceRowCapacity=uint64_t(RouteMin(bs+17,kMaxTokensPerRank))*k;c.maxPeerChunks=(c.sourceRowCapacity+cv-1)/cv;
        c.gmm1AheadMGroups=12;
        auto expected=BuildExpertPlan(counts.data(),c,uint64_t(ep)*c.sourceRowCapacity);
        DeviceRouteInfo info{};
        for(uint32_t e=0;e<experts;++e)info.outgoing[e+1]=info.outgoing[e]+counts[rank*experts+e];
        uint64_t compactPeer[kWorld+1]{};
        for(uint32_t s=0;s<ep;++s) {
            for(uint32_t e=0;e<rank*c.localExperts;++e)info.returnBase[s]+=counts[s*experts+e];
            for(uint32_t e=0;e<c.localExperts;++e)
                info.incoming[s][e+1]=info.incoming[s][e]+counts[s*experts+rank*c.localExperts+e];
            compactPeer[s+1]=compactPeer[s]+info.incoming[s][c.localExperts];
        }
        const bool wave=ep==4&&(!MEGAMOE_URMA_ROUTE_REORDER||shared);
#if MEGAMOE_URMA_ROUTE_REORDER
        // Independent legacy reorder compiler consumes its original COMPACT
        // receive layout, before we translate expected destination addresses.
        const auto expectedReorder=BuildPeerReorderPlan(expected.config,expected.fragments,expected.segments,expected.recvChunks);
        // Legacy unitflag Host planner predates shared EP4 wave12. Apply the
        // public S1 schedule contract without consulting the new summary.
        if(wave) {
            expected.computeBatches.clear();expected.computeTasks.clear();
            for(uint32_t first=0,w=0;first<expected.fragments.size();first+=12,++w) {
                const uint32_t count=std::min<uint32_t>(12,expected.fragments.size()-first);
                expected.computeBatches.push_back({first,count});
                expected.computeTasks.push_back({ComputeBatchStage::Gmm1,w});
                expected.computeTasks.push_back({ComputeBatchStage::Gmm2,w});
            }
            expected.config.computeTaskCount=expected.computeTasks.size();
        }
#endif
        Output out;
        BuildRouteSummary(c,info,wave);
        std::vector<uint32_t> gp(c.fragmentCount+1U),cp(c.recvChunkCount+1U);
        // Exercise every worker, including zero-task workers and empty ranks.
        for(uint32_t w=0;w<kLocalWorkers;++w) {
            auto g=RouteWork(c.fragmentCount,w),v=RouteWork(c.recvChunkCount,w);
            for(uint32_t i=g.first;i<g.last;++i)gp[i+1U]=CountGroupSegments(c,info,i);
            for(uint32_t i=v.first;i<v.last;++i)cp[i+1U]=CountCvLinks(c,info,i);
        }
        for(size_t i=1;i<gp.size();++i)gp[i]+=gp[i-1U];
        for(size_t i=1;i<cp.size();++i)cp[i]+=cp[i-1U];
        c.routeSegmentCount=gp.back();c.routeLinkCount=cp.back();
        struct Prefix {
            const std::vector<uint32_t>& g;const std::vector<uint32_t>& cv;
            uint32_t Group(uint32_t i){return g[i];}
            uint32_t Link(uint32_t i){return cv[i];}
        } prefix{gp,cp};
        // Reverse worker order proves results do not depend on append timing.
        for(uint32_t n=kLocalWorkers;n>0;--n) {
            uint32_t w=n-1U;
            EmitRouteGroups(c,info,prefix,out,RouteWork(c.fragmentCount,w));
            EmitRouteReceives(c,info,prefix,out,RouteWork(c.recvChunkCount,w),MEGAMOE_URMA_ROUTE_REORDER!=0);
            EmitRouteSends(c,info,out,RouteWork(c.sendChunkCount,w));
            EmitRoutePack(c,info,out,RouteWork(c.packRoundCount,w));
            if(w==0)EmitRouteTasks(c,info,out,wave);
        }
        assert(c.computeTaskCount==out.p.computeTasks.size());
        assert(c.routeSegmentCount==out.p.segments.size());
        assert(c.routeLinkCount==out.p.chunkGroupIndices.size());
        assert(out.p.packRounds.size()==expected.packRounds.size());
        for(size_t i=0;i<out.p.packRounds.size();++i) {
            assert(out.p.packRounds[i].rows==expected.packRounds[i].rows);
            for(uint32_t peer=0;peer<kWorld;++peer) {
                const auto& a=out.p.packRounds[i].chunks[peer];const auto& b=expected.packRounds[i].chunks[peer];
                assert(a.rows==b.rows);
                // Empty virtual CVs carry no payload. The legacy host oracle
                // retains peer/cv labels; Device emission intentionally zeros
                // the whole unused slot, as RuntimeSendChunk does.
                if(a.rows)assert(a.peer==b.peer&&a.cv==b.cv&&a.localRow==b.localRow);
                else assert(a.peer==0U&&a.cv==0U&&a.localRow==0U);
                if(a.rows)assert(a.remoteRow==c.rank*c.sourceRowCapacity+i*c.cvRows);
                else assert(a.remoteRow==0);
                assert(a.fragmentBegin==b.fragmentBegin&&a.fragmentCount==b.fragmentCount&&a.reserved==b.reserved);
            }
        }

        assert(c.receiveRows==expected.config.receiveRows);
        assert(c.gmm1Tiles==expected.config.gmm1Tiles&&c.gmm2Tiles==expected.config.gmm2Tiles);
        assert(out.p.fragments.size()==expected.fragments.size());
        assert(out.p.segments.size()==expected.segments.size());
        assert(out.p.chunkGroupIndices==expected.chunkGroupIndices);
        assert(out.p.segments.size()<=RouteSegmentCapacity(ep*c.sourceRowCapacity,ep,cv));
        for(size_t i=0;i<out.p.fragments.size();++i) {
            auto a=out.p.fragments[i],b=expected.fragments[i];
            assert(a.expert==b.expert&&a.rowBegin==b.rowBegin&&a.rows==b.rows&&a.segmentBegin==b.segmentBegin&&a.segmentCount==b.segmentCount);
            assert(a.gmm1TileBegin==b.gmm1TileBegin&&a.gmm2TileBegin==b.gmm2TileBegin);
        }
        for(size_t i=0;i<out.p.segments.size();++i) {
            auto a=out.p.segments[i],b=expected.segments[i];
            assert(a.receiveRow==b.receiveRow-compactPeer[b.sourceRank]+b.sourceRank*c.sourceRowCapacity);
            assert(a.returnRow==b.returnRow&&a.tileRow==b.tileRow&&a.rows==b.rows&&a.cv==b.cv&&a.inputReadyIndex==b.inputReadyIndex&&a.sourceRank==b.sourceRank&&a.reserved==b.reserved);
        }
        assert(out.p.sendChunks.size()==expected.sendChunks.size());
        assert(out.p.recvChunks.size()==expected.recvChunks.size());
        for(size_t i=0;i<out.p.sendChunks.size();++i) {
            auto a=out.p.sendChunks[i],b=expected.sendChunks[i];
            assert(a.localRow==b.localRow&&a.rows==b.rows&&a.cv==b.cv&&a.peer==b.peer);
            assert(a.remoteRow==rank*c.sourceRowCapacity+uint64_t(a.cv)*cv);
            assert(a.fragmentBegin==b.fragmentBegin&&a.fragmentCount==b.fragmentCount&&a.reserved==b.reserved);
        }
        for(size_t i=0;i<out.p.recvChunks.size();++i) {
            auto a=out.p.recvChunks[i],b=expected.recvChunks[i];
            assert(a.localRow==b.localRow-compactPeer[b.peer]+b.peer*c.sourceRowCapacity);
            assert(a.remoteRow==b.remoteRow&&a.rows==b.rows&&a.fragmentBegin==b.fragmentBegin&&a.fragmentCount==b.fragmentCount);
            assert(a.peer==b.peer&&a.cv==b.cv&&a.reserved==b.reserved);
#if MEGAMOE_URMA_ROUTE_REORDER
            const auto [first,n]=out.ranges[i];uint64_t cursor=a.peer==rank?a.remoteRow:a.localRow;
            for(uint32_t j=0;j<n;++j) {
                const auto [src,dst,g,rows]=out.spans[first+j];
                assert(dst==cursor&&src>=out.p.fragments[g].rowBegin&&src+rows<=out.p.fragments[g].rowBegin+out.p.fragments[g].rows);
                cursor+=rows;
            }
            assert(cursor==(a.peer==rank?a.remoteRow:a.localRow)+a.rows);
            const auto oldRange=expectedReorder.ranges[i];
            assert(first==oldRange.spanBegin&&n==oldRange.spanCount);
            for(uint32_t j=0;j<n;++j) {
                const auto [src,dst,g,rows]=out.spans[first+j];
                const auto& old=expectedReorder.spans[oldRange.spanBegin+j];
                const uint64_t wantedDestination=a.peer==rank ? old.destinationRow
                    : old.destinationRow-compactPeer[a.peer]+a.peer*c.sourceRowCapacity;
                assert(src==old.sourceRow&&dst==wantedDestination&&g==old.fragmentIndex&&rows==old.rows);
            }
#endif
        }
#if MEGAMOE_URMA_ROUTE_REORDER
        assert(out.ranges.size()==expectedReorder.ranges.size());
        assert(out.spans.size()==expectedReorder.spans.size());
#else
        assert(out.ranges.empty()&&out.spans.empty());
#endif
        assert(out.p.computeBatches.size()==expected.computeBatches.size());
        for(size_t i=0;i<out.p.computeBatches.size();++i) {
            assert(out.p.computeBatches[i].fragmentBegin==expected.computeBatches[i].fragmentBegin);
            assert(out.p.computeBatches[i].fragmentCount==expected.computeBatches[i].fragmentCount);
        }
        assert(out.p.computeTasks.size()==expected.computeTasks.size());
        for(size_t i=0;i<out.p.computeTasks.size();++i) {
            assert(out.p.computeTasks[i].stage==expected.computeTasks[i].stage);
            assert(out.p.computeTasks[i].batchIndex==expected.computeTasks[i].batchIndex);
        }
        // Every source route's self dependency names its exact merged group.
        for(uint32_t e=0;e<c.localExperts;++e)for(uint32_t i=0;i<info.incoming[rank][e+1]-info.incoming[rank][e];++i) {
            uint32_t group=info.groupBase[e]+(info.sources[e][rank]+i)/256U;
            assert(out.p.fragments[group].expert==e);
        }
    }
}
int main() {
    unsigned cases=0;
    for(auto ep:{4U,8U,16U})for(auto experts:{64U,128U,384U,512U})for(auto bs:{1U,1024U,1536U,8192U,16384U})
      for(auto k:{1U,6U,8U})for(auto cv:{128U,256U,384U,1024U})for(uint32_t distribution=0;distribution<5;++distribution)for(bool shared:{false,true}) {
        // The old oracle bounds capacities to the native absolute max.
        check(ep,experts,bs,k,cv,distribution,shared);++cases;
      }
    std::cout<<"parallel route S0/S1 descriptor/oracle cases passed="<<cases<<"\n";
}
