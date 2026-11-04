"""Execute real BS16384 planner/capacity and generic Reduce boundaries on CPU."""
import os,shlex,shutil,subprocess
from pathlib import Path
import pytest
from test_urma_reduce_pipeline import FAKE_ASCEND

DRIVER=r"""
#include "expert_plan.hpp"
#include "shared_contract.hpp"
#include "reduce_generic.hpp"
#if __has_include("peer_reorder_plan.hpp")
#include "peer_reorder_plan.hpp"
#endif
#include <iostream>
using namespace MegaMoeUrma;using namespace AscendC;
struct Timer {uint64_t TraceBegin(){return 1;}void TraceEnd(TraceEvent,uint64_t,const TraceMeta&){};};
void Plans(){
 unsigned cases=0;
 for(unsigned world:{4U,8U,16U})for(unsigned bs:{8192U,8193U,16384U})
 for(unsigned topk:{1U,6U,8U})for(unsigned cv:{128U,256U,4096U})for(bool hot:{false,true}) {
  Config c{};c.world=world;c.bs=bs;c.topk=topk;c.hidden=5120;c.intermediate=2304;
  c.localExperts=384/world;c.cvRows=cv;c.maxPeerChunks=(bs*topk+cv-1)/cv;
  std::vector<int64_t> counts(world*384);
  for(unsigned source=0;source<world;++source) {
   if(hot)counts[source*384]=bs*topk;
   else for(unsigned expert=0;expert<384;++expert)counts[source*384+expert]=bs*topk/384+(expert<bs*topk%384);
  }
  for(unsigned rank:{0U,world-1}) {
   c.rank=rank;auto p=BuildExpertPlan(counts.data(),c,uint64_t(world)*bs*topk);
   assert(p.reduceDependencies.size()==uint64_t(bs)*topk);
   uint64_t sent=0,recv=0;
   for(auto q:p.sendChunks){assert(q.cv<c.maxPeerChunks);sent+=q.rows;}
   for(auto q:p.recvChunks){assert(q.cv<c.maxPeerChunks);recv+=q.rows;}
   assert(sent==uint64_t(bs)*topk && recv==p.config.receiveRows);
   if(hot && rank==0){assert(recv==uint64_t(world)*bs*topk);assert(p.recvChunks.size()==world*((bs*topk+cv-1)/cv));}
   for(auto d:p.reduceDependencies) {
    if(d&kReduceSelfDependency)assert((d&~kReduceSelfDependency)<GenericReduce::ReduceUbContract::maxSelfDependencies);
    else assert(d<GenericReduce::ReduceUbContract::maxRemoteDependencies);
   }
#if __has_include("peer_reorder_plan.hpp")
   auto reorder=BuildPeerReorderPlan(p.config,p.fragments,p.segments,p.recvChunks);
   assert(reorder.ranges.size()==p.recvChunks.size());
   for(unsigned peer=0;peer<world;++peer){uint64_t spans=0;for(unsigned i=0;i<p.recvChunks.size();++i)if(p.recvChunks[i].peer==peer)spans+=reorder.ranges[i].spanCount;assert(spans<=kMaxPeerReorderSpans);}
#endif
   ++cases;
  }
 }
 Config c{};c.hidden=3584;c.topk=8;c.bs=8192;c.maxPeerChunks=512;assert(UseLegacyReduce(c));
 c.maxPeerChunks=1024;assert(!UseLegacyReduce(c));c.maxPeerChunks=512;c.bs=16384;assert(!UseLegacyReduce(c));
 c.world=4;c.rank=0;c.bs=16385;c.localExperts=96;c.intermediate=1280;c.cvRows=256;
 std::vector<int64_t> zeros(4*384);bool failed=false;
 try{BuildExpertPlan(zeros.data(),c,1000000);}catch(const std::invalid_argument&){failed=true;}assert(failed);
 std::vector<bool> seen(SharedTaskCount(kMaxTokensPerRank,16384),false);
 for(unsigned worker=0;worker<28;++worker) {
  uint64_t bits[kSharedDoneWords]{};
  for(unsigned local=0;local<SharedOwnedTasks(seen.size(),worker);++local){assert(local/64<kSharedDoneWords);bits[local/64]|=uint64_t{1}<<(local%64);const auto task=worker+28*local;assert(!seen[task]);seen[task]=true;}
 }
 for(bool v:seen)assert(v);
 std::cout<<"planner cases="<<cases<<" shared tasks="<<seen.size()<<"\n";
}
template<bool FP32> void Reduction(unsigned workers,unsigned worker,unsigned priorityValue){
 priority=priorityValue;physicalHidden=256;ub.fill(0xa5);
 Arguments a{};a.config.bs=16384;a.config.hidden=256;a.config.world=16;a.config.topk=8;a.config.maxPeerChunks=1024;a.config.epoch=1;
 a.symmetric=returnedBase;a.localFlags=uintptr_t{1}<<43;a.gmm2Done=uintptr_t{1}<<44;
 const unsigned count=a.config.bs*8;
 std::vector<int64_t> inverse(count);std::vector<float> weights(count);std::vector<uint64_t> deps(count);
 for(unsigned i=0;i<count;++i){inverse[i]=(i*17)%count;weights[i]=float(int(i%7)-3)/13;deps[i]=i%2?(kReduceSelfDependency|uint64_t{8287}):uint64_t{16383};}
 std::vector<bfloat16_t> out(uint64_t(a.config.bs)*256,bfloat16_t(101.0f));std::vector<float> fpout(out.size(),101.0f);
 a.output=uintptr_t(out.data());a.sharedRoutedOutput=uintptr_t(fpout.data());a.reduceIndex=uintptr_t(inverse.data());a.weights=uintptr_t(weights.data());a.reduceDependencies=uintptr_t(deps.data());
 readFlag=[&](uintptr_t addr){
  if(addr==a.localFlags+uint64_t(kLocalFlagIncomingDone)*kFlagBytes)return uint64_t{0};
  if(addr>=a.gmm2Done)return uint64_t{1};
#if __has_include("combine_ready.hpp")
  return ~uint64_t{0};
#else
  return uint64_t{1};
#endif
 };
 beforePayload=[](uintptr_t){};executePayload={};submitOutput={};executeOutput={};onCycle={};
 Timer timer;NoReduceObserver observer;ReduceGeneric<FP32>(a,worker,workers,timer,observer);
 const unsigned begin=worker*(a.config.bs/workers)+std::min(worker,a.config.bs%workers),owned=a.config.bs/workers+(worker<a.config.bs%workers);
 for(unsigned token=begin;token<begin+owned;++token)for(unsigned col=0;col<256;++col){
  float sum=0;for(unsigned j=0;j<8;++j)sum+=float(Returned(inverse[token*8+j],col))*weights[token*8+j];
  if constexpr(FP32)assert(fpout[uint64_t(token)*256+col]==sum);
  else assert(out[uint64_t(token)*256+col].bits==bfloat16_t(sum).bits);
 }
 if(begin){assert(fpout[uint64_t(begin)*256-1]==101.0f);assert(out[uint64_t(begin)*256-1].bits==bfloat16_t(101.0f).bits);}
 if(begin+owned<a.config.bs){assert(fpout[uint64_t(begin+owned)*256]==101.0f);assert(out[uint64_t(begin+owned)*256].bits==bfloat16_t(101.0f).bits);}
 for(auto&q:queues)assert(q.empty());for(auto&kind:events)for(bool bit:kind)assert(!bit);
}
int main(){
 static_assert(kMaxPeerChunks==1024 && kSharedDoneWords==5);
 static_assert(GenericReduce::ReduceUbContract::end+2*2048*4<=kRequiredUbBytes);
 Plans();
 for(unsigned workers:{41U,49U,53U,56U})for(unsigned worker:{0U,workers-1})for(unsigned pr:{0U,1U,2U}){Reduction<false>(workers,worker,pr);Reduction<true>(workers,worker,pr);}
 std::cout<<"PASS BS16384 BF16/FP32 output, metadata tails, dependency16383/self8287, DMA event reuse\n";
}
"""

def test_bs16384_capacity_and_reduction(tmp_path):
    command=shlex.split(os.environ.get('CXX','c++'))
    if not command or not shutil.which(command[0]):pytest.skip('Host C++ compiler required')
    fake=FAKE_ASCEND
    fake=fake.replace('inline bfloat16_t Returned','inline uint32_t physicalHidden=3584;\ninline bfloat16_t Returned')
    fake=fake.replace('/ 3584U','/ physicalHidden').replace('% 3584U','% physicalHidden')
    if 'struct DataCopyExtParams' not in fake:
        fake+=r"""
namespace AscendC {
struct DataCopyExtParams{uint16_t count;uint32_t bytes,src,dst,rsv;};
template<class T>struct DataCopyPadExtParams{bool pad;uint8_t left,right;T value;};
template<class T>void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){assert(p.count==1 && p.bytes/sizeof(T)<=dst.count);Enqueue(Mte2,[=]{std::copy_n(src.data,p.bytes/sizeof(T),dst.Data());return true;});}
}
"""
    fake+=r"""
namespace AscendC {
template<class T>void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,uint32_t n){Enqueue(Vector,[=]{std::copy_n(src.Data(),n,dst.Data());return true;});}
}
"""
    (tmp_path/'kernel_operator.h').write_text(fake,encoding='utf-8')
    source=tmp_path/'model.cpp';source.write_text(DRIVER,encoding='utf-8')
    binary=tmp_path/('model.exe' if os.name=='nt' else 'model')
    headers=Path(__file__).resolve().parents[1]/'csrc/kernels/mega_moe/urma'
    subprocess.run(command+['-std=c++17','-O2','-UNDEBUG','-ffp-contract=off','-I'+str(tmp_path),'-I'+str(headers),str(source),'-o',str(binary)],check=True,timeout=120)
    subprocess.run([str(binary)],check=True,timeout=180)
