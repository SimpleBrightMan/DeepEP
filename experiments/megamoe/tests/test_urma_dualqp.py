"""Exercise production lane ownership and selected-QP SO transport on CPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

def test_dual_peer_ownership_and_transport(tmp_path):
    cxx=shlex.split(os.environ.get('CXX','c++'))
    if not shutil.which(cxx[0]):pytest.skip('C++ compiler required')
    inc=Path(__file__).resolve().parents[1]/'csrc/kernels/mega_moe/urma'
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#include <cstring>
#define __aicore__
#define __gm__
#define __ubuf__
enum pipe_t {PIPE_S};
namespace AscendC {
enum class HardEvent {S_MTE3,MTE3_S};
enum TEventID {EVENT_ID0};
template<HardEvent>void SetFlag(TEventID){}
template<HardEvent>void WaitFlag(TEventID){}
template<class T>struct GlobalTensor {T* p;void SetGlobalBuffer(T* x){p=x;}};
template<class T>struct LocalTensor {T p[4];void SetValue(int i,T x){p[i]=x;}};
template<class T>void DataCopy(GlobalTensor<T>& g,LocalTensor<T>& l,int n){std::memcpy(g.p,l.p,n*sizeof(T));}
template<class T>T ReadGmByPassDCache(T* p){return *p;}
template<class T>void AtomicAdd(T* p,T x){*p+=x;}
inline uint64_t GetSystemCycle(){static uint64_t tick=0;return ++tick;}
}
''',encoding='utf-8')
    (tmp_path/'shmem.h').write_text(r'''
#pragma once
#include <vector>
#include <stdexcept>
struct aclshmemx_udma_op_config_t {uint32_t cqe,se,fence,odr;};
inline constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_RO{1,0,0,5},ACLSHMEMX_UDMA_OP_CONFIG_SO{1,0,0,6};
struct Op {uint32_t peer,qp,odr;uint64_t bytes,value;};
inline std::vector<Op> ops;
template<class T>T* aclshmem_ptr(T* p,uint32_t){return p;}
template<class... T>void aclshmemi_kernel_abort(const char*,T...){throw std::runtime_error("abort");}
template<class T,pipe_t,const aclshmemx_udma_op_config_t& C>void aclshmemx_udma_qp_put_nbi(T*,T*,void*,uint32_t n,uint32_t pe,uint32_t qp,uint32_t){ops.push_back({pe,qp,C.odr,n*sizeof(T),0});}
template<class T,const aclshmemx_udma_op_config_t& C>void aclshmemx_udma_atomic_add(T*,T v,uint32_t pe){ops.push_back({pe,0,C.odr,sizeof(T),v});}
enum class aclshmemi_udma_opcode_t {UDMA_OP_WRITE_WITH_REDUCE,UDMA_OPCODE_FAA};
template<class T,aclshmemi_udma_opcode_t>struct aclshmemi_udma_params_t {T value,cond;};
template<class T,aclshmemi_udma_opcode_t Op,const aclshmemx_udma_op_config_t& C>
void aclshmemi_udma_post_send(uint8_t*,void*,uint32_t pe,uint32_t qp,uint64_t n,const aclshmemi_udma_params_t<T,aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE>& p){
    static_assert(Op==aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA);ops.push_back({pe,qp,C.odr,n,p.value});
}
inline void aclshmemx_udma_qp_quiet(uint32_t pe,uint32_t qp){ops.push_back({pe,qp,99,0,0});}
''',encoding='utf-8')
    src=tmp_path/'test.cpp';src.write_text(r'''
#include "transport.hpp"
#include "compute_contract.hpp"
#include "tail_completion.hpp"
#include <cassert>
#include <set>
using namespace MegaMoeUrma;
int main(){
 for(auto world:{4U,8U,16U}){
  const auto lanes=PeerLanes(world),workers=CommunicationWorkers(world);
  assert(workers==(world-1)*lanes);
  assert(ReduceWorkers(world)==(world==16?41:56));
  for(uint32_t rank=0;rank<world;++rank){
   std::set<uint32_t> pairs;
   for(uint32_t owner=0;owner<workers;++owner){auto peer=PeerForOwner(rank,owner,world);assert(peer!=rank&&peer<world);pairs.insert(peer*2+OwnerQp(owner,world));}
   assert(pairs.size()==workers);
  }
  for(uint32_t total=0;total<=1024;++total){
   std::vector<uint32_t> seen(total,0);
   for(uint32_t lane=0;lane<lanes;++lane){uint32_t count=0,last=0;
    for(uint32_t cv=lane;cv<total;cv+=lanes){++seen[cv];++count;last=cv;}
    assert(count==PeerOwnedCvs(total,lane,lanes));if(count)assert(last==PeerLastCv(total,lane,lanes));
   }
   for(auto n:seen)assert(n==1);
  }
  for(auto rows:{1U,55U,56U,57U,1024U,8192U,16384U}){
   std::vector<uint32_t> seen(rows,0);std::set<uint32_t> ownerIds;
   for(uint32_t b=0;b<28;++b)for(uint32_t v=0;v<2;++v){auto w=ReduceWorkerIndex(b,v,world);
    if(w>=ReduceWorkers(world))continue;assert(ownerIds.insert(w).second);
    auto begin=ReduceRowBegin(rows,w,world),n=ReduceOwnedRows(rows,w,world);
    for(uint64_t i=begin;i<begin+n;++i)++seen[i];
   }
   assert(ownerIds.size()==ReduceWorkers(world));for(auto n:seen)assert(n==1);
  }
 }
 uint8_t memory[4096]{},source[256]{};
 for(uint32_t qp=0;qp<2;++qp){Put(memory,0,source,256,2,qp);PublishOrdered(memory,1024,2,7,qp);Quiet(2,qp);}
 assert(ops.size()==6);
 for(uint32_t qp=0;qp<2;++qp){auto i=qp*3;assert(ops[i].qp==qp&&ops[i].odr==5);assert(ops[i+1].qp==qp&&ops[i+1].odr==6&&ops[i+1].value==7);assert(ops[i+2].qp==qp&&ops[i+2].odr==99);}
 // Rank1: owner0/1 -> peer0. One lane cannot finish the source rank.
 std::vector<uint64_t> sym(4*4*128/8,0),local(16*128/8,0);Arguments a{};a.config.world=4;a.config.rank=1;a.config.epoch=9;
 a.symmetric=reinterpret_cast<uint64_t>(sym.data());a.localFlags=reinterpret_cast<uint64_t>(local.data());
 bool leader=false,self=false,secondary=false,secondarySelf=false;
 sym[(4+0)*16]=9;
 assert(!ObserveIncomingCompletion(a,0,leader,self));assert(*LocalCompletionFlag(a,kLocalFlagIncomingDone)==0);
 assert(ObserveIncomingCompletion(a,1,secondary,secondarySelf));assert(*LocalCompletionFlag(a,kLocalFlagIncomingDone)==0);
 sym[(4+0)*16+8]=9;
 assert(!ObserveIncomingCompletion(a,0,leader,self));assert(*LocalCompletionFlag(a,kLocalFlagIncomingDone)==1);
 *LocalCompletionFlag(a,kLocalFlagAicDone)=28;*LocalCompletionFlag(a,kLocalFlagReorderDone)=6;
 assert(ObserveIncomingCompletion(a,0,leader,self));assert(*LocalCompletionFlag(a,kLocalFlagIncomingDone)==2);
 ObserveIncomingCompletion(a,0,leader,self);assert(*LocalCompletionFlag(a,kLocalFlagIncomingDone)==2);
}
''',encoding='utf-8')
    exe=tmp_path/('test.exe' if os.name=='nt' else 'test')
    subprocess.run([*cxx,'-std=c++17','-UNDEBUG','-I',str(tmp_path),'-I',str(inc),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_dual_raw_timer_keeps_both_lanes_and_one_receive_observer():
    import numpy as np
    from deep_ep_experimental.megamoe.urma_timer import communication_bandwidth
    raw=np.zeros((28,3,8,16),dtype=np.int64)
    for owner in range(6):
        peer=owner//2+1;lane=owner%2
        for phase in (1,5):
            raw[owner,2,phase,:8]=[100+owner,200+owner,100,1000,1,1,1,500+owner if phase==5 else 0]
            raw[owner,2,phase,10:13]=[peer,lane,2]
        if lane==0:raw[owner,2,1,5:10]=[2,1,400+peer,2000,256]
    result=communication_bandwidth(raw,0,world=4)
    d=next(r for r in result['classes'] if r['phase']=='Dispatch' and r['communication_class']=='INTRA')
    c=next(r for r in result['classes'] if r['phase']=='Combine' and r['communication_class']=='INTRA')
    assert d['payload_bytes']==6000 and d['envelope_ticks']==303
    assert d['receive_peers']==[1,2,3] and d['received_payload_bytes']==6000
    assert c['payload_bytes']==6000 and c['envelope_ticks']==405
