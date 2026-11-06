"""Execute the actual AIV0 pending-tile/channel code with delayed pipe queues."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

STUB = r'''
#pragma once
#include <cassert>
#include <cstdint>
#define __aicore__
enum {PIPE_FIX,PIPE_MTE2,PIPE_MTE3};
namespace Model {void Enqueue(unsigned,unsigned,bool);void Drain(unsigned);void Copy(unsigned,unsigned);}
namespace AscendC {
enum class HardEvent {MTE2_S};using TEventID=uint32_t;
template<HardEvent> void SetFlag(TEventID){}
template<HardEvent> void WaitFlag(TEventID){Model::Drain(PIPE_MTE2);}
}
namespace MegaMoeUrma {
constexpr uint32_t kComputeTileM=256,kComputeTileN=256,kRequiredUbBytes=248*1024;
struct Arguments{};
struct Fragment{uint32_t expert=0;uint64_t rowBegin=0;uint32_t rows=256;};
struct TraceMeta{uint64_t fragment,expert,row,rows,tileN,columns;};
enum class TraceEvent{Gmm2Drain};
template<uint32_t Base,class Timer>
void ReorderGmm2Tile(const Arguments&,const Fragment&,uint32_t group,uint32_t tile,Timer&){
    static_assert(Base==120*1024);Model::Copy(group,tile);
}
}
'''
DRIVER = r'''
#include "aiv0_reorder.hpp"
#include <array>
#include <deque>
#include <vector>
#include <iostream>
#include <stdexcept>
namespace Model {
struct Op{unsigned flag;bool arrive;};
std::array<std::deque<Op>,3> q;std::array<unsigned,16> credit{};
unsigned supplied=0,produced=0,consumed=0,copied=0,payload=0;
bool compact=false;
bool Pump(){
 bool progress=false;
 for(unsigned p=0;p<3;++p){
  if(q[p].empty())continue;auto x=q[p].front();
  if(x.arrive&&x.flag==13&&produced==supplied)continue;
  if(!x.arrive&&!credit[x.flag])continue;
  if(x.arrive){assert(!credit[x.flag]);++credit[x.flag];
   if(x.flag==13){assert(compact);payload=produced++;}}
  else{--credit[x.flag];if(x.flag==13)++consumed;}
  q[p].pop_front();progress=true;
 }
 return progress;
}
void PumpAll(){while(Pump()){};}
void Enqueue(unsigned p,unsigned f,bool arrive){assert(f==13||f==15);q[p].push_back({f,arrive});PumpAll();}
void Drain(unsigned p){PumpAll();if(!q[p].empty())throw std::runtime_error("deadlock");}
void Copy(unsigned group,unsigned tile){
 assert(compact && group==copied && tile==0 && payload==copied && consumed==copied+1);
 // Delay consumer completion: no next output may overwrite this mailbox.
 for(unsigned n=0;n<13;++n){PumpAll();assert(payload==copied && produced==copied+1);}
 ++copied;
}
}
struct Timer{uint64_t TraceBegin(){return 0;}void TraceEnd(MegaMoeUrma::TraceEvent,uint64_t,const MegaMoeUrma::TraceMeta&){};};
int main(){
 using namespace MegaMoeUrma;
 unsigned cases=0;
 for(unsigned pattern=0;pattern<32;++pattern){
  Model::credit.fill(0);for(auto& q:Model::q)q.clear();
  Model::supplied=Model::produced=Model::consumed=Model::copied=0;Model::compact=false;
  std::vector<unsigned> stages{pattern%3,(pattern*7)%13,0,1,2,pattern%5};
  unsigned total=0;for(auto n:stages)total+=n;
  // AIC may enqueue all its future FIX operations long before AIV0 catches up.
  for(unsigned i=0;i<total;++i){Aiv0Gmm2Channel::Acquire();Aiv0Gmm2Channel::Publish();}
  Arguments a;Timer t;Aiv0Reorder<Timer> consumer(a,t);
  unsigned next=0;
  for(auto n:stages){
   Model::compact=true;
   for(unsigned i=0;i<n;++i){
    consumer.BeginStage(); // repeated calls must not create extra EMPTYs
    ++Model::supplied;Model::PumpAll();
    assert(Model::supplied-Model::copied<=2); // one tile ahead only
    consumer.AfterSupply(Fragment{},next++,0);
   }
   consumer.EndStage();consumer.EndStage();
   assert(Model::copied==next && Model::credit[15]==0 && Model::credit[13]==0);
   Model::compact=false;Model::PumpAll();assert(Model::produced==next);
  }
  for(auto& q:Model::q)assert(q.empty());
  assert(Model::copied==total);++cases;
 }
 std::cout<<cases<<" AIV0 mixed-stage ownership cases passed\n";
}
'''

def test_aiv0_stage_credit_and_one_tile_ahead(tmp_path):
    cxx=shlex.split(os.environ.get("CXX","c++"))
    if not shutil.which(cxx[0]):pytest.skip("requires a host C++ compiler")
    root=Path(__file__).resolve().parents[1]
    # Same production header, relocated only to substitute the hardware API.
    (tmp_path/"aiv0_reorder.hpp").write_text((root/"csrc/kernels/mega_moe/urma/aiv0_reorder.hpp").read_text(),encoding="utf-8")
    (tmp_path/"reorder.hpp").write_text(STUB,encoding="utf-8")
    (tmp_path/"c_api").mkdir()
    (tmp_path/"c_api/asc_simd.h").write_text("#pragma once\ninline void asc_sync_intra_arrive(unsigned p,unsigned f){Model::Enqueue(p,f,true);}\ninline void asc_sync_intra_wait(unsigned p,unsigned f){Model::Enqueue(p,f,false);}\n",encoding="utf-8")
    (tmp_path/"test.cpp").write_text(DRIVER,encoding="utf-8")
    exe=tmp_path/("state.exe" if os.name=="nt" else "state")
    subprocess.run(cxx+["-std=c++17","-O2","-UNDEBUG","-I",str(tmp_path),str(tmp_path/"test.cpp"),"-o",str(exe)],check=True)
    output=subprocess.run([str(exe)],capture_output=True,text=True,check=True).stdout
    assert "32 AIV0 mixed-stage ownership cases passed" in output
