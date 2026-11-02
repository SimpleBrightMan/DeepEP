"""Run the production PeerCombine class against CPU transport/ready doubles."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def source_text(root):
    kernel = (root / "experiments/megamoe/csrc/kernels/mega_moe/urma/kernel.hpp").read_text(encoding="utf-8")
    helpers = kernel[kernel.index("__aicore__ inline Chunk LoadChunk"):kernel.index("__aicore__ inline void TraceLocalSync")]
    progress = kernel[kernel.index("__aicore__ inline bool IsChunkComputed"):kernel.index("// One owner per peer/QP handles both phases")]
    return r'''
#include "compute_contract.hpp"
#include "communication_schedule.hpp"
#include "combine_ready_layout.hpp"
#include <array>
#include <vector>
#include <cassert>
#include <cstdint>
#define __aicore__
#define __gm__
using bfloat16_t=uint16_t;
namespace AscendC {
enum class TPosition {VECCALC};
inline std::array<uint8_t,196608> ub{};
template<class T> struct LocalTensor {
    T* p; LocalTensor(TPosition,uint32_t offset,uint32_t count):p(reinterpret_cast<T*>(ub.data()+offset)){
        assert(offset+count*sizeof(T)<=ub.size());
    }
    T GetValue(uint32_t i){return p[i];} void SetValue(uint32_t i,T v){p[i]=v;}
};
template<class T>T ReadGmByPassDCache(T* p){return *p;}
inline uint64_t clock=0;
inline uint64_t GetSystemCycle(){return ++clock;}
}
namespace MegaMoeUrma {
using namespace AscendC;
constexpr uint32_t kPackUbBase=131072;
enum class Phase {Combine}; enum class CoarsePhase {Combine};
struct Timer {
    std::vector<TraceEvent> events;
    uint64_t TraceBegin(){return GetSystemCycle();}
    void TraceEnd(TraceEvent event,uint64_t,const TraceMeta&){events.push_back(event);}
    void Begin(Phase){} void End(Phase,uint64_t){}
    void CoarseBegin(CoarsePhase){} void CoarseEnd(CoarsePhase){}
};
inline int32_t* ComputeFlag(uint64_t p,uint32_t index){return reinterpret_cast<int32_t*>(p+index*kFlagBytes);}
inline uint32_t PeerForOwner(uint32_t rank,uint32_t owner){return owner<rank?owner:owner+1;}
inline void aclshmemi_kernel_abort(const char*){assert(false);}
struct Sent {uint64_t offset;uint32_t bytes,cv,count;};
inline std::vector<Sent> sent;
inline uint32_t quiets=0,finished=0;
inline void Put(uint8_t*,uint64_t offset,uint8_t*,uint32_t bytes,uint32_t peer){assert(peer==1);sent.push_back({offset,bytes,999,0});}
inline void PublishCombineRange(uint8_t*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t cv,uint32_t count){assert(count>=1&&count<=2);sent.back().cv=cv;sent.back().count=count;}
inline void PublishOrdered(uint8_t*,uint64_t,uint32_t,uint64_t){++finished;}
inline void Quiet(uint32_t){++quiets;}
inline bool ObserveIncomingCompletion(const Arguments&,uint32_t,bool&,bool&){return true;}
inline uint64_t PeerPhaseSlot(const Arguments&,uint32_t,uint32_t){return 0;}
''' + helpers + progress + r'''
}
int main(){
    using namespace MegaMoeUrma;
    for(uint32_t cv : {128U,256U,1024U,4096U}) {
        sent.clear();quiets=finished=0;
        Arguments a{};a.config.rank=0;a.config.world=4;a.config.hidden=3584;a.config.maxPeerChunks=128;
        a.config.cvRows=cv;a.config.recvChunkCount=5;a.config.returnYOffset=4096;
        a.gmm2Output=0x100000000ULL;
        std::array<Fragment,5> groups{};for(auto& g:groups)g.rows=256;
        a.fragments=uintptr_t(groups.data());
        std::array<std::array<int32_t,32>,5> ready{};a.gmm2Done=uintptr_t(ready.data());
        std::array<uint32_t,5> deps{0,1,2,3,4};a.chunkGroupIndices=uintptr_t(deps.data());
        std::array<Chunk,5> blocks{};
        for(uint32_t i=0;i<5;++i)blocks[i]={1,i,uint64_t(i)*cv,uint64_t(i)*cv,i==4?87U:cv,i,1,0};
        a.recvChunks=uintptr_t(blocks.data());
        Timer timer;PeerCombine c(a,0,timer);
        ready[0][0]=13;ready[1][0]=14;ready[2][0]=14;ready[3][0]=13;
        // Adjacent ready CV1/2 must not steal halves from different CV pairs.
        assert(!c.Scan() && sent.empty() && quiets==0);
        ready[3][0]=14;uint64_t wait=timer.TraceBegin();TraceMeta meta;
        assert(c.Scan(&wait,&meta,TraceEvent::WaitDispatch));
        assert(timer.events[0]==TraceEvent::WaitDispatch && timer.events[1]==TraceEvent::CombinePut);
        assert(sent.size()==1 && sent[0].cv==2 && sent[0].count==2 && sent[0].bytes==2*cv*3584*2 && quiets==0);
        assert(!c.Scan() && sent.size()==1);
        ready[0][0]=14;assert(c.Scan());assert(sent[1].cv==0 && sent[1].count==2);
        ready[4][0]=14;assert(c.Scan());assert(sent[2].cv==4 && sent[2].bytes==87*3584*2);
        assert(finished==0 && quiets==0);
        c.Finish();assert(finished==1 && quiets==1);
        assert(sent[0].offset==4096+2*cv*3584*2 && sent[2].offset==4096+4*cv*3584*2);
        for(uint32_t total=0;total<=5;++total) {
            sent.clear();quiets=finished=0;a.config.recvChunkCount=total;
            for(auto& r:ready)r[0]=14;
            for(uint32_t i=0;i<total;++i)blocks[i].rows=i+1==total?87U:cv;
            PeerCombine tail(a,0,timer);tail.Scan();tail.Finish();
            assert(sent.size()==(total+1)/2 && quiets==1 && finished==1);
            uint32_t seen=0;
            for(auto p:sent) {
                assert(p.cv==seen && p.count==std::min(2U,total-seen));
                uint32_t n=0;for(uint32_t j=0;j<p.count;++j)n+=blocks[seen+j].rows;
                assert(p.bytes==n*3584*2);seen+=p.count;
            }
            assert(seen==total);
        }
    }
}

'''


def test_original_ready_cv_pairs_can_be_sent_before_dispatch_finishes(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "progress.cpp"
    source.write_text(source_text(root), encoding="utf-8")
    binary = tmp_path / ("progress.exe" if os.name == "nt" else "progress")
    build = subprocess.run(command+["-std=c++17","-O1","-UNDEBUG","-I"+str(root / "experiments/megamoe/csrc/kernels/mega_moe/urma"),str(source),"-o",str(binary)], capture_output=True,text=True,timeout=120)
    assert build.returncode==0, build.stdout+build.stderr
    run = subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0, run.stdout+run.stderr
