"""Real diagnostic observer/timer with CPU CQ mocks; no hardware latency claim."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('coarse',[False,True])
def test_fixed_cq_watermark_observer(tmp_path,enabled,coarse):
    cxx=shlex.split(os.environ.get('CXX','c++'))
    if not shutil.which(cxx[0]):pytest.skip('C++ compiler required')
    root=Path(__file__).resolve().parents[3]
    inc=root/'experiments/megamoe/csrc/kernels/mega_moe/urma'
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
namespace AscendC {
constexpr int AIC=0,AIV=1;
inline uint64_t cycles=100,flushes=0,reads=0;
inline uint64_t GetSystemCycle(){return cycles++;}
inline uint32_t GetBlockIdx(){return 0;}
inline uint32_t GetTaskRation(){return 2;}
inline uint32_t GetSubBlockIdx(){return 1;}
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T>struct GlobalTensor {void SetGlobalBuffer(T*){}};
template<class T,CacheLine,DcciDst>void DataCacheCleanAndInvalid(GlobalTensor<T>&){++flushes;}
template<class T>T ReadGmByPassDCache(T* p){++reads;return *p;}
}
constexpr int g_coreType=AscendC::AIV;
''',encoding='utf-8')
    # Only replace dependencies with mocks; execute the checked-in observer body.
    observer=(inc/'dispatch_completion.hpp').read_text(encoding='utf-8')
    observer=observer.replace('#include "timer.hpp"','').replace('#include "transport.hpp"','')
    (tmp_path/'observer.hpp').write_text(observer,encoding='utf-8')
    cpp=r'''
#include "timer.hpp"
#include <cassert>
#include <vector>
#include <stdexcept>
constexpr bool ACLSHMEM_RELAY_SUPPORTED=false;
namespace shm {constexpr uint32_t UDMA_CQ_DEPTH_DEFAULT=4;}
struct aclshmemi_udma_queue_state_t {uint32_t sq_head,sq_tail,cqe_cnt,cq_head,cq_tail;};
struct aclshmemi_udma_cq_ctx_t {uint64_t buf_addr;uint32_t cqe_size;uint64_t state_addr;};
struct Info {uint32_t qp_num=1;};
struct Table {uint64_t scq_ptr;};
inline Info info;inline Table table;inline uint32_t fetches=0;
Info* aclshmemi_udma_qp_info_fetch(){++fetches;return &info;}
Table* aclshmemi_udma_active_table(Info*){return &table;}
template<class... T>void aclshmemi_kernel_abort(const char*,T...){throw std::runtime_error("CQ error");}
namespace MegaMoeUrma {constexpr uint32_t kCommunicationQp=0;}
#include "observer.hpp"
using namespace MegaMoeUrma;
int main(){
    const uint32_t capacity=8,stride=kTraceHeaderElements+capacity*kTraceRecordElements;
    std::vector<uint64_t> trace(kTimerCores*stride,0),aggregate(kTimerElements,0);
    Timer timer;timer.Init(reinterpret_cast<uint64_t>(aggregate.data()),reinterpret_cast<uint64_t>(trace.data()),capacity,COARSE);
    timer.Begin(Dispatch);timer.End(Dispatch,1024,true);
    DispatchCompletionObserver observer;observer.Init(timer);
    uint32_t cqe[4]{};
    aclshmemi_udma_queue_state_t state{31,9,11,0,9};
    aclshmemi_udma_cq_ctx_t ctx[2]{};
    ctx[1]={reinterpret_cast<uint64_t>(cqe),4,reinterpret_cast<uint64_t>(&state)};
    table.scq_ptr=reinterpret_cast<uint64_t>(ctx);
    observer.Arm(1,7); // target 11, ordinal10, old owner0
    const auto firstClock=AscendC::cycles;
    for(int i=0;i<3;++i)observer.Poll();
    assert(AscendC::cycles==firstClock);
    state.cqe_cnt=100; // Combine does NOT move the fixed Dispatch target.
    cqe[2]=4;observer.Poll();
#ifdef DEEPEP_MEGAMOE_TIMER
    assert(fetches==1 && AscendC::reads==4);
    assert(state.cq_tail==9 && state.sq_tail==9); // no credit consumption
    assert(AscendC::cycles==firstClock+1);
    assert(aggregate[(2*kTimerPhases+Dispatch)*kTimerRecordElements+7]==0);
    for(auto value:trace)assert(value==0); // no trace store at observation
    observer.Poll();assert(AscendC::reads==4); // one-shot
    // A later envelope may contain the observed point. Export at the end.
    timer.WriteTrace(TraceEvent::Reduce,firstClock-1,firstClock+10,TraceMeta{});
    timer.FinishTrace();
    if(!COARSE)assert(aggregate[(2*kTimerPhases+Dispatch)*kTimerRecordElements+7]==firstClock);
    const auto* header=trace.data()+2*stride;
    const auto* point=header+kTraceHeaderElements+kTraceRecordElements;
    assert(header[2]==2 && point[2]==43 && point[0]==firstClock && point[1]==firstClock && point[3]==1 && point[4]==7);

    // Normal SHMEM CQ retirement can pass/reuse the observed entry.
    DispatchCompletionObserver retired;retired.Init(timer);state.cqe_cnt=20;state.cq_tail=23;
    retired.Arm(1,0);const auto reads=AscendC::reads;retired.Poll();assert(AscendC::reads==reads);
    // uint32 counter and CQ ring wrap, target0 is valid after wrap.
    DispatchCompletionObserver wrapped;wrapped.Init(timer);state.cqe_cnt=0;state.cq_tail=UINT32_MAX-1;
    wrapped.Arm(1,0);cqe[3]=4;const auto tick=AscendC::cycles;
    wrapped.Poll();assert(AscendC::cycles==tick);
    cqe[3]=0;wrapped.Poll();assert(AscendC::cycles==tick+1);
    DispatchCompletionObserver error;error.Init(timer);state.cqe_cnt=1;state.cq_tail=0;
    error.Arm(1,0);cqe[0]=4|(1U<<24);bool caught=false;
    try{error.Poll();}catch(const std::runtime_error&){caught=true;}
    assert(caught);
#else
    assert(fetches==0 && AscendC::reads==0 && AscendC::cycles==100);
    for(auto value:trace)assert(value==0);
    for(auto value:aggregate)assert(value==0);
#endif
    Timer off;off.Init(0);DispatchCompletionObserver disabled;disabled.Init(off);
    const auto f=fetches;const auto disabledReads=AscendC::reads;const auto cycles=AscendC::cycles;
    disabled.Arm(1,1);disabled.Poll();
    assert(fetches==f && AscendC::reads==disabledReads && AscendC::cycles==cycles);
}
'''
    src=tmp_path/'test.cpp';src.write_text(cpp,encoding='utf-8')
    exe=tmp_path/('test.exe' if os.name=='nt' else 'test')
    defines=['-DCOARSE='+str(int(coarse))]+(['-DDEEPEP_MEGAMOE_TIMER'] if enabled else [])
    subprocess.run([*cxx,'-std=c++17','-UNDEBUG',*defines,'-I',str(tmp_path),'-I',str(inc),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_deferred_completion_point_can_overlap_reduce_but_not_hide_bad_scopes():
    import numpy as np
    from deep_ep_experimental.megamoe import urma_trace as t
    capacity=4;stride=t.HEADER_WORDS+capacity*t.RECORD_WORDS
    raw=np.zeros((t.ENGINE_COUNT,stride),dtype=np.uint64)
    for i in range(t.ENGINE_COUNT):raw[i,:5]=[1,capacity,0,0,i]
    raw[2,2]=2
    for ordinal,(start,end,event) in enumerate([(100,200,25),(150,150,43)]):
        row=raw[2,t.HEADER_WORDS+ordinal*t.RECORD_WORDS:t.HEADER_WORDS+(ordinal+1)*t.RECORD_WORDS]
        row[:3]=[start,end,event];row[3:12]=t.UNKNOWN;row[3]=1
    doc=t.decode_trace(raw.reshape(-1),rank=0,clock_mhz=1000)
    assert any(e.get('args',{}).get('event_id')==43 for e in doc['traceEvents'])
    raw[2,t.HEADER_WORDS+t.RECORD_WORDS+1]=151
    with pytest.raises(ValueError,match='overlapping|out-of-order'):
        t.decode_trace(raw.reshape(-1),rank=0,clock_mhz=1000)
