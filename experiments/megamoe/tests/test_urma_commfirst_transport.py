"""Actual transport helpers under delayed RO/SO and MTE models, not NPU validation."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_bitmap_so_and_immutable_async_mte_publication(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "kernel_operator.h").write_text(r"""
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __gm__
enum pipe_t { PIPE_S };
namespace AscendC {
using TEventID = uint32_t;
enum class HardEvent { S_MTE3, MTE3_S };
template<class T> struct LocalTensor { T* data; void SetValue(uint32_t i,T v){data[i]=v;} };
template<class T> struct GlobalTensor { T* data; void SetGlobalBuffer(T* p){data=p;} };
struct Copy { void* dst; const void* src; uint32_t bytes; };
inline std::vector<Copy> copies;
inline uint32_t drains=0;
template<HardEvent> void SetFlag(TEventID) {}
template<HardEvent E> void WaitFlag(TEventID) {
    if constexpr(E == HardEvent::MTE3_S) {
        ++drains;
        for(auto c:copies) std::memcpy(c.dst,c.src,c.bytes);
        copies.clear();
    }
}
template<class T> void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,uint32_t n) {
    copies.push_back({dst.data,src.data,static_cast<uint32_t>(n*sizeof(T))});
}
template<class T> T ReadGmByPassDCache(T* p) {return *p;}
}
""", encoding="utf-8")
    (tmp_path / "shmem.h").write_text(r"""
#pragma once
#include <type_traits>
struct aclshmemx_udma_op_config_t {uint32_t cqe,se,fence,odr;};
inline constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_RO{1,0,0,5};
inline constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_SO{1,0,0,6};
alignas(128) inline uint8_t windows[4][131072]{};
struct Op {uint8_t* dst;uint8_t* src;uint32_t bytes,peer;uint64_t add;};
inline std::vector<Op> pending;
inline uint32_t quiets=0;
inline void* aclshmem_ptr(void* address,uint32_t peer) {
    auto offset=reinterpret_cast<uint8_t*>(address)-windows[0];
    assert(offset>=0 && offset<131072 && peer<4);
    return windows[peer]+offset;
}
template<class T,pipe_t P,const aclshmemx_udma_op_config_t& C>
void aclshmemx_udma_qp_put_nbi(T* dst,T* src,T* scratch,uint32_t n,uint32_t peer,uint32_t qp,uint32_t event) {
    static_assert(P==PIPE_S && C.odr==5);
    assert(qp==0 && scratch==nullptr && event==0);
    pending.push_back({static_cast<uint8_t*>(aclshmem_ptr(dst,peer)),src,n,peer,0});
}
template<class T,const aclshmemx_udma_op_config_t& C>
void aclshmemx_udma_atomic_add(T* dst,T value,int32_t peer) {
    static_assert(C.odr==6 && std::is_same_v<T,uint64_t>);
    pending.push_back({static_cast<uint8_t*>(aclshmem_ptr(dst,peer)),nullptr,0,uint32_t(peer),value});
}
inline bool Progress(uint32_t peer) {
    for(auto i=pending.begin();i!=pending.end();++i) if(i->peer==peer) {
        if(i->src)std::memcpy(i->dst,i->src,i->bytes);
        else *reinterpret_cast<uint64_t*>(i->dst)+=i->add;
        pending.erase(i);return true;
    }
    return false;
}
inline void aclshmemx_udma_qp_quiet(uint32_t peer,uint32_t qp) {
    assert(qp==0);++quiets;while(Progress(peer)){}
}
template<class...T>void aclshmemi_kernel_abort(const char*,T...){throw std::runtime_error("abort");}
""", encoding="utf-8")
    source = tmp_path / "commfirst.cpp"
    source.write_text(r"""
#include "transport.hpp"
#include <algorithm>
#include <random>
int main() {
    using namespace MegaMoeUrma;
    static_assert(kDispatchQp==0 && kCombineQp==0);
    static_assert(CombineReadyMask(0,64)==~uint64_t{0});
    static_assert(CombineReadyMask(63,1)==uint64_t{1}<<63);
    static_assert(!CombineRangeFitsWord(63,2) && !CombineRangeFitsWord(0,0));
    uint8_t data[64];std::memset(data,71,sizeof(data));
    uint64_t storage[4]{};
    AsyncPublishBuffer publish({storage},0);
    publish.Init(19);
    publish.Enqueue(windows[0],128,1);
    assert(AscendC::drains==0 && *reinterpret_cast<uint64_t*>(windows[1]+128)==0);
    Put(windows[0],0,data,sizeof(data),1); // next payload can be issued first
    publish.Drain();
    assert(AscendC::drains==1 && *reinterpret_cast<uint64_t*>(windows[1]+128)==19);
    assert(windows[1][0]==0 && pending.size()==1); // MTE drain is not UDMA quiet
    publish.Drain();assert(AscendC::drains==1);
    Quiet(1);assert(windows[1][0]==71);

    // Dispatch: merged X/scale writes precede every per-CV SO epoch.
    // No quiet is needed between submissions. Each invocation resets slots
    // before publishing start; the final quiet retires all in-flight writes.
    for(uint32_t epoch: {1U,2U,17U}) {
        assert(pending.empty());
        std::memset(windows[1]+2048,0,256);
        auto* cv0=reinterpret_cast<uint64_t*>(windows[1]+2048);
        auto* cv1=reinterpret_cast<uint64_t*>(windows[1]+2176);
        const auto previousQuiets=quiets;
        Put(windows[0],512,data,32,1);
        Put(windows[0],544,data+32,32,1);
        PublishOrdered(windows[0],2048,1,epoch);
        PublishOrdered(windows[0],2176,1,epoch);
        assert(quiets==previousQuiets && *cv0==0 && *cv1==0);
        assert(Progress(1) && *cv0==0 && *cv1==0);
        assert(Progress(1) && *cv0==0 && *cv1==0);
        assert(Progress(1) && *cv0==epoch && *cv1==0);
        assert(windows[1][512]==71 && windows[1][544]==71);
        Quiet(1);
        assert(*cv0==epoch && *cv1==epoch && pending.empty());
    }

    constexpr uint64_t readyOffset=4096;
    Arguments receive{};receive.symmetric=uintptr_t(windows[1]);
    receive.config.combineReadyOffset=readyOffset;receive.config.maxPeerChunks=128;
    for(uint32_t epoch: {2U,19U,100U}) {
        // Receiver clears before its start; sender must have completed its
        // previous final quiet and observe this start before submitting again.
        assert(pending.empty());
        std::memset(windows[1]+readyOffset,0,4*128*128);
        assert(!CombineCvReady(receive,62) && !CombineCvReady(receive,64));
        Put(windows[0],256,data,sizeof(data),1);
        PublishCombineRange(windows[0],readyOffset,128,0,1,62,2);
        assert(!CombineCvReady(receive,62));
        assert(Progress(1) && !CombineCvReady(receive,62)); // payload alone is not ready
        assert(Progress(1) && CombineCvReady(receive,62) && CombineCvReady(receive,63));
        assert(!CombineCvReady(receive,61) && !CombineCvReady(receive,64));
        PublishCombineRange(windows[0],readyOffset,128,0,1,64,3);
        PublishCombineRange(windows[0],readyOffset,128,0,1,0,2); // out-of-order, disjoint masks
        PublishOrdered(windows[0],1024,1,epoch);
        Quiet(1);
        for(uint32_t cv: {0U,1U,62U,63U,64U,65U,66U}) assert(CombineCvReady(receive,cv));
        for(uint32_t cv: {2U,61U,67U,127U}) assert(!CombineCvReady(receive,cv));
        *reinterpret_cast<uint64_t*>(windows[1]+1024)=0; // finished reset before next start
    }
    bool refused=false;
    try {PublishCombineRange(windows[0],readyOffset,128,0,1,63,2);}catch(const std::runtime_error&){refused=true;}
    assert(refused);
    // A complete word may arrive in any disjoint order: no carry into a hole.
    std::memset(windows[1]+readyOffset,0,128);
    std::vector<uint32_t> bits(64);for(uint32_t i=0;i<64;++i)bits[i]=i;
    std::mt19937 rng(12);std::shuffle(bits.begin(),bits.end(),rng);
    uint64_t expected=0;
    for(auto bit:bits){PublishCombineRange(windows[0],readyOffset,128,0,1,bit,1);Progress(1);expected|=uint64_t{1}<<bit;
        assert(*reinterpret_cast<uint64_t*>(windows[1]+readyOffset)==expected);}
}
""", encoding="utf-8")
    binary = tmp_path / ("commfirst.exe" if os.name == "nt" else "commfirst")
    result = subprocess.run(command + ["-std=c++17", "-O1", "-I" + str(tmp_path),
        "-I" + str(root / "experiments/megamoe/csrc/kernels/mega_moe/urma"), str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
