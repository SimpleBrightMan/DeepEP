"""Execute the S1 FP32 reducer and six-slot sum with asynchronous engine queues."""
import os, shlex, shutil, subprocess
from pathlib import Path
import pytest
from test_urma_reduce_pipeline import FAKE_ASCEND

DRIVER = r"""
#include "reduce_generic.hpp"
#include "shared_add.hpp"
#include <iostream>
using namespace AscendC;
using namespace MegaMoeUrma;
struct Timer {
 uint64_t TraceBegin(){return 1;}
 void TraceEnd(TraceEvent,uint64_t,const TraceMeta& = {}){}
};
int main(){
 unsigned cases=0,differentRoundings=0;
 for(unsigned h:{256U,3584U,4096U,8192U})
 for(unsigned bs:{1U,73U,1024U})for(unsigned pr:{0U,1U,2U}){
  physicalHidden=h;priority=pr;
  Arguments a{};a.config.bs=bs;a.config.hidden=h;a.config.topk=8;a.config.world=16;
  a.symmetric=returnedBase;a.localFlags=uintptr_t{1}<<43;
  std::vector<int64_t> inverse(bs*8);std::vector<float> weights(bs*8);
  for(unsigned i=0;i<bs*8;++i){inverse[i]=(i*17U)%(bs*8);weights[i]=float(int(i%7)-3)/13;}
  std::vector<float> route(uint64_t(bs)*h,12345.0f);
  std::vector<bfloat16_t> shared(route.size()),out(route.size());
  for(size_t i=0;i<route.size();++i){shared[i]=bfloat16_t(float(int(i%17)-8)/11);out[i].bits=0x7fc1;}
  a.output=uintptr_t(out.data());a.sharedRoutedOutput=uintptr_t(route.data());a.sharedOutput=uintptr_t(shared.data());
  a.reduceIndex=uintptr_t(inverse.data());a.weights=uintptr_t(weights.data());
  beforePayload=[](uintptr_t){};readFlag=[](uintptr_t){return uint64_t{16};};
  Timer timer;NoReduceObserver observer;
  for(unsigned worker=0;worker<28;++worker){
   ub.fill(0xa5);ReduceGeneric<true>(a,worker,28,timer,observer);
   for(auto&q:queues)assert(q.empty());for(auto&kind:events)for(bool bit:kind)assert(!bit);
  }
  for(unsigned worker=0;worker<56;++worker){
   ub.fill(0xa5);SharedAdd(a,worker,timer);
   for(auto&q:queues)assert(q.empty());for(auto&kind:events)for(bool bit:kind)assert(!bit);
  }
  for(unsigned token=0;token<bs;++token)for(unsigned col=0;col<h;++col){
   const auto index=uint64_t(token)*h+col;
   float sum=0;for(unsigned k=0;k<8;++k)sum+=float(Returned(inverse[token*8+k],col))*weights[token*8+k];
   assert(route[index]==sum);
   assert(out[index].bits==bfloat16_t(sum+float(shared[index])).bits);
   differentRoundings+=out[index].bits!=bfloat16_t(float(bfloat16_t(sum))+float(shared[index])).bits;
  }
  ++cases;
 }
 assert(differentRoundings>0);
 std::cout<<"PASS "<<cases<<" FP32 Reduce + six-slot vector-sum cases; detected "<<differentRoundings<<" double-rounding differences\n";
}
"""

def test_shared_reduce_sum_pipeline(tmp_path):
    command=shlex.split(os.environ.get("CXX","c++"))
    if not command or not shutil.which(command[0]):pytest.skip("Host C++ compiler required")
    fake=FAKE_ASCEND.replace("inline bfloat16_t Returned","inline uint32_t physicalHidden=3584;\ninline bfloat16_t Returned")
    fake=fake.replace("/ 3584U","/ physicalHidden").replace("% 3584U","% physicalHidden")
    fake += r"""
namespace AscendC {
template<class T> void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,uint32_t n){
 Enqueue(Vector,[=]{std::copy_n(src.Data(),n,dst.Data());return true;});
}
}
"""
    (tmp_path/"kernel_operator.h").write_text(fake,encoding="utf-8")
    source=tmp_path/"model.cpp";source.write_text(DRIVER,encoding="utf-8")
    binary=tmp_path/("model.exe" if os.name=="nt" else "model")
    headers=Path(__file__).resolve().parents[1]/"csrc/kernels/mega_moe/urma"
    subprocess.run(command+["-std=c++17","-O2","-UNDEBUG","-ffp-contract=off","-I"+str(tmp_path),"-I"+str(headers),str(source),"-o",str(binary)],check=True,timeout=120)
    subprocess.run([str(binary)],check=True,timeout=180)
