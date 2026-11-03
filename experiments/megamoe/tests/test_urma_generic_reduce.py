"""Exercise the real generic reducer under queued DMA, not just offset arithmetic."""
import os, shlex, shutil, subprocess
from pathlib import Path
import pytest
from test_urma_reduce_pipeline import FAKE_ASCEND

DRIVER = r"""

#include "reduce_generic.hpp"
#include <iostream>
using namespace MegaMoeUrma;using namespace AscendC;
struct Timer { uint64_t TraceBegin(){return 1;} void TraceEnd(TraceEvent,uint64_t,const TraceMeta&){} };
int main(){
 unsigned cases=0;
 for(unsigned h:{256U,3584U,5120U,4096U,6144U,8192U})for(unsigned workers:{28U,41U,53U})for(unsigned pr:{0U,1U,2U}) {
  unsigned bs=(h==8192 && workers==28 && pr==0)?8192:73;
  for(unsigned topk=1;topk<=8;++topk) for(unsigned worker:{0U,workers-1}) {
   physicalHidden=h;priority=pr;ub.fill(0xa5);
   Arguments a{};a.config.bs=bs;a.config.hidden=h;a.config.topk=topk;a.config.world=16;
   a.symmetric=returnedBase;a.localFlags=uintptr_t{1}<<43;
   std::vector<int64_t> inverse(bs*topk);std::vector<float> weights(bs*topk);
   for(unsigned i=0;i<bs*topk;++i){inverse[i]=(i*17U)%(bs*topk);weights[i]=float(int(i%7)-3)/8;}
   std::vector<bfloat16_t> output(uint64_t(bs)*h);for(auto&v:output)v.bits=0x7fc1;
   a.output=uintptr_t(output.data());a.reduceIndex=uintptr_t(inverse.data());a.weights=uintptr_t(weights.data());
   beforePayload=[](uintptr_t){};executePayload={};submitOutput={};executeOutput={};
   readFlag=[](uintptr_t){return uint64_t{16};};onCycle={};
   Timer timer;ReduceGeneric(a,worker,workers,timer);
   const unsigned begin=worker*(bs/workers)+std::min(worker,bs%workers),count=bs/workers+(worker<bs%workers);
   for(unsigned token=0;token<bs;++token)for(unsigned col=0;col<h;++col) {
    if(token<begin||token>=begin+count){assert(output[uint64_t(token)*h+col].bits==0x7fc1);continue;}
    float sum=0;for(unsigned k=0;k<topk;++k)sum+=float(Returned(inverse[token*topk+k],col))*weights[token*topk+k];
    assert(output[uint64_t(token)*h+col].bits==bfloat16_t(sum).bits);
   }
   for(auto&q:queues)assert(q.empty());for(auto&kind:events)for(bool bit:kind)assert(!bit);
   ++cases;
  }
 }
 std::cout<<"PASS "<<cases<<" generic Reduce DMA/event/arithmetic cases\n";
}
"""
def test_generic_reduce_event_and_arithmetic_model(tmp_path):
    command=shlex.split(os.environ.get("CXX","c++"))
    if not command or not shutil.which(command[0]):pytest.skip("Host C++ compiler required")
    fake=FAKE_ASCEND.replace("inline bfloat16_t Returned","inline uint32_t physicalHidden=3584;\ninline bfloat16_t Returned")
    fake=fake.replace("/ 3584U","/ physicalHidden").replace("% 3584U","% physicalHidden")
    (tmp_path/"kernel_operator.h").write_text(fake,encoding="utf-8")
    source=tmp_path/"model.cpp";source.write_text(DRIVER,encoding="utf-8")
    binary=tmp_path/("model.exe" if os.name=="nt" else "model")
    headers=Path(__file__).resolve().parents[1]/"csrc/kernels/mega_moe/urma"
    subprocess.run(command+["-std=c++17","-O2","-UNDEBUG","-ffp-contract=off","-I"+str(tmp_path),"-I"+str(headers),str(source),"-o",str(binary)],check=True,timeout=120)
    subprocess.run([str(binary)],check=True,timeout=120)
