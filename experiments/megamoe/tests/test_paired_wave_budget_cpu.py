"""编译执行实际host预算分支，覆盖原路径与尾wave；不能代替NPU同步验收。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class PairedWaveBudgetTests(unittest.TestCase):
    def test_actual_host_policy(self):
        compiler = shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('Requires host C++ compiler; execute on build host')
        program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include "common/mega_moe_schedule_policy.hpp"
template<class T> T CeilDiv(T n,T d) { return (n+d-1)/d; }
uint32_t groups(uint32_t bs,uint32_t intermediate,uint32_t hidden,uint32_t aic,bool is_a8w4) {
    return MegaMoeImpl::CalcWaveMGroupsPerWave(is_a8w4,bs,intermediate*2,hidden,aic);
}
// 冻结decf4bb的预算公式作参考，执行被测生产helper而非检查其文本。
uint32_t frozen(uint32_t bs,uint32_t intermediate,uint32_t hidden,uint32_t aic,bool w4) {
    const uint64_t g1=(intermediate+127U)/128U;
    if(w4 && bs<=256U) return CeilDiv<uint64_t>(uint64_t(aic)*2U,g1)*2U;
    if(w4) return std::max(CeilDiv<uint64_t>(uint64_t(aic)*2U,g1),CeilDiv<uint64_t>(aic,(hidden+255U)/256U));
    return std::max(CeilDiv<uint64_t>(uint64_t(aic)*4U,(intermediate*2U+255U)/256U),CeilDiv<uint64_t>(aic,(hidden+255U)/256U));
}
int main() {
    for(unsigned bs : {1U,256U,257U,511U,512U,4096U,4097U,65536U}) {
      assert(MegaMoeImpl::UseGmm2WaveLag(bs)==(bs>=512U));
      for(unsigned i : {128U,256U,3008U,3072U,4096U})
       for(unsigned h : {256U,1024U,7168U,8192U})
        for(unsigned c : {1U,24U,32U,64U})
         for(bool w4 : {false,true}) assert(groups(bs,i,h,c,w4)==frozen(bs,i,h,c,w4));
    }
    // EP32/BS4096：12专家×8个M256块=96组；每组24个配对任务，3组/波，共32波。
    for(unsigned bs : {257U,512U,1024U,2048U,4096U,4097U,8192U,16384U,65536U})
        assert(groups(bs,3072,7168,32,true)==3);
    assert(groups(256,3072,7168,32,true)==6);
    assert(MegaMoeImpl::W4A8_PAIRED_GATE_UP_MAX_TOKENS==65536U);
    assert(groups(4096,3072,7168,32,false)==6);
    assert(groups(8192,3072,7168,32,false)==6);
    assert(groups(4096,3008,7168,32,true)==3);
    assert(groups(4096,3072,256,32,true)==32);
    assert(groups(4096,128,7168,32,true)==64);
    for(unsigned total=0;total<300;++total) {
        unsigned seen=0,waves=0,budget=groups(4096,3072,7168,32,true);
        while(seen<total) { unsigned count=std::min(budget,total-seen);assert(count>0);seen+=count;++waves; }
        assert(seen==total && waves==(total+budget-1)/budget);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='r31-wave-budget-') as folder:
            path = Path(folder)
            (path / 'check.cpp').write_text(program, encoding='utf-8')
            binary = path / 'check'
            subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(ROOT / 'experiments/megamoe/csrc/kernels/mega_moe'), str(path / 'check.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
