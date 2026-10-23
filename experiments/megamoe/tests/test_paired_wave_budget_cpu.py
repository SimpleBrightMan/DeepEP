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
        source = (ROOT / 'experiments/megamoe/csrc/forward.cpp').read_text(encoding='utf-8')
        start = source.index('    if (is_a8w4 && t.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS)')
        end = source.index('    t.isPerExpertWeightTensor', start)
        constants = (ROOT / 'experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_constants.hpp').read_text(encoding='utf-8')
        first = constants.index('constexpr uint32_t L1_TILE_M_256')
        last = constants.index('}  // namespace MegaMoeImpl', first)
        program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
template<class T> T CeilDiv(T n,T d) { return (n+d-1)/d; }
namespace MegaMoeImpl {
''' + constants[first:last] + r'''
}
uint32_t groups(uint32_t bs,uint32_t intermediate,uint32_t hidden,uint32_t aic,bool is_a8w4) {
    struct { uint32_t bs,hiddenDim,h,mGroupsPerWave; } t{bs,intermediate*2,hidden,0};
    struct { uint32_t aic; } p{aic};
''' + source[start:end] + r'''
    return t.mGroupsPerWave;
}
int main() {
    // EP32/BS4096：12专家×8个M256块=96组；每组24个配对任务，3组/波，共32波。
    for(unsigned bs : {257U,512U,1024U,2048U,4096U}) assert(groups(bs,3072,7168,32,true)==3);
    assert(groups(256,3072,7168,32,true)==6);
    assert(groups(4097,3072,7168,32,true)==6);
    assert(groups(4096,3072,7168,32,false)==6);
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
            subprocess.run([compiler, '-std=c++17', '-O2', str(path / 'check.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
