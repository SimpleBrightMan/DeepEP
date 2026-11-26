"""Host checks the production plan; native tests must verify the asynchronous data path."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_production_resident_plan_and_guard_regions(tmp_path):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('requires a host C++ compiler')
    header = Path(__file__).resolve().parents[1] / 'csrc/kernels/mega_moe/common'
    source = tmp_path / 'resident.cpp'
    source.write_text('''#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <vector>
#include "mega_moe_aiv1_ub_layout.hpp"
int main() {
  using namespace MegaMoeImpl;
  constexpr auto bs512 = MakeAiv1ResidentUbLayout(7168,8,48,512,6,7392,2);
  static_assert(bs512.valid && bs512.end == 157376 && bs512.dispatchBase == Aiv1ResidentUbLayout::ActivationBytes);
  static_assert(!MakeAiv1ResidentUbLayout(7168,8,48,512,7,7392,2).valid);
  for(unsigned ep : {8,32,64,128})
    for(unsigned h : {1024,7168,8192})
      for(unsigned items : {32,256,512,1024,4096})
        for(unsigned ds=2; ds<=6; ++ds)
          for(unsigned cs : {2,6}) {
            const auto token = (h+255)/256*256 + ((h+31)/32+31)/32*32;
            const auto p=MakeAiv1ResidentUbLayout(h,ep,384/ep,items,ds,token,cs);
            assert(p.valid && p.dispatchBase==Aiv1ResidentUbLayout::ActivationBytes && p.dispatchEnd<=p.combineRowsBase);
            assert(p.combineRowsBase<=p.combineMetaBase && p.combineMetaBase<p.end);
            assert(p.end-p.combineMetaBase==Aiv1ResidentUbLayout::CombineMetaRows*8*4 && p.end%32==0);
            if(p.end>256*1024) continue; // launcher rejects capacity before NPU launch
            std::vector<unsigned char> ub(p.end+32, 0xa5);
            std::fill(ub.begin()+p.dispatchBase,ub.begin()+p.dispatchEnd,0x31);
            for(unsigned turn=0;turn<8;++turn) {
              std::fill(ub.begin(),ub.begin()+Aiv1ResidentUbLayout::ActivationBytes,0x42);
              std::fill(ub.begin()+p.combineRowsBase,ub.begin()+p.end,0x53);
              assert(std::all_of(ub.begin()+p.dispatchBase,ub.begin()+p.dispatchEnd,[](auto x){return x==0x31;}));
              assert(std::all_of(ub.begin()+p.end,ub.end(),[](auto x){return x==0xa5;}));
            }
          }
}
''', encoding='utf-8')
    exe = tmp_path / 'resident'
    subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(header), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
