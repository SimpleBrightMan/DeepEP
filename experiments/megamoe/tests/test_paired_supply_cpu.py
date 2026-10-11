"""Host checks for the actual plan plus a logical producer/consumer model.

These checks do not emulate hardware flag timing or prove DMA layout correctness;
the EP8 accuracy gate remains mandatory before any performance measurement.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


HEADER = Path(__file__).resolve().parents[1] / "csrc/kernels/mega_moe/catlass_gmm/a8w4_k_plan.hpp"


def test_compiled_plan_capacity_and_paired_consumption(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("requires a host C++ compiler")
    source = tmp_path / "plan.cpp"
    source.write_text('''#include <cassert>
#include <iostream>
#define __aicore__
#include "a8w4_k_plan.hpp"
int main() {
  using namespace MegaMoeImpl::GmmKernel;
  using B = A8W4BufferPlan;
  static_assert(B::BBase == B::ABase + B::AStages*B::ABytes);
  static_assert(B::ScaleABase == B::BBase + B::BStages*B::BBytes);
  static_assert(B::ScaleBBase + B::ScaleStages*B::ScaleBytes == 512*1024);
  for (unsigned m : {1,4,12,32,64,127,128,129,256})
    for (unsigned n : {64,128,192,256}) {
      auto p = MakeA8W4KPlan(m,n,true);
      auto q = MakeA8W4KPlan(m,n,false);
      assert(p.ka==q.ka && p.kb==q.kb && p.bColumns==q.bColumns);
      assert(((m+15)/16*16)*p.ka <= B::ABytes);
      assert(p.kb*p.bColumns <= B::BBytes);
      assert(p.ka==p.kb && 2048%p.kb==0);
      assert((n+p.bColumns-1)/p.bColumns <= B::BStages);
      std::cout << m << " " << n << " " << p.ka << " " << p.kb << " " << p.bColumns << "\\n";
    }
}
''', encoding="utf-8")
    executable = tmp_path / "plan"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(HEADER.parent),
                    str(source), "-o", str(executable)], check=True)
    plans = subprocess.check_output([str(executable)], text=True).splitlines()
    for line in plans:
        m, n, ka, kb, cols = map(int, line.split())
        producer_cursor = consumer_cursor = 0
        # Carry slot indices across heterogeneous problems, as phase handoff does.
        for total_k in (64, 192, 256, 320, 512, 3072, 7168):
            produced, consumed = [], []
            cells = {}
            for ko in range(0, total_k, kb):
                nk = min(kb, total_k-ko)
                chunks = (n+cols-1)//cols
                for no in range(0, n, cols):
                    produced.append((producer_cursor, ko, no, nk, min(cols, n-no)))
                    producer_cursor = (producer_cursor+1)%4
                for chunk in range(chunks):
                    consumed.append(((consumer_cursor+chunk)%4, ko, chunk*cols,
                                     nk, min(cols, n-chunk*cols)))
                for inner_k in range(0, nk, 256 if m <= 128 else 128):
                    actual_k = min(256 if m <= 128 else 128, nk-inner_k)
                    for no in range(0, n, 128 if m <= 128 else 256):
                        actual_n = min(128 if m <= 128 else 256, n-no)
                        assert ((m+15)//16*16)*actual_k <= 32768
                        assert actual_k*actual_n <= 32768
                        for k64 in range(ko+inner_k, ko+inner_k+actual_k, 64):
                            for n64 in range(no, no+actual_n, 64):
                                cells[k64,n64] = cells.get((k64,n64),0)+1
                consumer_cursor = (consumer_cursor+chunks)%4
            assert produced == consumed
            assert producer_cursor == consumer_cursor
            assert cells == {(k,n0):1 for k in range(0,total_k,64) for n0 in range(0,n,64)}
