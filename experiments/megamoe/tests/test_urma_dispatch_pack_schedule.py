"""Check the production batch/Pack helpers without an NPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_next_batch_is_prepacked_without_duplicate_or_missing_quota(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "schedule.cpp"
    source.write_text(r'''
#include "communication_schedule.hpp"
#include <cassert>
#include <vector>
using namespace MegaMoeUrma;
int main() {
    // Full batches: after two startup rounds, increments are 1,2,4,8,8.
    unsigned p=2;
    for(unsigned step=0;step<5;++step) {
        unsigned next=DispatchPackLimit(step,100,p,100);
        assert(next-p==(step==0?1:step==1?2:step==2?4:8)); p=next;
    }
    for(unsigned total=0;total<=kMaxPeerChunks;++total)
    for(unsigned peer=0;peer<=total;++peer)
    for(unsigned split : {1U,3U,512U}) {
        std::vector<unsigned> visits(total,0);
        for(unsigned r=0;r<2 && r<total;++r) ++visits[r];
        unsigned packed=2, sent=0, step=0;
        while(sent<peer) {
            unsigned count=DispatchBatchLimit(step,peer-sent,packed>=total);
            if(count>split) count=split; // adjacency/PUT-size truncation
            assert(count>0 && sent+count<=packed); // no pre-PUT catch-up
            sent+=count;
            unsigned limit=DispatchPackLimit(step,peer-sent,packed,total);
            while(packed<limit) { assert(packed<total); ++visits[packed++]; }
            ++step;
        }
        // Existing short/empty-peer remainder loop is unchanged.
        while(packed<total) ++visits[packed++];
        for(auto n:visits) assert(n==1);
    }
}
''', encoding="utf-8")
    binary = tmp_path / ("schedule.exe" if os.name == "nt" else "schedule")
    result = subprocess.run(command + ["-std=c++17", "-O2", "-I" + str(
        root / "experiments/megamoe/csrc/kernels/mega_moe/urma"), str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
