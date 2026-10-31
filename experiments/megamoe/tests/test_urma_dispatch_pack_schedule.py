"""Check the production batch/Pack helpers without an NPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_pack_before_put_covers_batch_without_duplicate_or_missing_quota(tmp_path):
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
    // Full batches: after two startup rounds, increments are 2,2,3,5,9.
    unsigned p=2;
    for(unsigned step=0;step<5;++step) {
        unsigned count=DispatchBatchLimit(step,100,false);
        unsigned next=DispatchPackLimit(count,p,100);
        assert(next-p==count+1); p=next;
    }
    { // Exact EP4/BS2048/CV256 example, including final clamping.
        unsigned sends[]{1,1,2,4,8}, packs[]{2,2,3,5,2};
        unsigned packed=2,sent=0;
        for(unsigned step=0;step<5;++step) {
            unsigned count=DispatchBatchLimit(step,16-sent,packed>=16);
            unsigned limit=DispatchPackLimit(count,packed,16);
            assert(count==sends[step] && limit-packed==packs[step]);
            packed=limit;sent+=count;
        }
        assert(packed==16 && sent==16);
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
            unsigned limit=DispatchPackLimit(count,packed,total);
            assert(limit-packed<=count+1);
            while(packed<limit) { assert(packed<total); ++visits[packed++]; }
            assert(count>0 && sent+count<=packed); // this owner's entire batch is ready
            if(step<2) assert(count==1);
            sent+=count;
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
