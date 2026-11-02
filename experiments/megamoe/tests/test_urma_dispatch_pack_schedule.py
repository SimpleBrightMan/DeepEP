"""Check the production batch/Pack helpers without an NPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_pack_after_put_preserves_ramp_and_never_waits_for_own_future_pack(tmp_path):
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
    // Full ramp: Dispatch first; extend the 4->8 transition's Pack quota.
    {
        unsigned packed=2,sent=0;
        unsigned packs[]{2,2,3,7,9,9};
        unsigned sends[]{1,1,2,4,8,8};
        for(unsigned step=0;step<6;++step) {
            unsigned count=DispatchBatchLimit(step,100-sent,false);
            assert(count==sends[step] && sent+count<=packed);
            sent+=count;
            unsigned limit=DispatchPackAfterLimit(count,sent,step+1,100,packed,100);
            assert(limit-packed==packs[step]);packed=limit;
        }
    }
    { // Exact EP4/BS2048/CV256 example, including final clamping.
        unsigned sends[]{1,1,2,4,8}, packs[]{2,2,3,7,0};
        unsigned packed=2,sent=0;
        for(unsigned step=0;step<5;++step) {
            unsigned count=DispatchBatchLimit(step,16-sent,packed>=16);
            assert(sent+count<=packed); // Must already be ready BEFORE post-submit Pack.
            sent+=count;
            unsigned limit=DispatchPackAfterLimit(count,sent,step+1,16,packed,16);
            assert(count==sends[step] && limit-packed==packs[step]);
            packed=limit;
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
            assert(count>0 && sent+count<=packed); // No pre-PUT Pack is allowed.
            if(step<2) assert(count==1);
            sent+=count;
            ++step;
            unsigned limit=DispatchPackAfterLimit(count,sent,step,peer,packed,total);
            assert(limit>=packed);
            while(packed<limit) { assert(packed<total); ++visits[packed++]; }
            if(sent<peer) assert(sent+DispatchBatchLimit(step,peer-sent,packed>=total)<=packed);
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
