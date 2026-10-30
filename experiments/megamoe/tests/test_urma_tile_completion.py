"""Host execution of the actual bounded completion helper; no NPU calls."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_same_stage_defers_but_switch_dependency_and_exit_publish(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required")
    source = tmp_path / "completion.cpp"
    source.write_text(r'''
#include "tile_completion.hpp"
#include <cassert>
#include <cstdint>
using namespace MegaMoeUrma;
struct Backend {
    unsigned submitted=0, captured=0, published=0;
    bool armed=false;
    void Arm() { assert(!armed); captured=submitted; armed=true; }
    void Complete(bool, uint64_t tile, const TraceMeta&) {
        assert(armed && tile==published && captured==tile+1);
        assert(submitted==captured || submitted==captured+1);
        ++published; armed=false;
    }
};
int main() {
    for (unsigned types=0;types<256;++types) for(unsigned readyMask=0;readyMask<256;++readyMask) {
        Backend b; TileCompletion<Backend> q(b);
        for(unsigned t=0;t<8;++t) {
            const bool g1=(types>>t)&1, ready=(readyMask>>t)&1;
            q.BeforeStage(g1);
            if(t && g1!=bool((types>>(t-1))&1)) assert(b.published==t);
            q.BeforeDependency(ready);
            if(!ready) assert(b.published==t);
            ++b.submitted;
            q.AfterSubmit(g1,t,{});
            assert(b.published==t && b.captured==t+1);
        }
        q.Flush(); q.Flush(); assert(b.published==8 && !b.armed);
    }
    Backend empty; TileCompletion<Backend> q(empty); q.BeforeStage(false); q.Flush(); assert(!empty.armed);
}
''', encoding="utf-8")
    header = Path(__file__).resolve().parents[1] / "csrc/kernels/mega_moe/urma"
    binary = tmp_path / "completion"
    subprocess.run(command + ["-std=c++17", "-O1", "-UNDEBUG", "-I" + str(header), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
