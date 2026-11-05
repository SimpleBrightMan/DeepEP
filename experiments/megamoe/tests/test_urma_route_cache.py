"""Execute the actual RouteFlush loop with a 64B DCache test adapter."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest


def test_route_flush_covers_every_dirty_cache_line(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        pytest.skip("requires a C++ compiler")
    root = Path(__file__).resolve().parents[1]
    text = (root / "csrc/kernels/mega_moe/urma/device_routes.hpp").read_text(encoding="utf-8")
    begin = text.index("__aicore__ inline void RouteFlush(")
    end = text.index("\nstruct DeviceRouteOutput", begin)
    function = text[begin:end]
    source = tmp_path / "route_cache.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>
#define __aicore__
#define __gm__
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T> struct GlobalTensor {
    uintptr_t address;
    void SetGlobalBuffer(T* ptr){address=reinterpret_cast<uintptr_t>(ptr);}
};
std::vector<uint64_t> calls;
std::map<uint64_t,bool> dirty;
template<class T,CacheLine,DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T> tensor){
    uint64_t line=tensor.address&~uint64_t{63};calls.push_back(line);dirty[line]=false;
}
''' + function + r'''
int main(){
    for(uint64_t offset=0;offset<64;offset+=4)for(uint64_t bytes:{0U,1U,4U,32U,64U,68U,128U,193U,86016U}){
        const uint64_t begin=0x1000+offset,end=begin+bytes;
        std::vector<uint64_t> expected;dirty.clear();calls.clear();
        if(bytes)for(uint64_t line=begin&~uint64_t{63};line<end;line+=64){expected.push_back(line);dirty[line]=true;}
        RouteFlush(begin,bytes);
        assert(calls==expected);
        // No old Scalar count line may survive into the DMA prefix rewrite.
        for(const auto& entry:dirty)assert(!entry.second);
    }
}
''', encoding="utf-8")
    exe = tmp_path / ("route_cache.exe" if os.name == "nt" else "route_cache")
    subprocess.run(compiler + ["-std=c++17", "-O2", str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
