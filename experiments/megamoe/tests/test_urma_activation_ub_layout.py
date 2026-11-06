"""Actual UB layout/pointer helpers, including in-place fused row compaction."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest


def test_fused_mailbox_and_relocated_shared_layout(tmp_path):
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(cxx[0]):
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[1]
    text = (root / "csrc/kernels/mega_moe/epilogue/block_epilogue_ub_layout.hpp").read_text(encoding="utf-8")
    begin = text.index("struct ActivationMxQuantUbOffsets")
    end = text.index("}  // namespace Detail", begin)
    helpers = text[begin:end]
    source = tmp_path / "activation_layout.cpp"
    source.write_text(r'''
#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#define __aicore__
#define __ubuf__
using bfloat16_t = uint16_t;
namespace AscendC { constexpr uint32_t ONE_BLK_SIZE=32U; }
constexpr uint32_t ACTIVATION_N_HALF=2U,INT32_PER_256B=8U;
''' + helpers + r'''
int main() {
    constexpr auto fused=BuildActivationMxQuantUbOffsets<uint16_t,256,256,true>();
    static_assert(fused.firstInputElementCapacity*2==128*1024);
    static_assert(fused.activationQuantElementCapacity==256*128);
    static_assert(fused.quantOutputOffsetBytes==64*1024);
    static_assert(fused.reciprocalScaleOffsetBytes+fused.scaleElementCapacity*2==101*1024);
    constexpr auto shared=BuildActivationMxQuantUbOffsets<uint16_t,64,256,false>();
    static_assert(shared.secondInputOffsetBytes==32*1024);
    static_assert(128*1024+shared.secondInputOffsetBytes+shared.firstInputElementCapacity*2==192*1024);
    for (uint32_t rows: {1U,17U,64U,127U,128U,129U,255U,256U}) {
        std::vector<uint16_t> ub(128*1024/2,0xA55A),reference(ub.size());
        for (uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<256;++c)
            ub[r*256+c]=reference[r*256+c]=uint16_t((r*397+c*7)%65535);
        auto p=ResolveActivationMxQuantUbPointers<uint16_t,256*256,true>(
            ub.data(),nullptr,ub.data(),reinterpret_cast<int8_t*>(ub.data())+fused.quantOutputOffsetBytes,
            ub.data()+fused.quantScaleOffsetBytes/2,ub.data()+fused.maxExpOffsetBytes/2,
            ub.data()+fused.reciprocalScaleOffsetBytes/2,128,0);
        assert(p.firstInput==ub.data()&&p.secondInput==ub.data()+128);
        // Existing vector kernel loads gate/up64 values before its output
        // store. Check that compacting to rowstride128 never destroys an
        // unread gate/up value, including the second64-vector of each row.
        for(uint32_t r=0;r<rows;++r)for(uint32_t v=0;v<128;v+=64) {
            std::array<uint16_t,64> gate,up;
            for(uint32_t j=0;j<64;++j) {
                gate[j]=p.firstInput[r*256+v+j];up[j]=p.secondInput[r*256+v+j];
                assert(gate[j]==reference[r*256+v+j]);
                assert(up[j]==reference[r*256+128+v+j]);
            }
            for(uint32_t j=0;j<64;++j)p.activationOutput[r*128+v+j]=gate[j]^up[j];
        }
        for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<128;++c)
            assert(ub[r*128+c]==uint16_t(reference[r*256+c]^reference[r*256+128+c]));
        assert(rows*128<=fused.activationQuantElementCapacity);
        assert(rows*4<=fused.scaleElementCapacity);
    }
}
''', encoding="utf-8")
    binary = tmp_path / ("activation_layout.exe" if os.name == "nt" else "activation_layout")
    subprocess.run(cxx + ["-std=c++17", "-O2", "-UNDEBUG", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
