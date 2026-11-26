"""Execute production byte offsets and compare both MAD shapes with an independent packed byte stream."""
from test_device_schedule import compile_and_run
from pathlib import Path


def test_packed_weight_tile_address_and_dma_runs(tmp_path):
    fixture=tmp_path/'packed_weight_tiles.cpp'
    fixture.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "catlass_gmm/packed_weight_tiles.hpp"
using namespace MegaMoeImpl::GmmKernel;
int main(){for(uint32_t fullK:{64U,128U,256U,384U,512U,768U,3072U,7168U}){
const uint32_t n=512;
std::vector<uint32_t> physical;
for(uint32_t nb=0;nb<n;nb+=256)for(uint32_t kw=0;kw<fullK;kw+=512)
for(uint32_t ns=0;ns<256;ns+=128)for(uint32_t k=kw;k<std::min(kw+512,fullK);k+=32)
for(uint32_t row=nb+ns;row<nb+ns+128;++row)for(uint32_t byte=0;byte<16;++byte)
physical.push_back(row*(fullK/2)+k/2+byte);
assert(physical.size()==uint64_t(n)*fullK/2);
for(uint32_t tileK:{256U,512U}){
const uint32_t tileN=tileK==512?128:256;
for(uint32_t ns=0;ns<n;ns+=tileN)for(uint32_t ks=0;ks<fullK;ks+=tileK){
const uint32_t validK=std::min(tileK,fullK-ks),bytes=validK*64;
const uint32_t gap=(PackedWeightWindowK(fullK,ks)-validK)*64;
const uint64_t offset=PackedWeightByteOffset(fullK,ns,ks);
assert(PackedWeightLogicalOffset(fullK,ns,ks)==2*offset);
for(uint32_t panel=0;panel<tileN/128;++panel){
uint64_t cursor=offset+uint64_t(panel)*(bytes+gap);
for(uint32_t kb=ks;kb<ks+validK;kb+=32)for(uint32_t row=ns+panel*128;row<ns+(panel+1)*128;++row)
for(uint32_t byte=0;byte<16;++byte)assert(physical.at(cursor++)==row*(fullK/2)+kb/2+byte);
assert(cursor==offset+uint64_t(panel)*(bytes+gap)+bytes);
}
}
}
}}
''')
    compile_and_run(fixture,tmp_path)
