import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

SOURCE=r'''
#include "plan.hpp"
#include <cassert>
#include <iostream>
int main() {
    unsigned cases=0;
    for(unsigned world:{4U,8U,16U})for(unsigned bs:{1024U,2048U,4096U,8192U})for(unsigned rank:{0U,world-1}) {
        auto p=Demo::MakePlan(bs,world,rank);auto& c=p.config;
        assert(c.receiveRows==uint64_t(bs)*8);
        assert(p.computeTasks.size()==2*p.computeBatches.size());
        for(unsigned b=0;b<p.computeBatches.size();++b) {
            assert(p.computeTasks[2*b].stage==MegaMoeUrma::ComputeBatchStage::Gmm1);
            assert(p.computeTasks[2*b+1].stage==MegaMoeUrma::ComputeBatchStage::Gmm2);
            assert(p.computeTasks[2*b].batchIndex==b && p.computeTasks[2*b+1].batchIndex==b);
        }
        uint64_t tiles1=0,tiles2=0,rows=0;
        for(const auto& f:p.fragments) {
            assert(f.rows>0 && f.rows<=256);rows+=f.rows;
            uint32_t n1=MegaMoeUrma::Gmm1ComputeTileCount(f.rows,c),n2=14;
            for(unsigned core=0;core<28;++core) {
                for(unsigned t=MegaMoeUrma::FirstOwnedComputeTile(f.gmm1TileBegin,core,28);t<n1;t+=28)++tiles1;
                for(unsigned t=MegaMoeUrma::FirstOwnedComputeTile(f.gmm2TileBegin,core,28);t<n2;t+=28)++tiles2;
            }
        }
        assert(rows==c.receiveRows && tiles1==c.gmm1Tiles && tiles2==c.gmm2Tiles);
        assert(c.selfSendRow+c.selfRows<=uint64_t(bs)*8 && c.selfRecvRow+c.selfRows<=c.receiveRows);
        ++cases;
    }
    std::cout<<"PASS "<<cases<<" EP/BS plans: exact row coverage, tile ownership and alternating batches\n";
}
'''


def test_production_task_plan(tmp_path):
    command=shlex.split(os.environ.get('CXX','c++'))
    if not command or not shutil.which(command[0]):pytest.skip('Host C++ compiler required')
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'plan.cpp';source.write_text(SOURCE,encoding='utf-8')
    binary=tmp_path/('plan.exe' if os.name=='nt' else 'plan')
    subprocess.run(command+['-std=c++17','-O2','-UNDEBUG','-I'+str(root),str(source),'-o',str(binary)],check=True,timeout=120)
    subprocess.run([str(binary)],check=True,timeout=60)
