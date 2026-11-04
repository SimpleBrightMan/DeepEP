import os,shlex,shutil,subprocess
from pathlib import Path
import pytest

SOURCE=r"""
#include "completion_policy.hpp"
#include <cassert>
#include <vector>
using namespace MegaMoeUrma;
struct Backend {
 unsigned submitted=0,fence=0,arms=0;
 std::vector<unsigned> retired;
 void Arm(){assert(!fence);fence=submitted;++arms;}
 void Complete(bool first,uint64_t tile,const TraceMeta& meta){
  assert(fence==tile+1);assert(fence<=submitted);
  retired.push_back(unsigned(tile));fence=0;
 }
};
template<bool Enabled>void exercise(){
 Backend b;Demo::OptionalTileCompletion<Enabled,Backend> p(b);TraceMeta m;
 ++b.submitted;p.AfterSubmit(true,0,m);assert(b.retired.empty());
 ++b.submitted;p.AfterSubmit(true,1,m);
 if constexpr(Enabled)assert((b.retired==std::vector<unsigned>{0}));else assert(b.retired.empty());
 p.BeforeStage(false);
 if constexpr(Enabled)assert((b.retired==std::vector<unsigned>{0,1}));
 ++b.submitted;p.AfterSubmit(false,2,m);
 ++b.submitted;p.AfterSubmit(false,3,m);
 if constexpr(Enabled)assert((b.retired==std::vector<unsigned>{0,1,2}));
 p.Flush();p.Flush();
 if constexpr(Enabled){assert((b.retired==std::vector<unsigned>{0,1,2,3}));assert(b.arms==4);}
 else {assert(b.retired.empty());assert(b.arms==0);}
 assert(!b.fence);
}
int main(){exercise<false>();exercise<true>();}
"""
def test_actual_optional_completion_policy(tmp_path):
    cmd=shlex.split(os.environ.get('CXX','c++'))
    if not cmd or not shutil.which(cmd[0]):pytest.skip('Host C++ compiler required')
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'check.cpp';source.write_text(SOURCE,encoding='utf-8')
    binary=tmp_path/('check.exe' if os.name=='nt' else 'check')
    subprocess.run(cmd+['-std=c++17','-O2','-UNDEBUG','-I'+str(root),str(source),'-o',str(binary)],check=True,timeout=120)
    subprocess.run([str(binary)],check=True,timeout=30)
