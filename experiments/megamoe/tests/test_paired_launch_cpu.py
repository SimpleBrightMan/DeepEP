"""执行实际kernel派发分支；仅替换launch语法，不模拟NPU计算或证明性能。"""
from pathlib import Path

from test_device_schedule import compile_and_run, cpp_body


def test_routed_a8w4_keeps_paired_gm_for_all_supported_bs(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = (root / 'experiments/megamoe/csrc/kernels/mega_moe/mega_moe.cpp').read_text(encoding='utf-8')
    body = cpp_body(source, 'void mega_moe_kernel_do(')
    dispatch = body[body.index('    // 按本次实际 BS'): -1]
    # 编译执行原分支和实参；CPU记录模板身份，不提交设备kernel。
    dispatch = dispatch.replace('<<<block_dim, ub_bytes, stream>>>', '')
    (tmp_path / 'paired_launch.inc').write_text(dispatch, encoding='utf-8')
    program = r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "common/mega_moe_constants.hpp"
#include "common/mega_moe_aiv1_ub_layout.hpp"
int selected = 0;
template<bool A8W4, bool Paired=false, bool Cv=false, class... Args>
void mega_moe_kernel(Args...) { selected=(A8W4 ? 400 : 800)+(Paired ? 10 : 0)+(Cv ? 1 : 0); }
void Launch(unsigned bs, unsigned ep, bool is_a8w4) {
    struct { unsigned bs,topK,epWorldSize,moeExpertPerRank; } tiling{bs,6,ep,384/ep};
    int context=0,x=0,x_scales=0,topk_ids=0,topk_weights=0,weight1=0,weight2=0;
    int weight_scales1=0,weight_scales2=0,shared_weight1=0,shared_weight2=0;
    int shared_weight_scales1=0,shared_weight_scales2=0,y=0,expert_token_nums=0;
    int workspace=0,timer=0,ffts_addr=0,compute_replay=0;
    selected=0;
#include "paired_launch.inc"
}
int main() {
    assert(MegaMoeImpl::W4A8_PAIRED_GATE_UP_MAX_TOKENS==65536U);
    for(unsigned ep : {8U,32U,64U,128U})
        for(unsigned bs : {1U,32U,96U,256U,257U,511U,512U,1024U,4096U,4097U,8192U,65536U}) {
            Launch(bs,ep,true); assert(selected==410); // paired GM，不是planar或CV
            Launch(bs,ep,false); assert(selected==800); // A8W8不受影响
        }
}
'''
    check = tmp_path / 'paired_launch.cpp'
    check.write_text(program, encoding='utf-8')
    compile_and_run(check, tmp_path)
