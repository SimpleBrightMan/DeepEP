"""Full-output independent NumPy Golden gate using the public MegaMoE API."""
import argparse
import faulthandler
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import CASES,CONFIG,package,sha,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--golden-dir',type=Path,required=True)
p.add_argument('--label',required=True)
a=p.parse_args()
rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
identity=package(a.package_root)
from deep_ep_experimental.megamoe import golden
from deep_ep_experimental.megamoe.benchmark import accuracy,inputs
from deep_ep_experimental.megamoe.reference import make_weights,quantize_input

# Fail before opening a device context if any of this rank's full-output oracle files is absent/corrupt.
for routing,bs in CASES:
    golden.load(a.golden_dir,golden.contract(CONFIG,world,bs,'fp8',routing,0),rank)
import numpy as np
import torch
import torch_npu
from deep_ep_experimental.megamoe import api
from coordinator import open_control_group,CONTROL_KIND

torch.set_num_threads(1)
torch.npu.set_device(local)
if api._extension().api_protocol!=8:raise ValueError('Native API protocol mismatch')
group=open_control_group(api,rank,world,a.label)
class LocalAccuracyReport:
    # 复用仓库原有逐元素/计数判定，只把最终报告汇总移到离线，避免大报告Gloo汇总超时。
    def all_gather(self,value,stage):return [value]
model=CONFIG['model'];rdir=a.output/f'rank_{rank}'
rdir.mkdir(parents=True,exist_ok=False)
write(rdir/'identity.json',dict(rank=rank,local_rank=local,ep=world,native=identity['native'],
    revision=identity['revision'],worker_sha256=sha(Path(__file__)),config=CONFIG,
    torch=torch.__version__,torch_npu=torch_npu.__version__,capacity=4096,timer=False,
    host_coordination=CONTROL_KIND,device_communication='shmem_mte',check_tokens=0))
buffer=api.SymmBuffer(group,num_max_tokens_per_rank=4096,**model,ranks_per_node=8)
raw=make_weights(model['num_experts']//world,model['hidden'],model['intermediate_hidden'],CONFIG['seed']+rank)
def pair(values):
    data,scale=(torch.from_numpy(v).npu() for v in values)
    return data,scale.view(torch.float8_e8m0fnu)
first,second=api.transform_weights_for_mega_moe(*(pair(v) for v in raw))
del raw
try:
    for routing,bs in CASES:
        faulthandler.dump_traceback_later(120,repeat=True)
        x,ids,weights=inputs(CONFIG,rank,bs,routing,world=world)
        data,scale=quantize_input(x)
        spec=golden.contract(CONFIG,world,bs,'fp8',routing,0)
        write(rdir/f'{routing}-bs{bs}-contract.json',spec)
        buffer.x[:bs].copy_(torch.from_numpy(data).npu().view(torch.float8_e4m3fn))
        buffer.x_sf[:bs].copy_(torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))
        buffer.topk_idx[:bs].copy_(torch.from_numpy(ids).npu())
        buffer.topk_weights[:bs].copy_(torch.from_numpy(weights).npu())
        buffer.validate_inputs(bs);buffer.prepare(bs)
        y=torch.empty((bs,model['hidden']),device='npu',dtype=torch.bfloat16)
        api.fp8_fp4_mega_moe(y,first,second,buffer)
        group.barrier(f'{routing}/BS{bs}/warmup')
        stats=torch.full((model['num_experts']//world,),7,device='npu',dtype=torch.int32)
        api.fp8_fp4_mega_moe(y,first,second,buffer,cumulative_local_expert_recv_stats=stats)
        counts=stats.clone()-7
        previous=y.clone()
        api.fp8_fp4_mega_moe(y,first,second,buffer,cumulative_local_expert_recv_stats=stats)
        torch.npu.synchronize()
        repeats=group.all_gather(bool(torch.equal(y,previous)),f'{routing}/BS{bs}/repeat_output')
        reports=accuracy(CONFIG,rank,world,bs,y,counts,0,routing,stats,group=LocalAccuracyReport(),
            golden_dir=a.golden_dir,input_kind='fp8')
        write(rdir/f'{routing}-bs{bs}-accuracy.json',dict(reports=reports,repeat_equal=repeats,report_scope='local'))
        if not all(repeats) or not all(r['passed'] for r in reports):
            raise AssertionError(f'EP{world} {routing} BS{bs} rank={rank}: Golden/count/repeat FAILED')
        group.barrier(f'{routing}/BS{bs}/all_local_checks_saved')
        if rank==0:print(f'GOLDEN_PASS EP{world} {routing} BS{bs} full_tokens={bs} ranks={world}',flush=True)
        faulthandler.cancel_dump_traceback_later()
        del x,ids,weights,data,scale,y,previous,counts,stats
    group.barrier('all_accuracy_cases_complete')
    write(rdir/'complete.json',dict(passed=True,cases=len(CASES),check_tokens=0,ep=world,rank=rank))
    buffer.destroy();group.close()
except BaseException as error:
    group.abort(f'golden EP{world} rank={rank}: {error}')
    raise
