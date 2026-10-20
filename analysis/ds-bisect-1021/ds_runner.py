"""Full DS endpoint, fresh package/JIT identity; never substitutes for an ours migration."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--node',required=True)
p.add_argument('--label',required=True)
p.add_argument('--bs',type=int,choices=(512,1024),required=True)
p.add_argument('--trace',action='store_true')
a=p.parse_args()
root=a.root.resolve(); node=root/'ds-bisect-1021'/a.node; out=root/a.label
if not out.is_relative_to(root/'ds-bisect-1021') or '/' in a.node or a.node in ('.','..'):
    raise ValueError('Invalid private experiment path')
identity=json.loads((node/'identity.json').read_text())
package=node/'package'
reference=root/'reference/ep128-supplement-1012-r07-r01'
harness=node/'source/analysis/ds-bisect-1021/harness'
sys.path[:0]=[str(package),str(harness/'ds-harness'),str(harness),str(reference/'ds-python'),str(root)]
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import deep_gemm
from test_mega_moe import MegaMoECase
from deep_gemm.testing.envs import init_seed
from megamoe_measure import collect,summarize,DEFAULT_ALIGNMENT
from layout import tile_shape

rank=int(os.environ['RANK']); bs=a.bs
actual=Path(deep_gemm.__file__).resolve()
if not actual.is_relative_to(package):raise RuntimeError(f'rank={rank} stage=identity wrong package={actual}')
for name,digest in identity['files'].items():
    f=package/name
    if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
        raise RuntimeError(f'case=EP8/BS{bs} rank={rank} stage=identity file={name}')
torch.npu.set_device(rank)
dist.init_process_group('hccl',timeout=timedelta(minutes=5),device_id=torch.device(f'npu:{rank}'))
torch.set_default_dtype(torch.bfloat16);torch.set_default_device('npu');init_seed(20260922)
case=MegaMoECase(dist.distributed_c10d._get_default_group(),7168,3072,384,6,bs,0,
                 activation_clamp=float('inf'),with_baseline=False)
case.create_inputs(bs)
routes=np.arange(rank*bs*6,(rank+1)*bs*6,dtype=np.int64)
ids=((routes%8)*48+(routes//8)%48).reshape(bs,6)
case.buffer.topk_idx[:bs].copy_(torch.from_numpy(ids).to(case.y.device))
all_routes=np.arange(8*bs*6,dtype=np.int64)
assert np.all(np.bincount((all_routes%8)*48+(all_routes//8)%48,minlength=384)==bs//8)
tile_n,tile_k=tile_shape(bs)
for attr in ('transformed_l1_weights','transformed_l2_weights'):
    weight,scale=getattr(case,attr)
    e,n,_,half_k=weight.shape
    original=weight.reshape(e,n,256//tile_n,tile_n,half_k//(tile_k//2),tile_k//32,16)
    packed=original.permute(0,1,4,2,5,3,6).contiguous()
    assert torch.equal(packed.permute(0,1,3,5,2,4,6).contiguous(),original)
    sf_layout=identity.get('variant',{}).get('weight_sf_layout','pair_major')
    if sf_layout=='n_major':
        # 纯排列消融：每N行连续SF，保持外部shape/位模式；不计入kernel Duration。
        scale=scale.transpose(-1,-2).contiguous().reshape(scale.shape)
    elif sf_layout!='pair_major':raise ValueError(f'Unknown scale layout={sf_layout}')
    setattr(case,attr,(packed.reshape(weight.shape),scale))
out.mkdir(exist_ok=True)
for _ in range(10):case.run_fused(timer=None)
torch.npu.synchronize();dist.barrier()
prefix=torch.zeros((DEFAULT_ALIGNMENT['rows'],DEFAULT_ALIGNMENT['cols']),dtype=torch.float32,device=case.y.device)
torch.exp(prefix,out=prefix);torch.npu.synchronize()
samples=collect(case.run_fused,torch,dist,prefix,20,out/f'rank_{rank}/profiler')
rows=[None]*8;dist.all_gather_object(rows,[s['duration_us'] for s in samples])
bits=case.y.view(torch.int16).cpu().numpy().tobytes()
output=dict(rank=rank,shape=list(case.y.shape),finite=bool(torch.isfinite(case.y).all().item()),
            sha256=hashlib.sha256(bits).hexdigest(),scope='DS bitwise differential; no independent Golden')
(out/f'output-rank{rank}.json').write_text(json.dumps(output,indent=2))
if not output['finite']:raise RuntimeError(f'case=EP8/BS{bs} rank={rank} stage=output nonfinite')
if rank==0:
    report=summarize(rows)
    report.update(ep=8,bs=bs,timer=False,accuracy='not_run_user_requested',identity=identity,
                  config=dict(hidden=7168,intermediate=3072,experts=384,topk=6,shared=0,input='fp8',weight='fp4',
                              routing='balanced',seed=20260922,weight_tile_n=tile_n,weight_tile_k=tile_k,
                              prepack_in_kernel_duration=False,jit_cache=os.environ['DG_JIT_CACHE_DIR']))
    (out/'performance.json').write_text(json.dumps(report,indent=2))
    print('RESULT',a.label,report['mean_us'],report['rank_max_mean_us'],flush=True)
if a.trace:
    import megamoe_timer as protocol
    if int(deep_gemm._C._mega_moe_timer_numel)!=protocol.TOTAL_BUFFER_SIZE:raise RuntimeError('Timer protocol mismatch')
    timer=torch.zeros(protocol.TOTAL_BUFFER_SIZE,dtype=torch.int64,device=case.y.device) if rank==0 else None
    for _ in range(10):case.run_fused(timer=timer)
    torch.npu.synchronize();dist.barrier()
    on=collect(case.run_fused,torch,dist,prefix,20,out/f'rank_{rank}/timer_on',timer=timer)
    on_rows=[None]*8;dist.all_gather_object(on_rows,[s['duration_us'] for s in on])
    if rank==0:
        events=protocol.decode(timer.cpu().numpy().reshape(-1),rank)
        (out/'timer_all_events.json').write_text(json.dumps(events))
        off_report=summarize(rows);on_report=summarize(on_rows,timer_pass=True)
        overhead=dict(off=off_report,on_rank0=on_report,overhead_pct=100*(on_report['mean_us']/off_report['mean_us']-1),
                      scope='every measured on invocation uses rank0 full passive timer; separate off/on CSV')
        (out/'timer_overhead.json').write_text(json.dumps(overhead,indent=2))
        print('TRACE_COMPLETE',a.label,overhead['overhead_pct'],flush=True)
dist.barrier();case.destroy();dist.destroy_process_group()
