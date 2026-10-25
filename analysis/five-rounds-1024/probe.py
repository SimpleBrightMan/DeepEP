"""EP32初始化预检；不执行MegaMoE，不产生性能数字。"""
from datetime import timedelta
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
parser.add_argument('--method', choices=('control','combined'), default='combined')
parser.add_argument('--coordination', choices=('hccl','gloo'), default='gloo')
args = parser.parse_args()
if Path(args.label).name != args.label or args.label in ('.','..'):
    raise ValueError('Unsafe probe label')
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from coordinator import open_control_group
runtime = root/'packages'/args.method
sys.path.insert(0, str(runtime))
import torch
import torch_npu
import torch.distributed as dist
from deep_ep_experimental.megamoe import api

rank, local, world = (int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
if world not in (8,32): raise ValueError(f'EP8/32 probe got world={world}')
torch.set_num_threads(1)
faulthandler.dump_traceback_later(90,repeat=True)
print(f'DEVICE_INIT rank={rank}', flush=True)
torch.npu.set_device(local)
print(f'DEVICE_READY rank={rank}', flush=True)
if args.coordination=='hccl':
    dist.init_process_group('hccl', timeout=timedelta(seconds=120), device_id=torch.device(f'npu:{local}'))
    group=dist.distributed_c10d._get_default_group()
    value = torch.tensor([rank+1], device='npu', dtype=torch.float32)
    dist.all_reduce(value)
    control_sum=value.item()
else:
    group=open_control_group(api,rank,world,args.label)
    control_sum=sum(group.all_gather(rank+1,'probe_control_sum'))
if control_sum != world*(world+1)//2: raise RuntimeError(f'rank={rank}: control sum mismatch')
print(f'COORDINATOR_READY rank={rank} mode={args.coordination}', flush=True)
buffer = api.SymmBuffer(group, num_experts=384,
    num_max_tokens_per_rank=32, num_topk=6, hidden=7168, intermediate_hidden=3072,
    num_shared_experts=0, ranks_per_node=8)
print(f'BUFFER_READY rank={rank}', flush=True)
if args.coordination=='gloo':group.barrier('probe_all_buffers_ready')
buffer.destroy()
if args.coordination=='hccl':
    dist.barrier()
    dist.destroy_process_group()
else:
    group.close()
faulthandler.cancel_dump_traceback_later()
folder = root/'preflight'/args.label
folder.mkdir(parents=True,exist_ok=True)
with (folder/f'probe-rank{rank}.json').open('x') as output:
    output.write(json.dumps(dict(
    rank=rank, local_rank=local, world=world, control_sum=control_sum, buffer_destroyed=True,
    allreduce=control_sum if args.coordination=='hccl' else None,coordination=args.coordination,
    host_coordination='cpu_gloo' if args.coordination=='gloo' else 'hccl',
    torch=torch.__version__, torch_npu=torch_npu.__version__, protocol=api._extension().api_protocol,
    method=args.method)))
print(f'PROBE_PASS rank={rank}', flush=True)
