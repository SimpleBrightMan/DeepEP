"""Exercise host metadata transactions without importing torch_npu or opening NPU."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import package,write
from coordinator import open_control_group
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
package('/home/codex-deepep/gm-readable-slice-prepared-a-1027')
from deep_ep_experimental.megamoe import api
import torch
torch.set_num_threads(1)
rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
group=open_control_group(api,rank,world,a.label)
for index in range(24):
    value=dict(rank=rank,index=index,values=list(range(10+index*13)),message='metadata'*(index+1))
    peers=group.all_gather(value,f'cpu-probe/{index}')
    if len(peers)!=world or any(p['rank']!=r or p['index']!=index or p['values']!=value['values']
            for r,p in enumerate(peers)):raise RuntimeError('CPU coordinator payload mismatch')
    group.barrier(f'cpu-probe/{index}/checked')
group.close()
write(Path(__file__).resolve().parent/a.label/f'rank_{rank}.json',dict(rank=rank,world=world,passed=True,
    gathers=24,barriers=24,npu_initialized=False))
print(f'CPU_PROBE_PASS rank={rank}',flush=True)
