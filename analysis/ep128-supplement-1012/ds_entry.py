"""EP128 upstream compute/communication, external torchrun performance harness."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
import argparse
from datetime import timedelta

sys.path[:0] = [str(Path(__file__).resolve().parent/'ds-python'), str(Path(__file__).resolve().parent/'ds-harness')]
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import deep_gemm
from deep_gemm.testing.envs import init_seed
from test_mega_moe import MegaMoECase
from megamoe_measure import collect, DEFAULT_ALIGNMENT


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('/home/zyx/ep128-ds-20261011-run1'))
    parser.add_argument('--bs-values', type=int, nargs='+', default=[512,2048])
    args = parser.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world in (8, 32, 128)
    torch.npu.set_device(local)
    dist.init_process_group('hccl', init_method='env://', timeout=timedelta(minutes=10),
                            device_id=torch.device(f'npu:{local}'))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('npu')
    init_seed(20260922)
    group = dist.distributed_c10d._get_default_group()
    root = args.output_dir
    for bs in args.bs_values:
        out = root / f'bs_{bs}'
        rank_dir = out / f'rank_{rank}'
        rank_dir.mkdir(parents=True, exist_ok=False)
        print(f'[CASE] rank={rank} bs={bs}', flush=True)
        case = MegaMoECase(group, 7168, 3072, 384, 6, bs, 0,
                          activation_clamp=float('inf'), with_baseline=False)
        case.create_inputs(bs)
        routes = np.arange(rank * bs * 6, (rank + 1) * bs * 6, dtype=np.int64)
        ids = ((routes % world) * (384 // world) + (routes // world) % (384 // world)).reshape(bs, 6)
        case.buffer.topk_idx[:bs].copy_(torch.from_numpy(ids).to(case.y.device))
        for _ in range(10):
            case.run_fused()
        torch.npu.synchronize()
        prefix = torch.zeros((DEFAULT_ALIGNMENT['rows'], DEFAULT_ALIGNMENT['cols']),
                             dtype=torch.float32, device=case.y.device)
        samples = collect(case.run_fused, torch, dist, prefix, 20, rank_dir / 'profiler')
        save(rank_dir / 'kernel_samples.json', samples)
        per_rank = [None] * world
        dist.all_gather_object(per_rank, [s['duration_us'] for s in samples])
        assert all(len(row) == 20 and all(math.isfinite(x) and x > 0 for x in row) for row in per_rank)
        rows = [row[10:] for row in per_rank]
        report = dict(ep=world, bs=bs, hidden=7168, intermediate_hidden=3072,
                      experts=384, topk=6, shared_experts=0, routing='balanced_rank_first',
                      upstream_commit='8491bbb4b8c02a094a2318965f50c70438a3e73c',
                      package_commit='253b474', timer=False, accuracy='not_run',
                      metric='megamoe_kernel_duration_us', recorded_rounds=20, rounds=10,
                      aggregation='mean_of_per_round_min_rank',
                      mean_us=statistics.mean(min(r) for r in zip(*rows)),
                      rank_max_mean_us=statistics.mean(max(r) for r in zip(*rows)),
                      per_rank_rounds_us=rows, alignment=DEFAULT_ALIGNMENT,
                      scope='upstream FP8 input; excludes prelaunch memset and host work')
        if rank == 0:
            save(out / 'profiling.json', report)
            print(f'[PERF] bs={bs} min={report["mean_us"]:.4f} max={report["rank_max_mean_us"]:.4f}', flush=True)
        del prefix
        case.destroy()
        del case
        torch.npu.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
