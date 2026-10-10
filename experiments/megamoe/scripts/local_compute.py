"""Zero-communication functional smoke test of the original MegaMoE compute pipeline."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    import torch
    import torch_npu
    from deep_ep_experimental.megamoe import _C
    from deep_ep_experimental.megamoe.api import transform_weights_for_mega_moe
    from deep_ep_experimental.megamoe.reference import make_weights, quantize_input

    torch.npu.set_device(args.worker)
    e, h, intermediate, topk = 48, 7168, 3072, 6
    rows = args.bs * topk
    counts = torch.full((e,), rows // e, dtype=torch.int32)
    rng = np.random.default_rng(20261010 + args.worker)
    data, scales = quantize_input(rng.normal(0, .1, (rows, h)).astype(np.float32))
    dispatch = torch.from_numpy(data).npu()
    dispatch_sf = torch.from_numpy(scales).npu()
    example = torch.empty((args.bs, h), device="npu", dtype=torch.bfloat16)
    ids = torch.zeros((args.bs, topk), device="npu", dtype=torch.int32)
    probabilities = torch.full((args.bs, topk), 1 / topk, device="npu", dtype=torch.float32)
    raw = make_weights(e, h, intermediate, 20261010 + args.worker)

    def pair(value):
        w, sf = value
        return torch.from_numpy(w).npu(), torch.from_numpy(sf).npu().view(torch.float8_e8m0fnu)

    w1, w2 = transform_weights_for_mega_moe(pair(raw[0]), pair(raw[1]))
    w1 = w1.select(args.bs <= _C.small_batch_max_tokens)
    # 最后一个参数选择local-only构造；不调用SymmBuffer、torch.distributed或SHMEM初始化。
    plan = _C.Plan(example, intermediate, topk, e, 8, args.bs, True, 0, 8,
                   _C.local_ffts_config(), 0, 0, args.worker)
    torch.npu.synchronize()
    plan.prepare_local_compute(dispatch, dispatch_sf, counts)
    results = []
    previous = None
    for iteration in range(args.iters):
        begin = time.perf_counter()
        plan.forward(example, None, ids, probabilities, w1[0], w2[0], w1[1], w2[1],
                     None, None, None, None, None, None)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - begin
        actual = plan.local_compute_output().float().cpu()
        if not torch.isfinite(actual).all() or not torch.count_nonzero(actual):
            raise RuntimeError(f"Invalid local GMM2 output device={args.worker} iteration={iteration}")
        if previous is not None and not torch.equal(actual, previous):
            raise RuntimeError(f"Replay not repeatable device={args.worker} iteration={iteration}")
        previous = actual
        results.append(dict(iteration=iteration, host_wall_seconds=elapsed))
        print(f"[SMOKE] device={args.worker} iteration={iteration} finite/nonzero/repeatable", flush=True)
    report = dict(passed=True, device=args.worker, ep_shape=8, bs=args.bs, local_experts=e,
                  tokens_per_expert=rows//e, workspace_bytes=plan.workspace_bytes,
                  communication_initialized=False, input="synthetic locally pre-dispatched FP8",
                  validation="finite, nonzero, repeatability; NOT full numerical golden",
                  performance_valid=False, reason="functional smoke on shared devices; host wall is not kernel performance",
                  torch=torch.__version__, torch_npu=torch_npu.__version__, iterations=results)
    (args.output / f"device_{args.worker}.json").write_text(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, default=4096)
    p.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.bs < 1 or args.bs > 65536 or args.bs % 8 or args.iters < 2 or args.timeout < 1:
        p.error("BS must be in [8,65536] and divisible by 8; iters>=2 and timeout>0")
    if len(set(args.devices)) != len(args.devices) or any(d not in range(8) for d in args.devices):
        p.error("devices must be distinct IDs in [0,7]")
    if args.worker is not None:
        worker(args)
        return
    args.output = (args.output or Path("benchmark-runs") / datetime.now().strftime("local_compute_%Y%m%d_%H%M%S")).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"[OUTPUT] {args.output}", flush=True)
    procs, streams = [], []
    try:
        for device in args.devices:
            stream = (args.output / f"device_{device}.log").open("w")
            streams.append(stream)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--bs", str(args.bs),
                   "--iters", str(args.iters), "--output", str(args.output), "--worker", str(device)]
            procs.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")))
        deadline = time.monotonic() + args.timeout
        while any(proc.poll() is None for proc in procs):
            if any(proc.poll() not in (None, 0) for proc in procs):
                raise RuntimeError(f"Worker failed; inspect {args.output}/device_*.log")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Local compute timeout; inspect {args.output}")
            time.sleep(1)
        if any(proc.returncode for proc in procs):
            raise RuntimeError(f"Worker failed; inspect {args.output}")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for stream in streams:
            stream.close()
    reports = [json.loads((args.output / f"device_{d}.json").read_text()) for d in args.devices]
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2))
    print(f"[PASS] {len(reports)} independent devices; NO SHMEM/HCCL; NOT a performance result", flush=True)


if __name__ == "__main__":
    main()
