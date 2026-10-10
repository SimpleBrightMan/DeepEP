"""Zero-communication MegaMoE validation, optional kernel profiling and separate tracing."""
import argparse
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import subprocess
import statistics
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
    soc = torch.npu.get_device_name(args.worker)
    if "950" not in soc:
        raise RuntimeError(f"Local compute currently requires Ascend950, got {soc}")
    e, h, intermediate, topk = 48, 7168, 3072, 6
    rows = args.bs * topk
    print(f"[PREP] device={args.worker} BS={args.bs} local_experts={e} rows={rows}; local input/weights", flush=True)
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
    # dav-c310的util_set_ffts_config为空操作，与Ascend950 SHMEM自身使用0的约定一致。
    plan = _C.Plan(example, intermediate, topk, e, 8, args.bs, True, 0, 8,
                   0, 0, 0, args.worker)
    torch.npu.synchronize()
    plan.prepare_local_compute(dispatch, dispatch_sf, counts)
    print(f"[READY] device={args.worker} workspace_bytes={plan.workspace_bytes}; no communication", flush=True)
    results = []
    previous = None
    def call(timer=None):
        plan.forward(example, None, ids, probabilities, w1[0], w2[0], w1[1], w2[1],
                     timer, None, None, None, None, None)

    for iteration in range(args.iters):
        begin = time.perf_counter()
        call()
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
    output_hash = hashlib.sha256(memoryview(previous.numpy())).hexdigest()
    if args.reference_dir:
        reference = json.loads((args.reference_dir / f"device_{args.worker}.json").read_text())
        if (reference["bs"], reference["device"], reference["seed"]) != (args.bs, args.worker, 20261010 + args.worker):
            raise RuntimeError("Reference shape/device/seed does not match this run")
        if reference["output_sha256"] != output_hash:
            raise RuntimeError(f"Full output differs from reference device={args.worker} BS={args.bs}")
    measurements = []
    if args.profile:
        from deep_ep_experimental.megamoe.performance import read_kernel_samples
        for _ in range(args.warmup):
            call()
        torch.npu.synchronize()
        # 仅在批次前排入小型计算，避免host来不及提交；不计入MegaMoE kernel Duration。
        prefix = torch.zeros(16 * 1024 * 1024, device="npu", dtype=torch.float32)
        profiler = torch_npu.profiler
        for repeat in range(args.profile_rounds):
            directory = args.output / f"profile_device_{args.worker}_round_{repeat}"
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=profiler._ExperimentalConfig(
                    profiler_level=profiler.ProfilerLevel.Level0,
                    export_type=profiler.ExportType.Text,
                    data_simplification=False)):
                for _ in range(8):
                    torch.exp(prefix, out=prefix)
                for _ in range(args.profile_iters):
                    call()
                torch.npu.synchronize()
            samples = read_kernel_samples(directory, args.profile_iters)
            durations = [sample["duration_us"] for sample in samples]
            measured = durations[len(durations) // 2:]
            measurements.append(dict(round=repeat, samples=samples,
                mean_us=statistics.mean(measured), median_us=statistics.median(measured),
                min_us=min(measured), max_us=max(measured), selection="tail_half_ceil"))
            print(f"[KERNEL] device={args.worker} round={repeat} mean={measurements[-1]['mean_us']:.3f}us timer=off", flush=True)
        actual = plan.local_compute_output().float().cpu()
        if not torch.equal(actual, previous):
            raise RuntimeError(f"Output changed after profiled repetitions device={args.worker}")
    trace_report = None
    if args.trace:
        from deep_ep_experimental.megamoe.timer.breakdown import compact_events, rank_breakdown
        from deep_ep_experimental.megamoe.timer.trace_parser import trace_events, first_start_cycle
        if not _C.timer_enabled:
            raise RuntimeError("Installed wheel was built without timer")
        timer = torch.zeros(_C.timer_numel, dtype=torch.int64, device="npu")
        call(timer)
        torch.npu.synchronize()
        events = compact_events(timer.cpu().numpy(), args.worker)
        # 利用设备已写入的坐标，不增加读时钟/同步；保留原始区间供边界分析。
        from deep_ep_experimental.megamoe.timer import protocol as protocol
        from deep_ep_experimental.megamoe.timer.parse_timer_csv import core_info
        raw = timer.cpu().numpy()
        cores = {core_info(core): core for core in range(protocol.N_CORE_COUNT)}
        for event in events:
            family, _, index = event["name"].rpartition("_")
            if family not in protocol.DYNAMIC_NAMES or not index.isdecimal():
                continue
            core = cores[(event["group_id"], event["core_type"], event["sub_id"])]
            slot = len(protocol.FIXED_NAMES) + protocol.DYNAMIC_NAMES.index(family) * protocol.MAX_DYNAMIC_ITER + int(index)
            event.update(protocol.task_metadata(raw, core, slot))
            event["wave_id"] = int(raw[protocol.WAVE_ID_OFFSET + core * protocol.WAVE_ID_PER_CORE + slot])
        (args.output / f"device_{args.worker}_events.json").write_text(json.dumps(events, separators=(",", ":")))
        trace_report = rank_breakdown(events, args.worker, compute_replay=True)
        if trace_report["status"] != "ok":
            raise RuntimeError(f"Invalid compute-only timer trace: {trace_report}")
        if not torch.equal(plan.local_compute_output().float().cpu(), previous):
            raise RuntimeError("Timer pass changed the output")
        trace = dict(traceEvents=trace_events(events, "keep", first_start_cycle(events), time_origin="rank"),
                     displayTimeUnit="us", scope="local_compute", gmm2_semantics=trace_report["gmm2_semantics"])
        (args.output / f"device_{args.worker}_timer.json").write_text(json.dumps(trace, separators=(",", ":")))
        print(f"[TRACE] device={args.worker} events={len(events)} {trace_report['gmm2_semantics']}", flush=True)
    report = dict(passed=True, device=args.worker, ep_shape=8, bs=args.bs, local_experts=e,
                  tokens_per_expert=rows//e, workspace_bytes=plan.workspace_bytes,
                  communication_initialized=False, input="synthetic locally pre-dispatched FP8",
                  validation="finite, nonzero, repeatability; NOT full numerical golden",
                  performance_valid=False, reason="device exclusivity not automatically verified; host wall is not kernel Duration",
                  torch=torch.__version__, torch_npu=torch_npu.__version__, device_name=soc,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=Path(__file__).resolve().parents[3], text=True).strip(),
                  seed=20261010 + args.worker, timer=False, iterations=results,
                  output_sha256=output_hash, metric="megamoe_compute_replay_kernel_duration_us",
                  kernel_measurements=measurements, warmup=args.warmup,
                  profile_iters=args.profile_iters, profile_rounds=args.profile_rounds,
                  separate_trace=trace_report)
    binary = Path(_C.__file__).with_name("libmegamoe_kernel.so")
    with binary.open("rb") as stream:
        report["kernel_binary_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    report["reference_dir"] = str(args.reference_dir) if args.reference_dir else None
    (args.output / f"device_{args.worker}.json").write_text(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, default=4096)
    p.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--profile", action="store_true", help="Collect timer-off device kernel Duration; requires exclusive devices for performance claims")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--profile-iters", type=int, default=50)
    p.add_argument("--profile-rounds", type=int, default=3)
    p.add_argument("--reference-dir", type=Path, help="Require full output fingerprints to match a prior identical case")
    p.add_argument("--trace", action="store_true", help="Collect one separate diagnostic trace AFTER all timer-off samples")
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.bs < 1 or args.bs > 65536 or args.bs % 8 or args.iters < 2 or args.timeout < 1:
        p.error("BS must be in [8,65536] and divisible by 8; iters>=2 and timeout>0")
    if len(set(args.devices)) != len(args.devices) or any(d not in range(8) for d in args.devices):
        p.error("devices must be distinct IDs in [0,7]")
    if args.warmup < 1 or args.profile_iters < 2 or args.profile_rounds < 1:
        p.error("warmup>=1, profile-iters>=2, profile-rounds>=1 required")
    if args.reference_dir:
        args.reference_dir = args.reference_dir.resolve()
        if not args.reference_dir.is_dir():
            p.error("reference-dir must exist")
    if args.worker is not None:
        worker(args)
        return
    args.output = (args.output or Path("benchmark-runs") / datetime.now().strftime("local_compute_%Y%m%d_%H%M%S")).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"[OUTPUT] {args.output}", flush=True)
    if args.profile:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=True)
        (args.output / "npu_before.txt").write_text(snapshot.stdout)
    procs, streams = [], []
    try:
        for device in args.devices:
            stream = (args.output / f"device_{device}.log").open("w")
            streams.append(stream)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--bs", str(args.bs),
                   "--iters", str(args.iters), "--output", str(args.output), "--worker", str(device)]
            if args.profile:
                cmd += ["--profile", "--warmup", str(args.warmup), "--profile-iters", str(args.profile_iters),
                        "--profile-rounds", str(args.profile_rounds)]
            if args.reference_dir:
                cmd += ["--reference-dir", str(args.reference_dir)]
            if args.trace:
                cmd += ["--trace"]
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
    if args.profile:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=True)
        (args.output / "npu_after.txt").write_text(snapshot.stdout)
    print(f"[PASS] {len(reports)} independent devices; NO SHMEM/HCCL; "
          + ("kernel Duration collected, check device exclusivity" if args.profile else "functional smoke only"), flush=True)


if __name__ == "__main__":
    main()
