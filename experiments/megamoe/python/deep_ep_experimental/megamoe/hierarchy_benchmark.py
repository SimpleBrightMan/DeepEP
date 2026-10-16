# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""EP16 staged hierarchy prototype: full-output regression and complete call spans."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

from .hierarchy import expected_payload_bytes, stage_rates, summarize_measurements, validate_payload_metrics


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--routing", choices=("balanced", "random", "skewed"), default="balanced")
    parser.add_argument("--input", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative-rmse", type=float, default=0.01)
    parser.add_argument("--max-abs", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 384 <= args.bs <= 8192 or args.bs % 128:
        parser.error("BS must be a multiple of 128 in [384,8192]")
    if not 1024 <= args.hidden <= 8192 or args.hidden % 256:
        parser.error("hidden must be a multiple of 256 in [1024,8192]")
    if not 256 <= args.intermediate <= 4096 or args.intermediate % 128:
        parser.error("intermediate must be a multiple of 128 in [256,4096]")
    if args.experts % 16 or not 16 <= args.experts <= 2048 or not 1 <= args.topk <= min(32, args.experts):
        parser.error("Require E divisible by 16, local experts<=128 and K in [1,min(32,E)]")
    if args.warmup < 0 or args.iters < 1 or args.seed < 0:
        parser.error("Require warmup>=0, iters>=1 and seed>=0")
    if any(not math.isfinite(v) or v < 0 for v in (args.relative_rmse, args.max_abs)):
        parser.error("Accuracy tolerances must be finite and nonnegative")
    return args


def measure(call, group, torch, label, iterations, buffer=None):
    rows = []
    for iteration in range(iterations):
        torch.npu.synchronize()
        group.barrier(f"hierarchy timing {label} {iteration}")
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        start = time.perf_counter_ns()
        call()
        end.record()
        torch.npu.synchronize()
        rows.append(dict(acl_span_us=begin.elapsed_time(end) * 1000, wall_us=(time.perf_counter_ns() - start) / 1000))
        if buffer is not None:
            rows[-1]["stages"] = buffer.hierarchy_metrics
    return rows


def run(args, group, torch):
    import numpy as np
    from .api import SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe
    from .benchmark import inputs
    from .reference import error_metrics, make_weights, quantize_input
    from .validation import collective_check

    rank = group.rank()
    error = None
    try:
        if group.size() != 16 or int(os.environ["LOCAL_WORLD_SIZE"]) != 16:
            raise ValueError("Run all 16 ranks on the same UBX node")
        if any(
            os.environ.get(name)
            for name in ("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", "DEEPEP_MEGAMOE_COMBINE_INTRA_CORES")
        ):
            raise ValueError("Unset the other branch's core-sweep variables")
        rank_dir = args.output / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=False)
    except Exception as failure:
        error = str(failure)
    collective_check(group, "hierarchy runner", error, vars(args) | {"output": str(args.output)})
    model = dict(
        hidden=args.hidden,
        intermediate_hidden=args.intermediate,
        num_experts=args.experts,
        num_topk=args.topk,
        num_shared_experts=0,
    )
    config = dict(model=model, seed=args.seed)
    buffer = SymmBuffer(group, num_max_tokens_per_rank=args.bs, ranks_per_node=16, hierarchical=True, **model)

    def device_pair(pair):
        data, scale = pair
        return torch.from_numpy(data).npu(), torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu)

    raw = make_weights(args.experts // 16, args.hidden, args.intermediate, args.seed + rank)
    prepared = transform_weights_for_mega_moe(*map(device_pair, raw))
    x, ids, weights = inputs(config, rank, args.bs, args.routing, world=16)
    activation = None
    if args.input == "bf16":
        activation = torch.empty((args.bs, args.hidden), device="npu", dtype=torch.bfloat16)

    def set_inputs(values, route_weights):
        if activation is not None:
            activation.copy_(torch.from_numpy(values))
        else:
            codes, scales = quantize_input(values)
            buffer.x.copy_(torch.from_numpy(codes).npu().view(torch.float8_e4m3fn))
            buffer.x_sf.copy_(torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu))
        buffer.topk_weights.copy_(torch.from_numpy(route_weights))

    set_inputs(x, weights)
    buffer.topk_idx.copy_(torch.from_numpy(ids))
    buffer.validate_inputs(args.bs, x=activation)
    buffer.prepare(args.bs)
    y = torch.empty((args.bs, args.hidden), device="npu", dtype=torch.bfloat16)

    def call(stats=None):
        fp8_fp4_mega_moe(y, *prepared, buffer, x=activation, cumulative_local_expert_recv_stats=stats)

    for _ in range(args.warmup):
        call()
    baseline_counts = torch.zeros(args.experts // 16, device="npu", dtype=torch.int32)
    call(baseline_counts)
    baseline = y.float().cpu().numpy()
    expected_counts = baseline_counts.cpu().numpy()
    direct_rows = measure(call, group, torch, "direct", args.iters)
    direct = summarize_measurements(group.all_gather(direct_rows, "direct timing samples"))
    alternate_x, alternate_weights = np.roll(x, 1, axis=0).copy(), (-0.5 * weights).copy()
    set_inputs(alternate_x, alternate_weights)
    call()
    alternate_baseline = y.float().cpu().numpy()
    set_inputs(x, weights)
    global_ids = buffer.configure_hierarchy()
    oracle = expected_payload_bytes(global_ids, args.experts, args.hidden, rank)
    configure_wall = buffer.hierarchy_plan_wall_ms
    error = None
    try:
        validate_payload_metrics(buffer.hierarchy_metrics, oracle)
    except Exception as failure:
        error = str(failure)
    collective_check(group, "hierarchy payload oracle", error)
    actual_counts = torch.zeros_like(baseline_counts)
    call(actual_counts)
    actual = y.float().cpu().numpy()
    accuracy = error_metrics(actual, baseline)
    accuracy.update(
        rank=rank,
        compared_elements=int(actual.size),
        counts_equal=bool(np.array_equal(actual_counts.cpu().numpy(), expected_counts)),
    )
    accuracy["passed"] = bool(
        accuracy["finite"]
        and accuracy["counts_equal"]
        and accuracy["relative_rmse"] <= args.relative_rmse
        and accuracy["max_abs"] <= args.max_abs
    )
    checks = group.all_gather(accuracy, "hierarchy full output regression")
    (rank_dir / "accuracy.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(item["passed"] for item in checks):
        raise AssertionError("Hierarchy output/counts failed direct-path regression; inspect accuracy.json")
    # Fixed routes must still consume new activations and signed weights every call.
    set_inputs(alternate_x, alternate_weights)
    call()
    dynamic = error_metrics(y.float().cpu().numpy(), alternate_baseline)
    dynamic.update(rank=rank, compared_elements=int(actual.size))
    dynamic["passed"] = bool(
        dynamic["finite"] and dynamic["relative_rmse"] <= args.relative_rmse and dynamic["max_abs"] <= args.max_abs
    )
    dynamic_checks = group.all_gather(dynamic, "hierarchy changed input/weight regression")
    (rank_dir / "dynamic_accuracy.json").write_text(json.dumps(dynamic_checks, indent=2), encoding="utf-8")
    if not all(item["passed"] for item in dynamic_checks):
        raise AssertionError("Hierarchy failed changed activation/weight regression")
    set_inputs(x, weights)
    for _ in range(args.warmup):
        call()
    hierarchy_rows = measure(call, group, torch, "hierarchy", args.iters, buffer)
    # Re-check after repeated launches to catch stale cache/epoch bugs.
    repeat = error_metrics(y.float().cpu().numpy(), baseline)
    repeat["passed"] = bool(
        repeat["finite"] and repeat["relative_rmse"] <= args.relative_rmse and repeat["max_abs"] <= args.max_abs
    )
    repeats = group.all_gather(repeat, "hierarchy repeated output regression")
    if not all(item["passed"] for item in repeats):
        raise AssertionError("Repeated hierarchy calls changed the result")
    per_rank = group.all_gather(hierarchy_rows, "hierarchy timing samples")
    hierarchy = summarize_measurements(per_rank)
    plan_times = group.all_gather(configure_wall, "hierarchy plan time")
    local = dict(
        rank=rank,
        oracle=oracle,
        direct_samples=direct_rows,
        hierarchy_samples=hierarchy_rows,
        plan_wall_ms=configure_wall,
        resources=dict(workspace_bytes=buffer.workspace_bytes, symmetric_buffer_bytes=buffer.symmetric_buffer_bytes),
    )
    (rank_dir / "results.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
    all_rates = [stage_rates(row["stages"]) for rows in per_rank for row in rows]
    rates = {
        name: dict(
            mean=statistics.mean(row[name] for row in all_rates),
            minimum=min(row[name] for row in all_rates),
            maximum=max(row[name] for row in all_rates),
        )
        for name in all_rates[0]
    }
    report = dict(
        schema_version=1,
        prototype="staged_hierarchical_dedup_fp32_partial",
        config=vars(args) | {"output": str(args.output)},
        direct=direct,
        hierarchy=hierarchy,
        accuracy=checks,
        repeat_accuracy=repeats,
        dynamic_accuracy=dynamic_checks,
        plan_wall_ms=plan_times,
        stage_payload_gbps=rates,
        timer=False,
        bandwidth_scope="payload / synchronous stage wall time (includes synchronization and reduction)",
        total_scope="ACL event span and host wall include all kernels, stage gaps and Python collective preflight",
        rebuild_every_call_estimate_us=max(plan_times) * 1000 + hierarchy["wall_us"]["rank_max_mean"],
    )
    if rank == 0:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[ACCURACY] BS={args.bs} full outputs and counts PASSED on 16 ranks", flush=True)
        print(
            f"[TOTAL] direct ACL-span slow-rank mean={direct['acl_span_us']['rank_max_mean']:.3f}us; "
            f"hierarchy={hierarchy['acl_span_us']['rank_max_mean']:.3f}us; "
            f"hierarchy host-wall={hierarchy['wall_us']['rank_max_mean']:.3f}us",
            flush=True,
        )
        print(
            f"[PLAN] max-rank={max(plan_times):.3f}ms, fixed-route setup; "
            f"rebuild+forward estimate={report['rebuild_every_call_estimate_us']:.3f}us",
            flush=True,
        )
        for name, value in rates.items():
            print(
                f"[STAGE] {name} effective-payload GB/s/card mean={value['mean']:.3f} "
                f"min/max={value['minimum']:.3f}/{value['maximum']:.3f} "
                "(synchronous stage, not pure link bandwidth)",
                flush=True,
            )
        print(f"[PASSED] {args.output / 'summary.json'}", flush=True)
    group.barrier("hierarchy benchmark teardown")
    buffer.destroy()


def main(argv=None):
    args = arguments(argv)
    import torch
    import torch_npu  # noqa: F401
    from .api import StoreGroup

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env()
    try:
        run(args, group, torch)
    except Exception as error:
        group.abort(error)
        raise
    group.close()


if __name__ == "__main__":
    main()
