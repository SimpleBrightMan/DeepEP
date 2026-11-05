"""Profile the existing MegaMoE entry truncated immediately after RoutePrepare.

This requires the laboratory native build with ``urma_route_only=True``. It
does not time or validate FFN output. Original workspace allocation, caller /
owner stream ordering, epoch progression, and launch alignment are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs-values", default="8192")
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--intermediate", type=int, default=2304)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--cv-rows", type=int, default=256)
    p.add_argument("--gmm1-ahead-mgroups", type=int, default=12)
    p.add_argument("--routing", choices=("random-balanced", "random", "balanced", "hot", "duplicates"), default="random-balanced")
    p.add_argument("--check-patterns", default="", help="Extra untimed routing-only checks, e.g. random,hot,duplicates")
    p.add_argument("--ids-dtype", choices=("int32", "int64"), default="int64")
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--exp-repeat", type=int, default=50)
    p.add_argument("--exp-shape", type=int, nargs=2, default=(100000, 20000))
    p.add_argument("--launch-timeout", type=int, default=240)
    p.add_argument("--allow-multi-host", action="store_true")
    p.add_argument("--timer", action="store_true", help="One final coarse trace; excluded from formal profile samples")
    p.add_argument("--timer-clock-mhz", type=float, default=1000.0)
    p.add_argument("--variant", default="baseline")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    args.bs_values = list(map(int, args.bs_values.split(",")))
    args.check_patterns = [v for v in args.check_patterns.split(",") if v]
    if not args.bs_values or len(set(args.bs_values)) != len(args.bs_values) or any(not 1 <= b <= 16384 for b in args.bs_values):
        p.error("--bs-values must be unique integers in [1,16384]")
    if args.warmup < 0 or args.iters < 1 or args.exp_repeat < 0:
        p.error("warmup/exp-repeat must be nonnegative and iters positive")
    if args.experts != 384 or not 1 <= args.topk <= 8:
        p.error("the current native contract is E384, TopK1..8")
    if any(v not in ("random-balanced", "random", "balanced", "hot", "duplicates") for v in args.check_patterns):
        p.error("unknown --check-patterns entry")
    return args


def make_ids(bs, topk, world, rank, routing, seed, dtype):
    import numpy as np
    if routing == "random-balanced":
        from .random_balanced import random_balanced_indices
        ids = random_balanced_indices(bs, world, 384, topk, rank)
    elif routing == "balanced":
        linear = np.arange(rank * bs * topk, (rank + 1) * bs * topk)
        ids = ((linear % world) * (384 // world) + (linear // world) % (384 // world)).reshape(bs, topk)
    elif routing == "hot":
        # Only rank0's experts receive data; other ranks have empty receive plans.
        ids = np.random.default_rng(seed + rank).integers(0, 384 // world, (bs, topk))
    elif routing == "duplicates":
        # All local route entries belong to one expert. This intentionally
        # exercises legal repeated IDs without assuming token-level uniqueness.
        ids = np.full((bs, topk), rank % 384)
    else:
        ids = np.random.default_rng(seed + rank).integers(0, 384, (bs, topk))
    return np.asarray(ids, dtype=dtype)


def validate_routes(prepared, host_ids, source_counts, rank, world, max_bs, cv_rows):
    """Strict CPU oracle, called only after NPU completion and outside profiling."""
    import numpy as np
    flat = host_ids.reshape(-1).astype(np.int64)
    bs, topk = host_ids.shape
    count = np.bincount(flat, minlength=384)
    offsets = np.r_[0, count.cumsum()]
    expected_order = np.argsort(flat, kind="stable")
    expected_inverse = np.empty_like(expected_order)
    expected_inverse[expected_order] = np.arange(len(flat))
    checks = {}
    for field, expected in (("expert_counts", count), ("expert_offsets", offsets),
                            ("order", expected_order), ("reduce_index", expected_inverse.reshape(bs, topk))):
        actual = getattr(prepared.routing, field).cpu().numpy()
        checks[field] = bool(np.array_equal(actual, expected))
    counts = np.asarray(source_counts, dtype=np.int64)
    local = 384 // world
    received = counts[:, rank * local:(rank + 1) * local].sum(axis=0)
    group_counts = (received + 255) // 256
    group_base = np.r_[0, group_counts.cumsum()]
    max_cvs = (max_bs * topk + cv_rows - 1) // cv_rows
    expected_deps = np.empty(len(flat), dtype=np.uint64)
    for expert in range(384):
        first, end = map(int, offsets[expert:expert + 2])
        if first == end:
            continue
        peer = expert // local
        positions = expected_order[first:end]
        expanded = np.arange(first, end, dtype=np.int64)
        if peer == rank:
            source_prefix = int(counts[:rank, expert].sum())
            group = group_base[expert % local] + (source_prefix + expanded - first) // 256
            expected_deps[positions] = np.uint64(1 << 63) | group.astype(np.uint64)
        else:
            peer_base = offsets[peer * local]
            expected_deps[positions] = (peer * max_cvs + (expanded - peer_base) // cv_rows).astype(np.uint64)
    actual_deps = prepared.native.route_dependency_tensor(bs).cpu().numpy().reshape(-1).view(np.uint64)
    checks["reduce_dependencies"] = bool(np.array_equal(actual_deps, expected_deps))
    stats = list(map(int, prepared.native.route_stats()))
    checks["received_rows"] = stats[0] == int(received.sum())
    checks["m_groups"] = stats[1] == int(group_counts.sum())
    return dict(rank=rank, passed=all(checks.values()), checks=checks,
                received_rows=stats[0], expected_received_rows=int(received.sum()),
                m_groups=stats[1], segments=stats[2], links=stats[3])


def make_unused_weights(torch, torch_npu, local_experts, hidden, intermediate):
    """Real tensors satisfying the original ABI; never read by the route-only kernel.

    Constant nonzero FP4 bytes have identical planar/interleaved layouts, so
    one NZ allocation can serve both GMM1 views. This is setup, outside timing.
    """
    from .api import _RoutedWeightPair
    def pair(experts, n, k):
        raw = torch.full((experts, n, k // 2), 0x11, dtype=torch.uint8, device="npu")
        weight = torch_npu.npu_format_cast(raw, 29, customize_dtype=torch.float8_e4m3fn,
                                         input_dtype=torch_npu.float4_e2m1fn_x2)
        scale = torch.full((experts, n, k // 64, 2), 127, dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)
        return weight, scale
    first = pair(local_experts, 2 * intermediate, hidden)
    second = pair(local_experts, hidden, intermediate)
    return _RoutedWeightPair(first, first), second


def run(args, group):
    import numpy as np
    import torch
    import torch_npu
    from .urma import UrmaBuffer, _extension
    from .urma_timing import UrmaLaunchGuard, measure_profiled_forward, summarize_profile_rounds
    from .validation import collective_check
    rank, world = group.rank(), group.size()
    if not getattr(_extension(), "urma_route_only", False):
        raise RuntimeError("This runner requires the route-only laboratory wheel; refuse a full MegaMoE kernel")
    if world not in (4, 8, 16):
        raise ValueError("route-only runner requires EP4/8/16")
    max_bs = max(args.bs_values)
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    rank_dir = args.output_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    owner = UrmaBuffer(group, hidden=args.hidden, intermediate_hidden=args.intermediate,
                      num_experts=384, num_topk=args.topk, num_shared_experts=0,
                      num_max_tokens_per_rank=max_bs, cv_rows=args.cv_rows,
                      gmm1_ahead_mgroups=args.gmm1_ahead_mgroups,
                      allow_multi_host=args.allow_multi_host)
    weights = error = None
    try:
        weights = make_unused_weights(torch, torch_npu, 384 // world, args.hidden, args.intermediate)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    collective_check(group, "route-only ABI input allocation", error)
    guard = UrmaLaunchGuard(group, args.exp_shape, args.exp_repeat, args.launch_timeout, backend="hccl")
    guard.prepare_per_call_alignment()
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    summary = dict(scope="route_prepare_only", variant=args.variant, world=world,
                   created_utc=datetime.now(timezone.utc).isoformat(), config=config,
                   launch_guard=guard.describe(), numerical_validation="routing_only_not_FFN",
                   allocation_scope="unchanged full UrmaPlan capacity buffers; unused FFN operand tensors retained",
                   cases=[], passed=True)
    (rank_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for bs in args.bs_values:
        case = rank_dir / f"bs_{bs}"
        case.mkdir(parents=True, exist_ok=True)
        host_ids = make_ids(bs, args.topk, world, rank, args.routing, args.seed, args.ids_dtype)
        ids = torch.from_numpy(host_ids).npu()
        prepared = owner.prepare_routes(ids)
        data = torch.full((bs, args.hidden), 0x38, dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)
        scales = torch.full((bs, args.hidden // 32), 127, dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)
        probs = torch.full((bs, args.topk), 1.0 / args.topk, dtype=torch.float32, device="npu")
        output = torch.empty((bs, args.hidden), dtype=torch.bfloat16, device="npu")
        forward = partial(owner.forward_prepared, (data, scales), probs, *weights, prepared, out=output)
        trace = owner.allocate_trace(16) if args.timer else None
        coarse = partial(forward, trace=trace, coarse_trace=True) if args.timer else None
        torch.npu.synchronize()
        capture = measure_profiled_forward(forward, warmup=args.warmup, iters=args.iters, guard=guard,
                                          directory=case / "profile", coarse_forward=coarse)
        records = group.all_gather(capture["durations_us"], f"route-only profile BS={bs}")
        profile = summarize_profile_rounds(records, ranks_per_node=ranks_per_node)
        profile.update(scope="route_prepare_only", metric="RoutePrepare-only device kernel duration us")
        report = dict(bs=bs, routing=args.routing, profile=profile, profile_directory=capture["profile_dir"],
                      diagnostics=capture["diagnostics"], extra_validation=[])
        local_counts = np.bincount(host_ids.reshape(-1), minlength=384).tolist()
        source_counts = group.all_gather(local_counts, f"route-only reference counts BS={bs}")
        result = validate_routes(prepared, host_ids, source_counts, rank, world, max_bs, args.cv_rows)
        validation = group.all_gather(result, f"route-only validation BS={bs}")
        report["route_validation"] = dict(passed=all(row["passed"] for row in validation), per_rank=validation)
        if args.timer:
            from .urma_benchmark import export_trace_collective
            from .urma_trace_merge import merge_trace_collective
            (case / "coarse").mkdir(parents=True, exist_ok=True)
            model = dict(hidden=args.hidden, intermediate_hidden=args.intermediate, num_experts=384,
                         num_topk=args.topk, num_shared_experts=0)
            trace_records = export_trace_collective(group, trace.cpu().numpy(), case / "coarse", bs=bs,
                clock_mhz=args.timer_clock_mhz, include_waits=True, model=model)
            report["coarse_trace"] = merge_trace_collective(group, trace_records,
                args.output_dir / f"bs_{bs}" / "coarse" / "allrank.json", bs=bs, include_waits=True)
        for pattern in args.check_patterns:
            alternate = make_ids(bs, args.topk, world, rank, pattern, args.seed + 111, args.ids_dtype)
            alternate_ids = torch.from_numpy(alternate).npu()
            alternate_prepared = owner.prepare_routes(alternate_ids)
            guard.before_forward()
            owner.forward_prepared((data, scales), probs, *weights, alternate_prepared, out=output)
            torch.npu.synchronize()
            counts = group.all_gather(np.bincount(alternate.reshape(-1), minlength=384).tolist(),
                                      f"route-only extra counts {bs}/{pattern}")
            check = validate_routes(alternate_prepared, alternate, counts, rank, world, max_bs, args.cv_rows)
            checks = group.all_gather(check, f"route-only extra check {bs}/{pattern}")
            report["extra_validation"].append(dict(pattern=pattern, passed=all(v["passed"] for v in checks), per_rank=checks))
        report["passed"] = report["route_validation"]["passed"] and all(v["passed"] for v in report["extra_validation"])
        summary["passed"] &= report["passed"]
        summary["cases"].append(report)
        (case / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if rank == 0:
            print(f"[ROUTE_ONLY] variant={args.variant} BS={bs} EP={world} K={args.topk} "
                  f"round-max-mean={profile['rank_max_mean_us']:.3f}us "
                  f"round-min-mean={profile['rank_min_mean_us']:.3f}us "
                  f"routing={'PASS' if report['passed'] else 'FAIL'} (FFN skipped)", flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    guard.close()
    owner.destroy()
    group.barrier("route-only complete")
    if not summary["passed"]:
        raise RuntimeError("Route-only output mismatch; inspect per-rank validation reports")


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torch_npu  # noqa: F401
    from deep_ep._store import StoreGroup
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env(timeout=args.launch_timeout)
    try:
        run(args, group)
    except BaseException as error:
        group.abort(f"route-only benchmark: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
