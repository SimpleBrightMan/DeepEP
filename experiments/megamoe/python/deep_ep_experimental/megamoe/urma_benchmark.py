"""Full URMA MegaMoE benchmark; EP4/EP8/EP16, with explicit opt-in for multiple hosts."""

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import json
import math
import os
from pathlib import Path

from .benchmark import inputs, load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="urma")
    parser.add_argument(
        "--bs-values",
        default="1024,2048,3072,4096,5120,6144,7168,8192",
        help="Comma-separated BS/rank sweep; each BS must be in [1, num_max_tokens_per_rank] (native capacity <= 8192)",
    )
    parser.add_argument("--routing", choices=("balanced", "random", "skewed", "random-balanced"), default="balanced")
    parser.add_argument("--cv-rows", type=int, default=256)
    parser.add_argument(
        "--gmm1-ahead-mgroups",
        type=int,
        default=12,
        help="Sustained pending GMM1 M-group window; 0 preserves the two-expert baseline",
    )
    parser.add_argument(
        "--exp-shape",
        type=int,
        nargs=2,
        default=(100000, 20000),
        metavar=("ROWS", "COLS"),
        help="FP32 launch-guard tensor shape; two buffers, default 14.90 GiB/rank (same as mega_test)",
    )
    parser.add_argument(
        "--exp-repeat",
        type=int,
        default=50,
        help="Exp repetitions once before each continuous profile block; 0 retains only HCCL barrier",
    )
    parser.add_argument("--launch-timeout", type=int, default=300, help="Benchmark HCCL group timeout in seconds")
    parser.add_argument(
        "--guard-backend",
        choices=("hccl", "store"),
        default="hccl",
        help="Launch-alignment barrier: HCCL device barrier (default) or host Store barrier where HCCL cannot initialize",
    )
    parser.add_argument(
        "--allow-multi-host",
        action="store_true",
        help="Explicitly allow EP4/EP8/EP16 across hosts; requires cross-host SHMEM mapping and URMA connectivity",
    )
    parser.add_argument(
        "--max-recv-rows", type=int, help="Per-rank capacity; default 2 * max(BS) * topk; excessive skew is rejected"
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timer", action="store_true", help="Append one coarse diagnostic after formal rounds")
    parser.add_argument("--trace", action="store_true", help="Legacy alias for --fine (also enables coarse)")
    parser.add_argument(
        "--trace-include-waits",
        action="store_true",
        help="Include dependency waits in exported JSON; default hides them but retains final rank synchronization",
    )
    parser.add_argument(
        "--trace-capacity",
        type=int,
        default=4096,
        help="Maximum detailed records per physical engine; overflow fails explicitly",
    )
    parser.add_argument(
        "--timer-clock-mhz",
        type=float,
        default=1000.0,
        help="Counter ticks/us; default follows legacy MegaMoE protocol, not hardware detection",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        help="Prebuilt FP8 CPU Golden for URMA/NPU-cascade error budgets; absent skips accuracy",
    )
    parser.add_argument(
        "--accuracy-only", action="store_true", help="Run forwards and dual-reference accuracy without profiler"
    )
    parser.add_argument(
        "--debug-intermediates",
        action="store_true",
        help="Copy completed intermediates to CPU; accuracy-only diagnosis",
    )
    parser.add_argument(
        "--check-tokens", type=int, default=0, help="Must match --check-tokens used to create the optional Golden cache"
    )
    parser.add_argument("--fine", action="store_true", help="Append one detailed diagnostic after coarse; implies --timer")
    args = parser.parse_args(argv)
    if args.guard_backend != "hccl":
        parser.error("per-call device all-reduce requires --guard-backend hccl")
    if args.accuracy_only and not args.golden_dir:
        parser.error("--accuracy-only requires --golden-dir; generate the FP8 CPU Golden first")
    if args.debug_intermediates and not args.accuracy_only:
        parser.error("--debug-intermediates requires --accuracy-only")
    if not 1 <= args.trace_capacity <= 65536:
        parser.error("trace-capacity must be in [1,65536]")
    if not 0 <= args.gmm1_ahead_mgroups <= 65536:
        parser.error("gmm1-ahead-mgroups must be in [0,65536]")
    if any(n <= 0 for n in args.exp_shape) or args.exp_repeat < 0 or args.launch_timeout <= 0:
        parser.error("exp-shape/launch-timeout must be positive; exp-repeat must be nonnegative")
    args.fine = args.fine or args.trace
    args.timer = args.timer or args.fine
    return args


def format_profile_summary(reports, world):
    """Display the already aggregated formal profiler results; no new measurements."""
    headers = ("BS/rank", "EP", "Rounds", "round-max-mean(us)", "round-min-mean(us)")
    rows = [
        (str(report["bs"]), str(world), str(report["profile"]["rounds"]),
         f'{report["profile"]["rank_max_mean_us"]:.3f}',
         f'{report["profile"]["rank_min_mean_us"]:.3f}')
        for report in reports
    ]
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    line = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    def row_text(row):
        return "| " + " | ".join(value.rjust(width) for value, width in zip(row, widths)) + " |"
    return "\n".join([
        "[URMA PROFILE SUMMARY] device kernel; formal rounds only; timer/trace off",
        "round-max/min-mean = mean of per-round cross-rank maximum/minimum (not max/min of rank means)",
        line, row_text(headers), line, *(row_text(row) for row in rows), line,
    ])


def main():
    args = parse_args()
    config = load_config(args.config, args.bs_values)
    for key in ("warmup", "iters"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if (
        config["warmup"] < 0
        or config["iters"] < 1
        or args.cv_rows <= 0
        or args.check_tokens < 0
        or not math.isfinite(args.timer_clock_mhz)
        or args.timer_clock_mhz <= 0
    ):
        raise ValueError("warmup/check-tokens must be nonnegative; iters/cv-rows/clock must be positive and finite")
    import torch
    import torch_npu  # noqa: F401
    from deep_ep._store import StoreGroup

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env()
    try:
        run(args, config, group)
    except BaseException as error:
        group.abort(f"URMA benchmark: {type(error).__name__}: {error}")
        raise


def export_trace_collective(group, raw, case, *, bs, clock_mhz, include_waits=False):
    """All ranks exchange exporter errors before any rank raises on overflow."""
    from .urma_trace import export_trace

    case = Path(case)
    status = dict(
        rank=group.rank(),
        error=None,
        metadata=None,
        raw_path=str(case / "urma_trace.bin"),
        chrome_path=str(case / "urma_trace.json"),
    )
    try:
        raw.astype("<i8", copy=False).tofile(case / "urma_trace.bin")
        status["metadata"] = export_trace(
            raw, case / "urma_trace.json", rank=group.rank(), clock_mhz=clock_mhz, include_waits=include_waits
        )
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
    try:
        (case / "trace_metadata.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    except OSError as error:
        status["error"] = (status["error"] or "") + f"; failed to save trace metadata: {error}"
    records = group.all_gather(status, f"URMA trace export BS={bs}")
    failures = [f"rank={record['rank']}: {record['error']}" for record in records if record["error"]]
    if failures:
        raise RuntimeError("URMA detailed trace failed: " + "; ".join(failures))
    return records


def run(args, config, group):
    import torch
    from .api import transform_weights_for_mega_moe
    from .reference import make_weights, quantize_input
    from .urma import COMMUNICATION_PROTOCOL, UrmaBuffer, _extension, use_fused_gate_up
    from .urma_timing import UrmaLaunchGuard, measure_profiled_forward, summarize_profile_rounds
    from .validation import collective_check

    rank, world = group.rank(), group.size()
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    collective_check(
        group,
        "URMA benchmark options",
        None,
        (
            config,
            args.routing,
            args.cv_rows,
            args.gmm1_ahead_mgroups,
            args.exp_shape,
            args.exp_repeat,
            args.launch_timeout,
            args.guard_backend,
            args.max_recv_rows,
            args.timer,
            args.timer_clock_mhz,
            args.fine,
            args.trace_capacity,
            args.trace_include_waits,
            str(args.golden_dir),
            args.accuracy_only,
            args.debug_intermediates,
            args.check_tokens,
            args.allow_multi_host,
            ranks_per_node,
        ),
    )
    rank_dir = args.output_dir / f"rank_{rank}"
    error = None
    try:
        rank_dir.mkdir(parents=True, exist_ok=False)
    except OSError as failure:
        error = str(failure)
    collective_check(group, "URMA output directory", error)
    if args.golden_dir:
        from .golden import contract, load

        error = None
        try:
            for bs in config["bs_values"]:
                load(args.golden_dir, contract(config, world, bs, "fp8", args.routing, args.check_tokens), rank)
        except Exception as failure:
            error = str(failure)
        collective_check(group, "URMA CPU Golden", error)
    model = config["model"]
    owner = UrmaBuffer(
        group,
        **model,
        num_max_tokens_per_rank=max(config["bs_values"]),
        cv_rows=args.cv_rows,
        gmm1_ahead_mgroups=args.gmm1_ahead_mgroups,
        max_recv_rows=args.max_recv_rows,
        allow_multi_host=args.allow_multi_host,
    )
    if args.timer and not owner.timer_enabled:
        raise RuntimeError("coarse/fine diagnostics require native URMA timer support in the installed wheel")

    def device_pair(pair):
        data, scales = pair
        return torch.from_numpy(data).npu(), torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu)

    raw = make_weights(
        model["num_experts"] // world, model["hidden"], model["intermediate_hidden"], config["seed"] + rank
    )
    debug_weight_sources = None
    if args.debug_intermediates:
        from .urma_debug import source_weight_prefixes

        debug_weight_sources = source_weight_prefixes(raw)
    weights = transform_weights_for_mega_moe(*map(device_pair, raw))
    launch_guard = UrmaLaunchGuard(
        group, args.exp_shape, args.exp_repeat, args.launch_timeout, backend=args.guard_backend
    )
    cascade_reference = None
    if args.golden_dir:
        from .cascade import CascadeReference

        error = None
        try:
            cascade_reference = CascadeReference(model, raw, host_group=group)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        collective_check(group, "URMA cascade preparation", error)
    del raw
    launch_guard.prepare_per_call_alignment()
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        rank=rank,
        world=world,
        ranks_per_node=ranks_per_node,
        allow_multi_host=args.allow_multi_host,
        bandwidth_topology="logical contiguous rank groups of four; multi-host validation is not UBX bandwidth equivalence",
        transport="URMA QP0 RO PUT + per-CV SO Dispatch epochs + SO Combine bitmap publication",
        communication_protocol=COMMUNICATION_PROTOCOL,
        communication_qps_per_peer=1,
        dispatch_combine_overlap="across independent peer owners only",
        same_peer_dispatch_combine_overlap=False,
        dispatch_combine_transition="own Dispatch completion -> ready Combine scan; no local Dispatch/Pack join",
        dispatch_batch_cvs="1,1,2,4,8,8,...; ready remainder after Pack completes",
        dispatch_pack_cvs="startup2; before PUT: send-count+1 (2,2,3,5,9,...), capped by remaining quota",
        dispatch_publication="one SO epoch per covered original CV; no per-batch quiet",
        combine_publication="one disjoint SO bitmap mask per contiguous run within a 64-CV word",
        communication_completion="Dispatch/Combine final outbound QP quiet; independent of Reduce",
        remote_workers=world - 1,
        swiglu_workers=28 - (world - 1),
        reduce_workers=56 - (world - 1),
        routing=args.routing,
        cv_rows=args.cv_rows,
        gmm1_ahead_mgroups=args.gmm1_ahead_mgroups,
        launch_guard=launch_guard.describe(),
        performance_timing=(
            "disabled for accuracy-only"
            if args.accuracy_only
            else "torch_npu profiler device kernel duration; no Event or host-wall timing"
        ),
        performance_scope="mega_moe_urma_kernel only; routes prepared once before profiling",
        performance_rounds="formal iterations only; optional one-shot diagnostics excluded",
        profile_modes=["none"] + (["coarse"] if args.timer else []) + (["fine"] if args.fine else []),
        profile_config="Level0/AiCoreNone/Text; shapes/memory/stack/FLOPs/modules disabled",
        max_recv_rows=owner.max_recv_rows,
        symmetric_buffer_bytes=owner.symmetric_buffer_bytes,
        route_preparation="NPU sort/count/inverse + D2H counts + Store count exchange + native descriptors",
        numerical_validation="cpu_golden_and_npu_cascade_error_budget" if args.golden_dir else "skipped",
        accuracy_policy_source="local DeepEP 5a9d6ffad7474137fa134cd7a60aebb2c8ee3089",
        accuracy_only=args.accuracy_only,
        input="prequantized FP8 E4M3 + E8M0; conversion/input generation outside measurement",
        native_urma_protocol=int(_extension().urma_api_protocol),
        compute_schedule=getattr(_extension(), "urma_compute_schedule", "source-cv"),
        local_pipeline=getattr(_extension(), "urma_local_pipeline", "single-slot"),
        tail_schedule=_extension().urma_tail_schedule,
        timer_clock_mhz=args.timer_clock_mhz,
        timer_clock_source="explicit assumption; default follows legacy timer/protocol.py CYCLE_TO_US=1000",
        detailed_trace=args.fine,
        trace_capacity_per_engine=args.trace_capacity,
        trace_include_waits=args.trace_include_waits,
        trace_clock_alignment="per-rank independent zero; no synchronized cross-rank timeline",
        trace_scope="one optional coarse and one optional fine invocation, after formal rounds in the same profiler",
        coarse_compute_scope="stage entry through final output drain; includes initial input wait and all internal gaps",
    )
    (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if rank == 0:
        print(
            f"[URMA SCHEDULE] compute={manifest['compute_schedule']} communication_cv_rows={args.cv_rows} "
            f"gmm1_ahead_mgroups={args.gmm1_ahead_mgroups} "
            f"local_pipeline={manifest['local_pipeline']} tail={manifest['tail_schedule']} "
            f"communication={COMMUNICATION_PROTOCOL}",
            flush=True,
        )
        print(
            f"[URMA LAUNCH GUARD] {args.guard_backend} barrier + exp shape={tuple(args.exp_shape)} "
            f"repeat={args.exp_repeat}; "
            f"buffers={manifest['launch_guard']['allocated_bytes_per_rank'] / 2**30:.2f} GiB/rank; "
            "once per BS; device all-reduce before every call; both excluded from kernel timings",
            flush=True,
        )
    all_reports = []
    for bs in config["bs_values"]:
        case = rank_dir / f"bs_{bs}"
        case.mkdir()
        wave_mgroups = 12 if world == 4 and bs == 1024 else 0
        fused_gate_up = use_fused_gate_up(bs, world, model["num_topk"], model["num_experts"])
        if rank == 0:
            expected_rows = bs * world * model["num_topk"] / model["num_experts"]
            print(
                f"[URMA GMM1] BS={bs} EP={world} expected-rows/expert={expected_rows:.3f} "
                f"gate_up={'fused128+128' if fused_gate_up else 'planar256+256'}",
                flush=True,
            )
        if rank == 0 and wave_mgroups:
            print("[URMA WAVE] EP4 BS1024: 12 real M-groups/wave; GMM1(wave)->GMM2(wave); ahead-mgroups not applied", flush=True)
        host_x, host_ids, host_weights = inputs(config, rank, bs, args.routing, world=world)
        codes, scales = quantize_input(host_x)
        x = (
            torch.from_numpy(codes).npu().view(torch.float8_e4m3fn),
            torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu),
        )
        ids = torch.from_numpy(host_ids).npu()
        topk_weights = torch.from_numpy(host_weights).npu()
        output = torch.empty((bs, model["hidden"]), dtype=torch.bfloat16, device="npu")
        if args.debug_intermediates:
            # Diagnose missing stores separately from stores that produce zero.
            output.fill_(float("nan"))
        prepared = owner.prepare_routes(ids)
        # Reuse the exact formal forward with optional diagnostic buffers.
        timer, trace, coarse_trace, error = None, None, None, None
        try:
            if args.timer:
                coarse_trace = owner.allocate_trace(16)
            if args.fine:
                timer = owner.allocate_timer()
                trace = owner.allocate_trace(args.trace_capacity)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        collective_check(group, f"URMA diagnostic allocation BS={bs}", error)
        torch.npu.synchronize()
        startup_mgroups = int(prepared.native.gmm1_startup_mgroups)
        forward = partial(owner.forward_prepared, x, topk_weights, *weights, prepared, out=output)
        coarse_forward = partial(forward, trace=coarse_trace, coarse_trace=True) if args.timer else None
        fine_forward = partial(forward, timer=timer, trace=trace) if args.fine else None
        mode_reports = {}
        if args.accuracy_only:
            launch_guard.begin()
            for _ in range(config["warmup"] + config["iters"]):
                launch_guard.before_forward()
                forward()
            if coarse_forward is not None:
                launch_guard.before_forward()
                coarse_forward()
            if fine_forward is not None:
                launch_guard.before_forward()
                fine_forward()
            launch_guard.end()
            capture = dict(profile_dir=None, diagnostic_sample=None, diagnostics={})
            profile_report = dict(skipped=True, reason="accuracy-only; no profiler or Event timing")
        else:
            capture = measure_profiled_forward(
                forward, warmup=config["warmup"], iters=config["iters"], guard=launch_guard,
                directory=case / "profile", coarse_forward=coarse_forward, fine_forward=fine_forward,
            )
            (case / "kernel_samples.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")
            mode_values = {"none": capture["durations_us"]}
            mode_values.update({mode: [sample["duration_us"]] for mode, sample in capture["diagnostics"].items()})
            for mode, durations in mode_values.items():
                records = group.all_gather(durations, f"URMA {mode} profile samples BS={bs}")
                report = summarize_profile_rounds(records, ranks_per_node=ranks_per_node)
                report.update(instrumentation=mode, launch_guard=launch_guard.describe(),
                              per_call_device_all_reduce=True, measured_scope="profiler mega_moe_urma_kernel only",
                              scope="formal rounds" if mode == "none" else "single diagnostic invocation")
                report["last_round_rank_max_us"] = max(row[-1] for row in records)
                report["last_round_rank_min_us"] = min(row[-1] for row in records)
                mode_reports[mode] = report
                if mode != "none":
                    mode_case = case / mode
                    mode_case.mkdir(exist_ok=True)
                    (mode_case / "kernel_samples.json").write_text(json.dumps(dict(
                        instrumentation=mode, diagnostic_sample=capture["diagnostics"][mode],
                        profile_dir=capture["profile_dir"]), indent=2), encoding="utf-8")
                    if rank == 0:
                        print(f"[URMA {mode.upper()}] BS={bs} rank-max={report['rank_max_mean_us']:.3f}us "
                              f"rank-min={report['rank_min_mean_us']:.3f}us (one diagnostic; excluded from formal mean)", flush=True)
            profile_report = mode_reports["none"]
        reports = dict(
            bs=bs, received_rows_local=prepared.received_rows,
            gmm1_ahead_mgroups=args.gmm1_ahead_mgroups, gmm1_startup_mgroups_local=startup_mgroups,
            profile=profile_report, profiles=mode_reports, profile_directory=capture["profile_dir"],
            diagnostic_sample=capture["diagnostic_sample"], diagnostics=capture["diagnostics"], validation="skipped",
        )
        if rank == 0 and not args.accuracy_only:
            print(f"[URMA PROFILE] BS={bs} cv_rows={args.cv_rows} "
                  f"round-max-mean={profile_report['rank_max_mean_us']:.3f}us "
                  f"round-min-mean={profile_report['rank_min_mean_us']:.3f}us "
                  f"rounds={profile_report['rounds']} (device kernel; timer/trace off)", flush=True)
        if args.golden_dir:
            import numpy as np
            from .golden import contract, load
            from .urma_accuracy import compare_outputs

            expected, expected_counts, selected = load(
                args.golden_dir, contract(config, world, bs, "fp8", args.routing, args.check_tokens), rank
            )
            actual = output.float().cpu().numpy()
            if args.debug_intermediates:
                from .urma_debug import save_snapshot

                debug_raw = prepared.native.debug_snapshot()
                for name, value in (
                    ("x", x[0]),
                    ("x_scale", x[1]),
                    ("output", output),
                    ("weights", topk_weights),
                    ("reduce_index", prepared.routing.reduce_index),
                    ("order", prepared.routing.order),
                ):
                    try:
                        debug_raw[name] = value.view(torch.uint8).reshape(-1).cpu()
                    except RuntimeError as error:
                        raise RuntimeError(f"URMA diagnostic D2H failed for {name}: {error}") from error
                # Prepared FP4 weights use internal NZ format; .cpu() is unsupported.
                # These distinctly named prefixes describe the original CPU source only.
                debug_raw.update(debug_weight_sources)
                save_snapshot(
                    debug_raw,
                    case,
                    bs=bs,
                    hidden=model["hidden"],
                    intermediate=model["intermediate_hidden"],
                    topk=model["num_topk"],
                    received_rows=prepared.received_rows,
                    rank=rank,
                    fused_gate_up=fused_gate_up,
                )
                del debug_raw
            source_counts = group.all_gather(
                prepared.routing.expert_counts.cpu().tolist(), f"URMA validation counts BS={bs}"
            )
            local_experts = model["num_experts"] // world
            actual_counts = np.asarray(source_counts, dtype=np.int64).sum(axis=0)[
                rank * local_experts : (rank + 1) * local_experts
            ]
            cascade_output, cascade_counts = cascade_reference.forward(
                None, ids, topk_weights, input_kind="fp8", quantized=x
            )
            torch.npu.synchronize()
            local_validation = compare_outputs(
                actual,
                cascade_output.float().cpu().numpy(),
                expected,
                actual_counts,
                cascade_counts.cpu().numpy(),
                expected_counts,
                selected,
            )
            local_validation["received_rows_equal"] = int(actual_counts.sum()) == prepared.received_rows
            local_validation["passed"] &= local_validation["received_rows_equal"]
            if not local_validation["received_rows_equal"]:
                local_validation["reason"] += "; URMA descriptor received_rows disagrees with route counts"
            local_validation["rank"] = rank
            validations = group.all_gather(local_validation, f"URMA accuracy BS={bs}")
            reports["validation"] = validations
            (case / "accuracy.json").write_text(json.dumps(validations, indent=2, allow_nan=False), encoding="utf-8")
            if not all(value["passed"] for value in validations):
                np.savez_compressed(
                    case / "accuracy_tensors.npz",
                    actual=actual[selected],
                    cascade=cascade_output.float().cpu().numpy()[selected],
                    expected=expected,
                    selected=selected,
                    actual_counts=actual_counts,
                    cascade_counts=cascade_counts.cpu().numpy(),
                    expected_counts=expected_counts,
                )
                (case / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
                raise AssertionError(f"BS={bs} URMA dual-reference accuracy failed; inspect {case / 'accuracy.json'}")
            del cascade_output, cascade_counts
        if args.fine:
            from .urma_timer import communication_bandwidth, summarize_bandwidth

            raw_timer = timer.cpu().numpy().astype("<i8", copy=False)
            raw_timer.tofile(case / "urma_timer.bin")
            timer_metadata = dict(
                numel=int(timer.numel()),
                dtype="little-endian int64",
                schema=getattr(_extension(), "urma_timer_schema", None),
                note="URMA format; do not parse using the legacy MegaMoE Wave timer parser",
                scope="fine diagnostic invocation; excluded from formal profile samples",
                detailed_trace_enabled=args.fine,
            )
            (case / "timer.json").write_text(json.dumps(timer_metadata, indent=2), encoding="utf-8")
            bandwidth = communication_bandwidth(raw_timer, rank, world=world, clock_mhz=args.timer_clock_mhz)
            bandwidth_records = group.all_gather(bandwidth, f"URMA bandwidth BS={bs}")
            reports["bandwidth"] = summarize_bandwidth(bandwidth_records)
            reports["bandwidth_per_rank"] = bandwidth_records
            reports["bandwidth_scope"] = timer_metadata["scope"]
            reports["bandwidth_with_trace"] = args.fine
            if rank == 0:
                for row in reports["bandwidth"]:
                    if row["mean_gbps"] is not None:
                        payload_scope = "remote-peers+self-copy" if row["includes_self_copy"] else "remote-peers"
                        print(
                            f"[URMA BW] BS={bs} {row['phase']} {row['communication_class']} "
                            f"mean={row['mean_gbps']:.3f} GB/s/card "
                            f"rank-min/max={row['rank_min_gbps']:.3f}/{row['rank_max_gbps']:.3f} "
                            f"payload={payload_scope} "
                            "(final diagnostic; one-way payload / same-rank class envelope)",
                            flush=True,
                        )
            del raw_timer
        if args.timer:
            from .urma_trace_merge import merge_trace_collective
            coarse_case = case / "coarse"
            coarse_case.mkdir(exist_ok=True)
            records = export_trace_collective(group, coarse_trace.cpu().numpy(), coarse_case,
                                              bs=bs, clock_mhz=args.timer_clock_mhz, include_waits=True)
            reports["coarse_trace_per_rank"] = records
            reports["coarse_trace_allrank"] = merge_trace_collective(
                group, records, args.output_dir / f"bs_{bs}" / "coarse" / "allrank.json", bs=bs, include_waits=True)
        if args.fine:
            from .urma_trace_merge import merge_trace_collective

            # One final detailed diagnostic, excluded from formal samples.
            (case / "fine").mkdir(exist_ok=True)
            raw_trace = trace.cpu().numpy()
            reports["trace_per_rank"] = export_trace_collective(
                group,
                raw_trace,
                case / "fine",
                bs=bs,
                clock_mhz=args.timer_clock_mhz,
                include_waits=args.trace_include_waits,
            )
            reports["trace_allrank"] = merge_trace_collective(
                group,
                reports["trace_per_rank"],
                args.output_dir / f"bs_{bs}" / "fine" / "allrank.json",
                bs=bs,
                include_waits=args.trace_include_waits,
            )
            if rank == 0:
                print(
                    f"[URMA TRACE] BS={bs} all {world} ranks exported; "
                    f"capacity={args.trace_capacity}/engine; rank-local clocks; "
                    f"{args.output_dir}/rank_*/bs_{bs}/fine/urma_trace.json",
                    flush=True,
                )
                print(f"[URMA ALLRANK] BS={bs}; {reports['trace_allrank']['path']}", flush=True)
            del raw_trace
        del prepared, timer, trace, coarse_trace, coarse_forward, fine_forward, forward
        (case / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        all_reports.append(reports)
        if rank == 0:
            print(
                f"[URMA LOOKAHEAD] BS={bs} requested_mgroups={args.gmm1_ahead_mgroups} "
                f"rank0_startup_mgroups={startup_mgroups}",
                flush=True,
            )
            print(
                f"[URMA VALIDATION] BS={bs} "
                + ("CPU Golden + NPU cascade passed (all ranks)" if args.golden_dir else "skipped"),
                flush=True,
            )
        group.barrier(f"URMA case complete BS={bs}")
    if rank == 0:
        (args.output_dir / "summary.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    group.barrier("URMA before teardown")
    owner.destroy()
    launch_guard.close()
    group.close()
    if rank == 0 and not args.accuracy_only:
        print(format_profile_summary(all_reports, world), flush=True)
        with (args.output_dir / "mode_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["bs", "ep", "mode", "rounds", "rank_max_mean_us", "rank_min_mean_us", "scope"])
            for item in all_reports:
                for mode, row in item["profiles"].items():
                    writer.writerow([item["bs"], world, mode, row["rounds"], row["rank_max_mean_us"], row["rank_min_mean_us"], row["scope"]])



if __name__ == "__main__":
    main()
