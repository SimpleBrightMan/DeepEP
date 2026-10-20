"""Full URMA MegaMoE benchmark; EP16, with explicit opt-in for multiple hosts."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from .benchmark import inputs, load_config
from .performance import summarize_rounds


def summarize_metric(per_rank, metric, scope, ranks_per_node=16):
    report = summarize_rounds(per_rank, ranks_per_node)
    report.update(
        metric=metric,
        measured_scope=scope,
        excludes="weights/input generation, runtime creation, pre-iteration host barrier",
        timer_pass=False,
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="urma")
    parser.add_argument("--bs-values", default="2048,4096,8192")
    parser.add_argument("--routing", choices=("balanced", "random", "skewed"), default="balanced")
    parser.add_argument("--cv-rows", type=int, default=256)
    parser.add_argument(
        "--allow-multi-host",
        action="store_true",
        help="Explicitly allow EP16 across hosts; requires cross-host SHMEM mapping and URMA connectivity",
    )
    parser.add_argument(
        "--max-recv-rows", type=int, help="Per-rank capacity; default 2 * max(BS) * topk; excessive skew is rejected"
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timer", action="store_true", help="Separate native URMA timer pass, outside formal timing")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Additional detailed Chrome trace pass; implies --timer; neither pass enters formal timing",
    )
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
        "--golden-dir", type=Path, help="Optional prebuilt FP8 CPU Golden; absent means numerical validation skipped"
    )
    parser.add_argument(
        "--check-tokens", type=int, default=0, help="Must match --check-tokens used to create the optional Golden cache"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.trace_capacity <= 65536:
        parser.error("trace-capacity must be in [1,65536]")
    args.timer = args.timer or args.trace
    return args


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
    from .urma import UrmaBuffer, _extension
    from .validation import collective_check

    rank, world = group.rank(), group.size()
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "16"))
    collective_check(
        group,
        "URMA benchmark options",
        None,
        (
            config,
            args.routing,
            args.cv_rows,
            args.max_recv_rows,
            args.timer,
            args.timer_clock_mhz,
            args.trace,
            args.trace_capacity,
            args.trace_include_waits,
            str(args.golden_dir),
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
        max_recv_rows=args.max_recv_rows,
        allow_multi_host=args.allow_multi_host,
    )
    if args.timer and not owner.timer_enabled:
        raise RuntimeError("--timer requires native URMA timer support in the installed wheel")

    def device_pair(pair):
        data, scales = pair
        return torch.from_numpy(data).npu(), torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu)

    raw = make_weights(
        model["num_experts"] // world, model["hidden"], model["intermediate_hidden"], config["seed"] + rank
    )
    weights = transform_weights_for_mega_moe(*map(device_pair, raw))
    del raw
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        rank=rank,
        world=world,
        ranks_per_node=ranks_per_node,
        allow_multi_host=args.allow_multi_host,
        bandwidth_topology="logical contiguous rank groups of four; multi-host validation is not UBX bandwidth equivalence",
        transport="URMA PUT-only",
        remote_workers=15,
        swiglu_workers=13,
        routing=args.routing,
        cv_rows=args.cv_rows,
        max_recv_rows=owner.max_recv_rows,
        symmetric_buffer_bytes=owner.symmetric_buffer_bytes,
        full_e2e_includes_route_preparation=True,
        route_preparation="NPU sort/count/inverse + D2H counts + Store count exchange + native descriptors",
        numerical_validation="cpu_golden" if args.golden_dir else "skipped",
        input="prequantized FP8 E4M3 + E8M0; conversion/input generation outside measurement",
        native_urma_protocol=int(_extension().urma_api_protocol),
        compute_schedule=getattr(_extension(), "urma_compute_schedule", "source-cv"),
        local_pipeline=getattr(_extension(), "urma_local_pipeline", "single-slot"),
        timer_clock_mhz=args.timer_clock_mhz,
        timer_clock_source="explicit assumption; default follows legacy timer/protocol.py CYCLE_TO_US=1000",
        detailed_trace=args.trace,
        trace_capacity_per_engine=args.trace_capacity,
        trace_include_waits=args.trace_include_waits,
        trace_clock_alignment="per-rank independent zero; no synchronized cross-rank timeline",
        trace_scope="additional diagnostic invocation after aggregate timer pass; formal measurements have both disabled",
    )
    (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if rank == 0:
        print(
            f"[URMA SCHEDULE] compute={manifest['compute_schedule']} communication_cv_rows={args.cv_rows} "
            f"local_pipeline={manifest['local_pipeline']}",
            flush=True,
        )
    all_reports = []
    for bs in config["bs_values"]:
        case = rank_dir / f"bs_{bs}"
        case.mkdir()
        host_x, host_ids, host_weights = inputs(config, rank, bs, args.routing, world=world)
        codes, scales = quantize_input(host_x)
        x = (
            torch.from_numpy(codes).npu().view(torch.float8_e4m3fn),
            torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu),
        )
        ids = torch.from_numpy(host_ids).npu()
        topk_weights = torch.from_numpy(host_weights).npu()
        output = torch.empty((bs, model["hidden"]), dtype=torch.bfloat16, device="npu")
        for _ in range(config["warmup"]):
            owner.forward(x, ids, topk_weights, *weights, out=output)
        torch.npu.synchronize()
        group.barrier(f"URMA warmup BS={bs}")

        route_times = []
        for iteration in range(config["iters"]):
            torch.npu.synchronize()
            group.barrier(f"URMA route timing BS={bs} round={iteration}")
            start = time.perf_counter_ns()
            prepared = owner.prepare_routes(ids)
            torch.npu.synchronize()
            route_times.append((time.perf_counter_ns() - start) / 1000)
        route_records = group.all_gather(route_times, f"URMA route samples BS={bs}")
        route_report = summarize_metric(
            route_records, "ROUTE_PREP", "host wall time of complete route/descriptor preparation", ranks_per_node
        )

        # NPU event interval contains the entire native forward, including local
        # expansion and final reduction. It is not a communication-only timer.
        cached_events = []
        group.barrier(f"URMA cached timing BS={bs}")
        for _ in range(config["iters"]):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            owner.forward_prepared(x, topk_weights, *weights, prepared, out=output)
            end.record()
            cached_events.append((begin, end))
        torch.npu.synchronize()
        cached_times = [begin.elapsed_time(end) * 1000 for begin, end in cached_events]
        cached_records = group.all_gather(cached_times, f"URMA cached samples BS={bs}")
        cached_report = summarize_metric(
            cached_records,
            "CACHED_FORWARD",
            "NPU event interval: local expand through final weighted reduction; routes cached",
            ranks_per_node,
        )
        received_rows = prepared.received_rows
        del prepared, cached_events

        full_times = []
        for iteration in range(config["iters"]):
            torch.npu.synchronize()
            group.barrier(f"URMA full timing BS={bs} round={iteration}")
            start = time.perf_counter_ns()
            owner.forward(x, ids, topk_weights, *weights, out=output)
            torch.npu.synchronize()
            full_times.append((time.perf_counter_ns() - start) / 1000)
        full_records = group.all_gather(full_times, f"URMA full samples BS={bs}")
        full_report = summarize_metric(
            full_records,
            "FULL_E2E",
            "host wall time: fresh route preparation + local expand + complete native operator + synchronize",
            ranks_per_node,
        )
        reports = {
            "bs": bs,
            "received_rows_local": received_rows,
            "route_preparation": route_report,
            "cached_forward": cached_report,
            "full_e2e": full_report,
            "validation": "skipped",
        }
        if args.golden_dir:
            from .golden import compare, contract, load

            expected, _, selected = load(
                args.golden_dir, contract(config, world, bs, "fp8", args.routing, args.check_tokens), rank
            )
            actual = output.float().cpu().numpy()[selected]
            local_validation = compare(actual, expected)
            local_validation["rank"] = rank
            validations = group.all_gather(local_validation, f"URMA accuracy BS={bs}")
            reports["validation"] = validations
            if not all(value["passed"] for value in validations):
                (case / "accuracy.json").write_text(json.dumps(validations, indent=2), encoding="utf-8")
                raise AssertionError(f"BS={bs} URMA output differs from the CPU Golden")
        if args.timer:
            from .urma_timer import communication_bandwidth, summarize_bandwidth

            prepared = owner.prepare_routes(ids)
            timer = owner.allocate_timer()
            group.barrier(f"URMA timer BS={bs}")
            owner.forward_prepared(x, topk_weights, *weights, prepared, timer=timer, out=output)
            torch.npu.synchronize()
            raw_timer = timer.cpu().numpy().astype("<i8", copy=False)
            raw_timer.tofile(case / "urma_timer.bin")
            timer_metadata = dict(
                numel=int(timer.numel()),
                dtype="little-endian int64",
                schema=getattr(_extension(), "urma_timer_schema", None),
                note="URMA format; do not parse using the legacy MegaMoE Wave timer parser",
            )
            (case / "timer.json").write_text(json.dumps(timer_metadata, indent=2), encoding="utf-8")
            bandwidth = communication_bandwidth(raw_timer, rank, clock_mhz=args.timer_clock_mhz)
            bandwidth_records = group.all_gather(bandwidth, f"URMA bandwidth BS={bs}")
            reports["bandwidth"] = summarize_bandwidth(bandwidth_records)
            reports["bandwidth_per_rank"] = bandwidth_records
            if rank == 0:
                for row in reports["bandwidth"]:
                    if row["mean_gbps"] is not None:
                        payload_scope = "remote-peers+self-copy" if row["includes_self_copy"] else "remote-peers"
                        print(
                            f"[URMA BW] BS={bs} {row['phase']} {row['communication_class']} "
                            f"mean={row['mean_gbps']:.3f} GB/s/card "
                            f"rank-min/max={row['rank_min_gbps']:.3f}/{row['rank_max_gbps']:.3f} "
                            f"payload={payload_scope} "
                            "(one-way payload / same-rank class envelope)",
                            flush=True,
                        )
            del prepared, timer, raw_timer
        if args.trace:
            from .urma_trace_merge import merge_trace_collective

            # Keep the aggregate bandwidth pass free from detailed trace costs.
            # Formal cached/full samples above pass neither timer nor trace.
            prepared = owner.prepare_routes(ids)
            trace, error = None, None
            try:
                trace = owner.allocate_trace(args.trace_capacity)
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            collective_check(group, f"URMA trace allocation BS={bs}", error)
            group.barrier(f"URMA detailed trace BS={bs}")
            owner.forward_prepared(x, topk_weights, *weights, prepared, trace=trace, out=output)
            torch.npu.synchronize()
            raw_trace = trace.cpu().numpy()
            reports["trace_per_rank"] = export_trace_collective(
                group,
                raw_trace,
                case,
                bs=bs,
                clock_mhz=args.timer_clock_mhz,
                include_waits=args.trace_include_waits,
            )
            reports["trace_allrank"] = merge_trace_collective(
                group,
                reports["trace_per_rank"],
                args.output_dir / f"bs_{bs}" / "allrank.json",
                bs=bs,
                include_waits=args.trace_include_waits,
            )
            if rank == 0:
                print(
                    f"[URMA TRACE] BS={bs} all {world} ranks exported; "
                    f"capacity={args.trace_capacity}/engine; rank-local clocks; "
                    f"{args.output_dir}/rank_*/bs_{bs}/urma_trace.json",
                    flush=True,
                )
                print(f"[URMA ALLRANK] BS={bs}; {reports['trace_allrank']['path']}", flush=True)
            del prepared, trace, raw_trace
        (case / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        all_reports.append(reports)
        if rank == 0:
            for report in (route_report, cached_report, full_report):
                print(
                    f"[URMA {report['metric']}] BS={bs} cv_rows={args.cv_rows} "
                    f"rank-max-mean={report['rank_max_mean_us']:.3f}us "
                    f"rank-min-mean={report['mean_us']:.3f}us "
                    f"rounds={report['rounds']}/{report['recorded_rounds']}",
                    flush=True,
                )
            print(f"[URMA VALIDATION] BS={bs} " + ("CPU Golden passed" if args.golden_dir else "skipped"), flush=True)
        group.barrier(f"URMA case complete BS={bs}")
    if rank == 0:
        (args.output_dir / "summary.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    group.barrier("URMA before teardown")
    owner.destroy()
    group.close()


if __name__ == "__main__":
    main()
