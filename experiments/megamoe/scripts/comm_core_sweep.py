#!/usr/bin/env python3
# Copyright (c) 2026, SimpleBright_Man

"""EP16 communication-only MegaMoE simulation and optional core-ratio sweeps."""

import argparse
import csv
from datetime import datetime, timezone
import itertools
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

ENV_DISPATCH = "DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES"
ENV_COMBINE = "DEEPEP_MEGAMOE_COMBINE_INTRA_CORES"
DEFAULT_CORES = "2,4,6,7,8,10,12,16,20,24,26"
ROOT = Path(__file__).resolve().parents[3]


def integers(text, maximum, label):
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label}: expected comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(not 1 <= v <= maximum for v in values):
        raise argparse.ArgumentTypeError(f"{label}: expected distinct values in 1..{maximum}")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bs-values", default="2048,4096,8192")
    parser.add_argument(
        "--intra-cores",
        default=DEFAULT_CORES,
        help="Dispatch candidates; both stages only in staged/fullgrid mode; CROSS=28-INTRA",
    )
    parser.add_argument("--dispatch-intra-cores", help="Override Dispatch candidate list")
    parser.add_argument("--combine-intra-cores", help="Combine candidates for explicit staged/fullgrid mode")
    parser.add_argument(
        "--combine-intra",
        "--initial-combine-intra",
        dest="initial_combine_intra",
        type=int,
        default=0,
        help="Fixed Combine INTRA cores; 0 keeps its original schedule (initial value in staged mode)",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "dispatch", "staged", "fullgrid"),
        default="baseline",
        help="Default: original production schedule only; explicitly select dispatch for a Dispatch sweep",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--dispatch-metadata-mode",
        choices=("dma", "scalar"),
        default="dma",
        help="Communication-only Dispatch metadata implementation; full reference stays scalar",
    )
    parser.add_argument(
        "--combine-buffer-count",
        type=int,
        choices=(2, 4, 6),
        default=2,
        help="Communication-only Prefill Combine pipeline slots; full reference retains its original policy",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per fresh torchrun case")
    parser.add_argument(
        "--case-gap-seconds",
        type=float,
        default=2.0,
        help="Wait between fully exited process groups; outside operator timing",
    )
    parser.add_argument(
        "--shmem-port-policy",
        choices=("fresh", "fixed"),
        default="fresh",
        help="Fresh SHMEM bootstrap port per case, or explicitly reuse the inherited endpoint",
    )
    parser.add_argument("--alignment-rows", type=int, default=100000)
    parser.add_argument("--alignment-cols", type=int, default=30000)
    parser.add_argument("--alignment-repeats", type=int, default=20)
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Performance only: skip payload/task audits and report bandwidth from timer row counts",
    )
    references = parser.add_mutually_exclusive_group()
    references.add_argument(
        "--run-full-reference",
        action="store_true",
        help="Run full MegaMoE once per BS and compare original per-core communication timers",
    )
    references.add_argument(
        "--full-reference-dir", type=Path, help="Reuse a full MegaMoE sweep root containing bs_<BS>/baseline"
    )
    parser.add_argument(
        "--timing-rtol", type=float, default=0.10, help="Per-core communication timing comparison relative tolerance"
    )
    parser.add_argument("--timing-atol-us", type=float, default=0.0)
    parser.add_argument(
        "--require-timing-match",
        action="store_true",
        help="Stop if any original-schedule core differs beyond timing tolerance",
    )
    parser.add_argument("--keep-raw-timers", action="store_true", help="Can consume hundreds of GB across a sweep")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="No build, device access, process launch or output files")
    args = parser.parse_args(argv)
    try:
        args.bs_values = integers(args.bs_values, 65536, "BS")
        args.dispatch_cores = integers(args.dispatch_intra_cores or args.intra_cores, 27, "Dispatch cores")
        args.combine_cores = (
            [args.initial_combine_intra]
            if args.mode in ("baseline", "dispatch")
            else integers(args.combine_intra_cores or args.intra_cores, 27, "Combine cores")
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if any(bs not in (2048, 4096, 8192) for bs in args.bs_values):
        parser.error("Communication simulator currently supports BS2048/4096/8192 only")
    if not 0 <= args.initial_combine_intra <= 27 or args.warmup < 0 or args.seed < 0:
        parser.error("Invalid initial cohort, warmup or seed")
    if args.mode in ("baseline", "dispatch") and args.combine_intra_cores is not None:
        parser.error("Combine candidates require explicit staged/fullgrid mode")
    if args.mode == "baseline" and args.initial_combine_intra != 0:
        parser.error("Baseline mode uses the original Dispatch and Combine schedules; select dispatch for experiments")
    if min(args.iters, args.timeout, args.alignment_rows, args.alignment_cols, args.alignment_repeats) < 1:
        parser.error("Iterations, timeout and alignment dimensions/repeats must be positive")
    if not math.isfinite(args.case_gap_seconds) or not 0 <= args.case_gap_seconds <= 60:
        parser.error("case-gap-seconds must be finite and in [0,60]")
    if any(not 0 <= value < float("inf") for value in (args.timing_rtol, args.timing_atol_us)):
        parser.error("Comparison tolerances must be finite and nonnegative")
    if args.no_validation and (args.run_full_reference or args.full_reference_dir or args.require_timing_match):
        parser.error("--no-validation cannot be combined with a full reference or timing comparison")
    if args.require_timing_match and not (args.run_full_reference or args.full_reference_dir):
        parser.error("--require-timing-match requires a full MegaMoE reference")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        args.output_dir = ROOT / "benchmark-runs" / "megamoe_comm_core_sweep" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    args.output_dir = args.output_dir.resolve()
    return args


def configuration(args, bs):
    return dict(
        schema_version=1,
        model=dict(hidden=3584, intermediate_hidden=1280, num_experts=384, num_topk=8, num_shared_experts=0),
        bs_values=[bs],
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        alignment=dict(rows=args.alignment_rows, cols=args.alignment_cols, repeats=args.alignment_repeats),
    )


def plan(args):
    validation_stage = "validation skipped" if args.no_validation else "independent payload and wave/core validation"
    if args.mode == "baseline":
        candidates = 0
        stages = [f"communication-only original schedule + {validation_stage}"]
    elif args.mode == "dispatch":
        candidates = len(args.dispatch_cores)
        sweep_stage = "Dispatch sweep with Combine fixed; no Combine sweep"
    elif args.mode == "fullgrid":
        candidates = len(args.dispatch_cores) * len(args.combine_cores)
        sweep_stage = "all Dispatch x Combine candidate pairs"
    else:
        candidates = (
            len(args.dispatch_cores) + len(args.combine_cores) - int(args.initial_combine_intra in args.combine_cores)
        )
        sweep_stage = "Dispatch sweep with Combine fixed, then Combine sweep with best Dispatch"
    if args.mode != "baseline":
        stages = [
            f"communication-only original schedule baseline + {validation_stage}",
            sweep_stage,
            "fresh-process confirmation of selected pair",
        ]
    case_launches = (1 if args.mode == "baseline" else candidates + 2) + int(args.run_full_reference)
    return dict(
        schema_version=3,
        workload="communication_only",
        validation="skipped" if args.no_validation else "strict",
        mode=args.mode,
        dispatch_metadata_mode=args.dispatch_metadata_mode,
        combine_buffer_count=args.combine_buffer_count,
        ep_size=16,
        physical_aiv1_workers=28,
        bs_values=args.bs_values,
        dispatch_intra_candidates=[] if args.mode == "baseline" else args.dispatch_cores,
        combine_intra_candidates=[] if args.mode == "baseline" else args.combine_cores,
        combine_fixed=args.mode == "dispatch",
        initial_combine_intra=args.initial_combine_intra,
        shmem_port_policy=args.shmem_port_policy,
        case_gap_seconds=args.case_gap_seconds,
        case_launches_per_bs=case_launches,
        total_case_launches=len(args.bs_values) * case_launches,
        selection="minimum timer-off communication-only mean(per-round maximum rank kernel duration)",
        full_reference="run"
        if args.run_full_reference
        else str(args.full_reference_dir)
        if args.full_reference_dir
        else None,
        timing_rtol=args.timing_rtol,
        timing_atol_us=args.timing_atol_us,
        stages=stages,
        bandwidth="Each operation separately: rank payload bytes / (sum of all 28 core communication durations / 28)",
        global_optimum_guaranteed=False,
        model=configuration(args, args.bs_values[0])["model"],
        output_dir=str(args.output_dir),
        notes=[
            "Payload/output checks and transfer audits are skipped; bytes are accounted from measured timer row counts."
            if args.no_validation
            else "Every case checks Dispatch bytes/scales/metadata and Combine output against independent synthetic payload expectations.",
            "No weights, GMM1, SwiGLU or GMM2 run in communication-only cases; synthetic GMM2 rows are prepared once before timing.",
            f"Dispatch keeps its production slot policy; communication-only Prefill Combine uses {args.combine_buffer_count} slots (original: 2).",
            "Dispatch metadata DMA adds one local 32-byte read per row; remote payload bytes and task ownership are unchanged.",
            "Full MegaMoE reference retains original scalar Dispatch metadata; metadata modes need not match for optional task/timing comparisons.",
            "Combine=0 means original Combine ownership; Dispatch experiments can leave Combine unchanged.",
            "Timer structure and complete rank/profile reports are still checked; no task or full-operator timing comparison runs."
            if args.no_validation
            else "Baseline and candidates validate each wave/core against the production wave/task oracle; workload equality alone does not prove matching time.",
            "Staged search can miss interactions; fullgrid only finds the best sampled pair in this run.",
            "Timer pass is separate from timer-off communication-only kernel profiling; no plots are generated.",
            "Each launched case owns a new SHMEM runtime; fresh ports avoid cross-case bootstrap reuse.",
            f"Alignment scratch per rank: {args.alignment_rows * args.alignment_cols * 4} bytes.",
        ],
    )


def candidate_environment(pair, base=None):
    environment = dict(os.environ if base is None else base)
    for name in (ENV_DISPATCH, ENV_COMBINE):
        environment.pop(name, None)
    if pair is not None:
        environment[ENV_DISPATCH], environment[ENV_COMBINE] = map(str, pair)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment["PYTHON_EXEC"] = sys.executable
    return environment


def fresh_endpoint(environment, used):
    """Choose a local IPv4 bootstrap port not used by another case in this run."""
    inherited = environment.get("DEEPEP_SHMEM_ENDPOINT", "tcp://127.0.0.1:19091")
    if not inherited.startswith("tcp://"):
        raise ValueError("DEEPEP_SHMEM_ENDPOINT must be tcp://<local IPv4>:<port>")
    host, separator, port = inherited[6:].partition(":")
    if not separator or not port.isdecimal() or not 1024 <= int(port) <= 65535:
        raise ValueError("DEEPEP_SHMEM_ENDPOINT must use an unprivileged TCP port")
    host = str(ipaddress.IPv4Address(host))
    for _ in range(128):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind((host, 0))
            selected = reservation.getsockname()[1]
        endpoint = f"tcp://{host}:{selected}"
        if endpoint not in used:
            used.add(endpoint)
            return endpoint
    raise RuntimeError("Unable to choose a new SHMEM bootstrap port")


def stop_owned_group(process):
    """Only target the isolated process group created for this case."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def command(args, config_path, output, pair, baseline, *, full_reference=False):
    result = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=16",
        "--module",
        "deep_ep_experimental.megamoe.benchmark" if full_reference else "deep_ep_experimental.megamoe.comm_benchmark",
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    if full_reference:
        result += [
            "--mode",
            "perf",
            "--input",
            "fp8",
            "--routing",
            "balanced",
            "--timer",
            "--timer-ranks",
            "0-15",
            "--comm-report",
        ]
    else:
        result += ["--dispatch-metadata-mode", args.dispatch_metadata_mode]
        result += ["--combine-buffer-count", str(args.combine_buffer_count)]
        if args.no_validation:
            result.append("--no-validation")
    if not args.keep_raw_timers:
        result.append("--no-timer-raw")
    return result


def execute(argv, *, cwd, environment, logfile, timeout):
    """Each case owns a process group so timeout cannot leave competing rank workers."""
    with logfile.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                dict(
                    argv=argv,
                    shmem_endpoint=environment.get("DEEPEP_SHMEM_ENDPOINT"),
                    worker_python=environment.get("PYTHON_EXEC"),
                    comm_split={k: environment.get(k) for k in (ENV_DISPATCH, ENV_COMBINE)},
                )
            )
            + "\n"
        )
        log.flush()
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_owned_group(process)
            raise
        log.write(json.dumps(dict(event="torchrun_exit", pid=process.pid, returncode=code)) + "\n")
    if code:
        stop_owned_group(process)
        raise RuntimeError(f"Case exited with status {code}; inspect {logfile}")


def read_case(output, bs, pair, *, validate=True, dispatch_metadata_mode="dma", combine_buffer_count=2):
    expected_environment = {ENV_DISPATCH: str(pair[0]) if pair else None, ENV_COMBINE: str(pair[1]) if pair else None}
    builds = []
    for rank in range(16):
        manifest = json.loads((output / f"rank_{rank}" / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["rank"] != rank
            or manifest["ep_size"] != 16
            or manifest["mode"] != "perf"
            or manifest["routing"] != "balanced"
            or manifest["input"] != "fp8"
            or manifest.get("workload") != "communication_only"
            or manifest["comm_core_split"] != expected_environment
            or manifest["config"]["bs_values"] != [bs]
            or manifest.get("validation", "strict") != ("strict" if validate else "skipped")
            or manifest.get("dispatch_metadata_mode", "scalar") != dispatch_metadata_mode
            or manifest.get("combine_buffer_count", 2) != combine_buffer_count
        ):
            raise ValueError("Benchmark manifest does not match the requested case")
        builds.append(manifest["build"])
    if any(build != builds[0] for build in builds):
        raise ValueError("Ranks loaded different installed builds")
    rank0 = output / "rank_0" / f"bs_{bs}"
    profile = json.loads((rank0 / "profiling.json").read_text(encoding="utf-8"))
    if validate:
        comparisons = json.loads((rank0 / "payload_accuracy.json").read_text(encoding="utf-8"))
        if sorted(r["rank"] for r in comparisons) != list(range(16)) or not all(r["passed"] for r in comparisons):
            raise ValueError("Missing or failed independent communication payload verification ranks")
    per_rank = profile["per_rank_rounds_us"]
    if (
        profile["timer_pass"]
        or profile.get("metric") != "communication_only_kernel_duration_us"
        or len(per_rank) != 16
        or not per_rank[0]
        or any(len(v) != len(per_rank[0]) for v in per_rank)
    ):
        raise ValueError("Expected complete timer-off 16-rank performance samples")
    if any(not 0 < v < float("inf") for row in per_rank for v in row):
        raise ValueError("Nonfinite/nonpositive profile samples")
    slowest = statistics.mean(max(values) for values in zip(*per_rank))
    report = dict(
        bs=bs,
        pair=list(pair) if pair else None,
        output_dir=str(output),
        passed=True,
        validation="strict" if validate else "skipped",
        dispatch_metadata_mode=dispatch_metadata_mode,
        combine_buffer_count=combine_buffer_count,
        comm_kernel_rank_max_mean_us=slowest,
        comm_kernel_rank_min_mean_us=statistics.mean(min(values) for values in zip(*per_rank)),
        comm_kernel_all_ranks_mean_us=statistics.mean(v for row in per_rank for v in row),
        timer_pass=False,
        selected_rounds=len(per_rank[0]),
        communication=None,
    )
    report["communication"] = json.loads(
        (output / f"bs_{bs}" / "timer" / "communication.json").read_text(encoding="utf-8")
    )
    if report["communication"].get("dispatch_metadata_mode", "scalar") != dispatch_metadata_mode:
        raise ValueError("Communication report metadata mode does not match the requested case")
    if report["communication"].get("combine_buffer_count", 2) != combine_buffer_count:
        raise ValueError("Communication report Combine buffer count does not match the requested case")
    if sorted(r["rank"] for r in report["communication"]["ranks"]) != list(range(16)):
        raise ValueError("Missing communication ranks")
    if report["communication"].get("schema_version") != 2:
        raise ValueError("Expected communication report schema version 2")
    validation = report["communication"].get("validation")
    if validate:
        if (validation is not None and validation != dict(strict_wave_core=True, status="passed")) or not all(
            r.get("validation", {}).get("strict_wave_core") is True for r in report["communication"]["ranks"]
        ):
            raise ValueError("Missing strict per-wave/core communication validation")
    elif validation != dict(strict_wave_core=False, status="skipped") or not all(
        r.get("validation", {}).get("strict_wave_core") is False
        and r.get("validation", {}).get("status") == "skipped"
        and r.get("validation", {}).get("accounting") == "timer_row_counts"
        for r in report["communication"]["ranks"]
    ):
        raise ValueError("Expected explicit validation-skipped timer-row accounting for every rank")
    if set(report["communication"].get("operations", {})) != {"DISPATCH", "COMBINE"}:
        raise ValueError("Expected separate Dispatch and Combine mean-core-time bandwidth reports")
    for value in report["communication"]["operations"].values():
        if any(
            not 0 < value[key] < float("inf")
            for key in ("gbps_per_card_mean", "gbps_per_card_min", "gbps_per_card_max")
        ):
            raise ValueError("Nonfinite/nonpositive communication bandwidth")
    if validate:
        timer_comparison = json.loads((rank0 / "timer_accuracy.json").read_text(encoding="utf-8"))
        if sorted(r["rank"] for r in timer_comparison) != list(range(16)) or not all(
            r["passed"] for r in timer_comparison
        ):
            raise ValueError("Missing/failed timer-on/off validation")
        audit_comparison = json.loads((rank0 / "communication_audit_accuracy.json").read_text(encoding="utf-8"))
        if sorted(r["rank"] for r in audit_comparison) != list(range(16)) or not all(
            r["passed"] for r in audit_comparison
        ):
            raise ValueError("Missing/failed communication audit output validation")
    return report


def compare_reference(args, reference, candidate, bs, destination):
    """Installed comparison code checks identical tasks before timing tolerances."""
    reference_report = reference / f"bs_{bs}" / "timer" / "communication.json"
    candidate_report = candidate / f"bs_{bs}" / "timer" / "communication.json"
    if not reference_report.is_file():
        raise ValueError(f"Missing full MegaMoE reference: {reference_report}")
    for rank in range(16):
        manifest = json.loads((reference / f"rank_{rank}" / "manifest.json").read_text(encoding="utf-8"))
        simulated = json.loads((candidate / f"rank_{rank}" / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("api") != "fp8_fp4_mega_moe"
            or manifest.get("rank") != rank
            or manifest.get("ep_size") != 16
            or manifest.get("mode") != "perf"
            or manifest.get("input") != "fp8"
            or manifest.get("routing") != "balanced"
            or manifest.get("timer_ranks") != list(range(16))
            or simulated.get("timer_ranks") != list(range(16))
            or simulated.get("workload") != "communication_only"
            or bs not in manifest.get("config", {}).get("bs_values", [])
            or manifest.get("config", {}).get("model") != simulated["config"]["model"]
            or manifest.get("config", {}).get("alignment") != simulated["config"].get("alignment")
            or any(
                manifest.get("comm_core_split", {}).get(name) not in (None, "0") for name in (ENV_DISPATCH, ENV_COMBINE)
            )
        ):
            raise ValueError(
                "Full reference must use EP16 original schedule, matching model/BS/alignment and balanced FP8 performance inputs"
            )
    argv = [
        sys.executable,
        "-I",
        "-m",
        "deep_ep_experimental.megamoe.comm_compare",
        "--reference",
        str(reference_report),
        "--candidate",
        str(candidate_report),
        "--timing-rtol",
        str(args.timing_rtol),
        "--timing-atol-us",
        str(args.timing_atol_us),
        "--output",
        str(destination),
    ]
    if args.require_timing_match:
        argv.append("--require-timing-match")
    subprocess.run(argv, check=True)
    return json.loads(destination.read_text(encoding="utf-8"))


def choose(records):
    if not records or any(not item["passed"] for item in records):
        raise ValueError("Cannot select from missing/failed cases")
    return min(records, key=lambda item: (item["comm_kernel_rank_max_mean_us"], item["pair"]))


def write_summary(path, launch_plan, records, selections):
    document = dict(plan=launch_plan, cases=records, selected=selections)
    (path / "summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "bs",
            "stage",
            "dispatch_metadata_mode",
            "combine_buffer_count",
            "dispatch_intra",
            "dispatch_cross",
            "combine_intra",
            "combine_cross",
            "comm_kernel_rank_max_mean_us",
            "comm_kernel_rank_min_mean_us",
            "comm_kernel_all_ranks_mean_us",
            "DISPATCH_GBps",
            "COMBINE_GBps",
            "communication_report",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in records:
            pair = item["pair"]
            row = {
                key: item[key]
                for key in (
                    "bs",
                    "stage",
                    "comm_kernel_rank_max_mean_us",
                    "comm_kernel_rank_min_mean_us",
                    "comm_kernel_all_ranks_mean_us",
                )
            }
            row["dispatch_metadata_mode"] = item.get("dispatch_metadata_mode", "scalar")
            row["combine_buffer_count"] = item.get("combine_buffer_count", 2)
            if pair:
                row.update(
                    dispatch_intra=pair[0],
                    dispatch_cross=28 - pair[0] if pair[0] else "",
                    combine_intra=pair[1],
                    combine_cross=28 - pair[1] if pair[1] else "",
                )
            else:
                row.update(dispatch_intra=0, combine_intra=0)
            row.update(
                {
                    name + "_GBps": value["gbps_per_card_mean"]
                    for name, value in item["communication"]["operations"].items()
                }
            )
            row["communication_report"] = str(
                Path(item["output_dir"]) / f"bs_{item['bs']}" / "timer" / "communication.json"
            )
            writer.writerow(row)


def print_case(item):
    pair = item["pair"]

    def schedule(cores):
        return f"{cores}/{28 - cores}" if cores else "original"

    label = f"D={schedule(pair[0])} C={schedule(pair[1])}" if pair else "original schedule"
    validation = (
        "validation skipped" if item.get("validation") == "skipped" else "independent payload verification passed"
    )
    print(
        f"[CASE] BS={item['bs']} {label} dispatch_metadata_mode={item.get('dispatch_metadata_mode', 'scalar')} "
        f"combine_buffer_count={item.get('combine_buffer_count', 2)} "
        f"COMM-KERNEL rank-max-mean={item['comm_kernel_rank_max_mean_us']:.3f}us "
        f"rank-min-mean={item['comm_kernel_rank_min_mean_us']:.3f}us (timer off; {validation})",
        flush=True,
    )
    if item["communication"]:
        for family, value in item["communication"]["operations"].items():
            print(
                f"[BW] {family} per-card mean={value['gbps_per_card_mean']:.3f} "
                f"rank-min/max={value['gbps_per_card_min']:.3f}/{value['gbps_per_card_max']:.3f} GB/s "
                "(operation payload / (sum of all 28 core communication times / 28))",
                flush=True,
            )
        detail_index = Path(item["output_dir"]) / f"bs_{item['bs']}" / "timer" / "communication.json"
        task_status = (
            "validation skipped; timer row-count accounting"
            if item.get("validation") == "skipped"
            else "strict per-wave/core validation passed"
        )
        detail_format = "JSON" if item.get("validation") == "skipped" else "JSON/CSV"
        print(f"[WAVE_CORE] {task_status}; rank {detail_format} paths in: {detail_index}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    launch_plan = plan(args)
    if args.plan:
        print(json.dumps(launch_plan, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("NPU execution requires Linux; --plan works without an NPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.json").write_text(json.dumps(launch_plan, indent=2) + "\n", encoding="utf-8")
    snapshot = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, check=True)
    (args.output_dir / "npu_smi.txt").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
    print(
        f"[RESOURCE] Read-only NPU snapshot: {args.output_dir / 'npu_smi.txt'}; all 16 devices must be allocated to this job.",
        flush=True,
    )
    records, selections = [], []
    used_endpoints, launched = set(), False
    for bs in args.bs_values:
        directory = args.output_dir / f"bs_{bs}"
        directory.mkdir()
        config_path = directory / "model.json"
        config_path.write_text(json.dumps(configuration(args, bs), indent=2) + "\n", encoding="utf-8")
        baseline = directory / "baseline"
        cached = {}
        full_reference = None
        if args.run_full_reference:
            full_reference = directory / "full_reference"
            if launched and args.case_gap_seconds:
                time.sleep(args.case_gap_seconds)
            environment = candidate_environment(None)
            if args.shmem_port_policy == "fresh":
                environment["DEEPEP_SHMEM_ENDPOINT"] = fresh_endpoint(environment, used_endpoints)
            print(f"[REFERENCE] BS={bs} full MegaMoE; log={directory / 'full_reference.log'}", flush=True)
            execute(
                command(args, config_path, full_reference, None, baseline, full_reference=True),
                cwd=directory,
                environment=environment,
                logfile=directory / "full_reference.log",
                timeout=args.timeout,
            )
            launched = True
        elif args.full_reference_dir:
            full_reference = args.full_reference_dir.resolve() / f"bs_{bs}" / "baseline"

        def run_case(pair, stage, fresh=False):
            nonlocal launched
            if pair is not None and pair in cached and not fresh:
                return cached[pair]
            name = "baseline" if pair is None else f"{stage}_d{pair[0]}_c{pair[1]}"
            output = directory / name
            if launched and args.case_gap_seconds:
                print(f"[CASE_GAP] previous torchrun exited; waiting {args.case_gap_seconds:g}s", flush=True)
                time.sleep(args.case_gap_seconds)
            environment = candidate_environment(pair)
            if args.shmem_port_policy == "fresh":
                environment["DEEPEP_SHMEM_ENDPOINT"] = fresh_endpoint(environment, used_endpoints)
            print(f"[SHMEM_SESSION] new process group; endpoint={environment.get('DEEPEP_SHMEM_ENDPOINT')}", flush=True)
            print(f"[RUN] BS={bs} stage={stage} pair={pair}; log={directory / (name + '.log')}", flush=True)
            execute(
                command(args, config_path, output, pair, baseline),
                cwd=directory,
                environment=environment,
                logfile=directory / (name + ".log"),
                timeout=args.timeout,
            )
            launched = True
            item = dict(
                read_case(
                    output,
                    bs,
                    pair,
                    validate=not args.no_validation,
                    dispatch_metadata_mode=args.dispatch_metadata_mode,
                    combine_buffer_count=args.combine_buffer_count,
                ),
                stage=stage,
            )
            records.append(item)
            if pair is not None:
                cached[pair] = item
            print_case(item)
            write_summary(args.output_dir, launch_plan, records, selections)
            return item

        original = run_case(None, "baseline")
        if full_reference is not None:
            original["full_megamoe_comparison"] = compare_reference(
                args, full_reference, baseline, bs, directory / "full_megamoe_comparison.json"
            )
            comparison = original["full_megamoe_comparison"]
            print(
                f"[REFERENCE_COMPARE] BS={bs} workload_match={comparison['workload_match']} "
                f"timing_match={comparison['timing_match']}; {directory / 'full_megamoe_comparison.json'}",
                flush=True,
            )
            write_summary(args.output_dir, launch_plan, records, selections)
        if args.mode == "baseline":
            continue
        if args.mode == "dispatch":
            for n in args.dispatch_cores:
                run_case((n, args.initial_combine_intra), "dispatch")
        elif args.mode == "fullgrid":
            for pair in itertools.product(args.dispatch_cores, args.combine_cores):
                run_case(pair, "grid")
        else:
            dispatch = [run_case((n, args.initial_combine_intra), "dispatch") for n in args.dispatch_cores]
            best_dispatch = choose(dispatch)["pair"][0]
            for n in args.combine_cores:
                run_case((best_dispatch, n), "combine")
        # Retain first-stage winners even when initial C is absent from C candidates.
        best = choose(list(cached.values()))
        confirmed = run_case(tuple(best["pair"]), "confirm", fresh=True)
        selections.append(
            dict(
                bs=bs,
                pair=best["pair"],
                selection_us=best["comm_kernel_rank_max_mean_us"],
                confirmation_us=confirmed["comm_kernel_rank_max_mean_us"],
                baseline_us=original["comm_kernel_rank_max_mean_us"],
                speedup=original["comm_kernel_rank_max_mean_us"] / confirmed["comm_kernel_rank_max_mean_us"],
            )
        )
        write_summary(args.output_dir, launch_plan, records, selections)
        print(
            f"[SELECTED] BS={bs} D_INTRA={best['pair'][0]} C_INTRA={best['pair'][1]} "
            f"confirmed COMM-KERNEL={confirmed['comm_kernel_rank_max_mean_us']:.3f}us "
            f"speedup={selections[-1]['speedup']:.3f}x against original schedule",
            flush=True,
        )
    timing_mismatch = any(item.get("full_megamoe_comparison", {}).get("timing_match") is False for item in records)
    status = (
        "COMPLETED_VALIDATION_SKIPPED"
        if args.no_validation
        else "COMPLETED_TIMING_MISMATCH"
        if timing_mismatch
        else "PAYLOAD_AND_TASKS_PASSED"
    )
    print(f"[{status}] {len(records)} communication case launches; {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
