#!/usr/bin/env python3
# Copyright (c) 2026, SimpleBright_Man

"""EP16 production W4A8 per-wave communication baseline and optional core-ratio sweeps."""

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
    parser.add_argument("--comparison-rtol", type=float, default=0.0)
    parser.add_argument("--comparison-atol", type=float, default=0.0)
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
    if any(not 0 <= value < float("inf") for value in (args.comparison_rtol, args.comparison_atol)):
        parser.error("Comparison tolerances must be finite and nonnegative")
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
    if args.mode == "baseline":
        candidates = 0
        stages = ["original production schedule + per-wave/core communication validation + full-output save"]
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
            "original production schedule baseline + per-wave/core communication validation + full-output save",
            sweep_stage,
            "fresh-process confirmation of selected pair",
        ]
    case_launches = 1 if args.mode == "baseline" else candidates + 2
    return dict(
        schema_version=2,
        mode=args.mode,
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
        selection="minimum timer-off mean(per-round maximum rank kernel duration)",
        stages=stages,
        bandwidth="Each operation separately: rank payload bytes / (sum of all 28 core communication durations / 28)",
        global_optimum_guaranteed=False,
        model=configuration(args, args.bs_values[0])["model"],
        output_dir=str(args.output_dir),
        notes=[
            "Every candidate checks every output element and expert counts against the original schedule.",
            "Baseline is a regression reference, not an independent CPU Golden correctness proof.",
            "Dispatch and Combine keep production slot policies (Prefill Combine has two slots).",
            "Combine=0 means original Combine ownership; Dispatch experiments can leave Combine unchanged.",
            "Baseline and candidates validate each wave/core against the production wave/task oracle.",
            "Staged search can miss interactions; fullgrid only finds the best sampled pair in this run.",
            "Timer pass is separate from timer-off operator profiling; no plots are generated.",
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


def command(args, config_path, output, pair, baseline):
    result = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=16",
        "--module",
        "deep_ep_experimental.megamoe.benchmark",
        "--config",
        str(config_path),
        "--mode",
        "perf",
        "--input",
        "fp8",
        "--routing",
        "balanced",
        "--output-dir",
        str(output),
    ]
    if pair is None:
        result.append("--save-output-reference")
    else:
        result += [
            "--reference-output-dir",
            str(baseline),
            "--comparison-rtol",
            str(args.comparison_rtol),
            "--comparison-atol",
            str(args.comparison_atol),
        ]
    result += ["--timer", "--timer-ranks", "0-15", "--comm-report"]
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


def read_case(output, bs, pair):
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
            or manifest["comm_core_split"] != expected_environment
            or manifest["config"]["bs_values"] != [bs]
        ):
            raise ValueError("Benchmark manifest does not match the requested case")
        builds.append(manifest["build"])
    if any(build != builds[0] for build in builds):
        raise ValueError("Ranks loaded different installed builds")
    rank0 = output / "rank_0" / f"bs_{bs}"
    profile = json.loads((rank0 / "profiling.json").read_text(encoding="utf-8"))
    comparisons = json.loads((rank0 / "output_comparison.json").read_text(encoding="utf-8"))
    if sorted(r["rank"] for r in comparisons) != list(range(16)) or not all(r["passed"] for r in comparisons):
        raise ValueError("Missing or failed full-output regression ranks")
    per_rank = profile["per_rank_rounds_us"]
    if (
        profile["timer_pass"]
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
        operator_rank_max_mean_us=slowest,
        operator_rank_min_mean_us=statistics.mean(min(values) for values in zip(*per_rank)),
        operator_all_ranks_mean_us=statistics.mean(v for row in per_rank for v in row),
        timer_pass=False,
        selected_rounds=len(per_rank[0]),
        communication=None,
    )
    report["communication"] = json.loads(
        (output / f"bs_{bs}" / "timer" / "communication.json").read_text(encoding="utf-8")
    )
    if sorted(r["rank"] for r in report["communication"]["ranks"]) != list(range(16)):
        raise ValueError("Missing communication ranks")
    if report["communication"].get("schema_version") != 2 or not all(
        r.get("validation", {}).get("strict_wave_core") is True for r in report["communication"]["ranks"]
    ):
        raise ValueError("Missing strict per-wave/core communication validation")
    if set(report["communication"].get("operations", {})) != {"DISPATCH", "COMBINE"}:
        raise ValueError("Expected separate Dispatch and Combine mean-core-time bandwidth reports")
    for value in report["communication"]["operations"].values():
        if any(
            not 0 < value[key] < float("inf")
            for key in ("gbps_per_card_mean", "gbps_per_card_min", "gbps_per_card_max")
        ):
            raise ValueError("Nonfinite/nonpositive communication bandwidth")
    timer_comparison = json.loads((rank0 / "timer_accuracy.json").read_text(encoding="utf-8"))
    if sorted(r["rank"] for r in timer_comparison) != list(range(16)) or not all(r["passed"] for r in timer_comparison):
        raise ValueError("Missing/failed timer-on/off validation")
    audit_comparison = json.loads((rank0 / "communication_audit_accuracy.json").read_text(encoding="utf-8"))
    if sorted(r["rank"] for r in audit_comparison) != list(range(16)) or not all(r["passed"] for r in audit_comparison):
        raise ValueError("Missing/failed communication audit output validation")
    return report


def choose(records):
    if not records or any(not item["passed"] for item in records):
        raise ValueError("Cannot select from missing/failed cases")
    return min(records, key=lambda item: (item["operator_rank_max_mean_us"], item["pair"]))


def write_summary(path, launch_plan, records, selections):
    document = dict(plan=launch_plan, cases=records, selected=selections)
    (path / "summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "bs",
            "stage",
            "dispatch_intra",
            "dispatch_cross",
            "combine_intra",
            "combine_cross",
            "operator_rank_max_mean_us",
            "operator_rank_min_mean_us",
            "operator_all_ranks_mean_us",
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
                    "operator_rank_max_mean_us",
                    "operator_rank_min_mean_us",
                    "operator_all_ranks_mean_us",
                )
            }
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
    print(
        f"[CASE] BS={item['bs']} {label} E2E rank-max-mean={item['operator_rank_max_mean_us']:.3f}us "
        f"rank-min-mean={item['operator_rank_min_mean_us']:.3f}us (timer off; full-output regression passed)",
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
        print(f"[WAVE_CORE] strict per-wave/core validation passed; rank JSON/CSV paths in: {detail_index}", flush=True)


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
            item = dict(read_case(output, bs, pair), stage=stage)
            records.append(item)
            if pair is not None:
                cached[pair] = item
            print_case(item)
            write_summary(args.output_dir, launch_plan, records, selections)
            return item

        original = run_case(None, "baseline")
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
                selection_us=best["operator_rank_max_mean_us"],
                confirmation_us=confirmed["operator_rank_max_mean_us"],
                baseline_us=original["operator_rank_max_mean_us"],
                speedup=original["operator_rank_max_mean_us"] / confirmed["operator_rank_max_mean_us"],
            )
        )
        write_summary(args.output_dir, launch_plan, records, selections)
        print(
            f"[SELECTED] BS={bs} D_INTRA={best['pair'][0]} C_INTRA={best['pair'][1]} "
            f"confirmed E2E={confirmed['operator_rank_max_mean_us']:.3f}us "
            f"speedup={selections[-1]['speedup']:.3f}x against original schedule",
            flush=True,
        )
    print(f"[PASSED] {len(records)} case launches; {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
