#!/usr/bin/env python3
# Copyright (c) 2026, SimpleBright_Man

"""Run the staged EP16 MegaMoE hierarchy prototype, one torchrun per BS."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("dispatch_cross", "dispatch_intra", "combine_intra", "combine_cross")
CORE_ENV = ("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", "DEEPEP_MEGAMOE_COMBINE_INTRA_CORES")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bs-values", default="2048,4096,8192")
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
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per BS process group")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print commands without device access or files")
    args = parser.parse_args(argv)
    try:
        args.bs_values = [int(value) for value in args.bs_values.split(",")]
    except ValueError:
        parser.error("BS values must be comma-separated integers")
    if (
        not args.bs_values
        or len(set(args.bs_values)) != len(args.bs_values)
        or any(not 384 <= bs <= 8192 or bs % 128 for bs in args.bs_values)
    ):
        parser.error("BS values must be distinct multiples of 128 in [384,8192]")
    if not 1024 <= args.hidden <= 8192 or args.hidden % 256:
        parser.error("hidden must be a multiple of 256 in [1024,8192]")
    if not 256 <= args.intermediate <= 4096 or args.intermediate % 128:
        parser.error("intermediate must be a multiple of 128 in [256,4096]")
    if args.experts % 16 or not 16 <= args.experts <= 2048 or not 1 <= args.topk <= min(32, args.experts):
        parser.error("Require E divisible by 16, E<=2048 and K in [1,min(32,E)]")
    if args.warmup < 0 or min(args.iters, args.timeout) < 1 or args.seed < 0:
        parser.error("Require warmup>=0, iters/timeout>=1 and seed>=0")
    if any(not math.isfinite(value) or value < 0 for value in (args.relative_rmse, args.max_abs)):
        parser.error("Comparison tolerances must be finite and nonnegative")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        args.output_dir = ROOT / "benchmark-runs" / "megamoe_hierarchical" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    args.output_dir = args.output_dir.resolve()
    return args


def command(args, bs):
    result = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=16",
        "--module",
        "deep_ep_experimental.megamoe.hierarchy_benchmark",
        "--bs",
        str(bs),
        "--output",
        str(args.output_dir / f"bs_{bs}"),
    ]
    for name in (
        "hidden",
        "intermediate",
        "experts",
        "topk",
        "routing",
        "input",
        "warmup",
        "iters",
        "seed",
        "relative_rmse",
        "max_abs",
    ):
        result += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return result


def plan(args):
    return dict(
        schema_version=1,
        prototype="staged_hierarchical_dedup_fp32_partial",
        ep_size=16,
        cores=28,
        group_size=4,
        bs_values=args.bs_values,
        output_dir=str(args.output_dir),
        commands=[command(args, bs) for bs in args.bs_values],
        timeout_seconds_per_bs=args.timeout,
        inherited_variables_cleared=list(CORE_ENV),
        notes=[
            "One fresh torchrun per BS, comparing original direct schedule and hierarchy in each case.",
            "Prototype uses multiple kernels, stream synchronization and host SHMEM stage barriers.",
            "Stage payload bandwidth includes synchronization/reduction, not pure link busy time.",
            "ACL spans include all kernels and host stage gaps; route-plan setup is reported separately.",
            "Full-output/count regression and independently counted payloads must pass on all 16 ranks.",
            "No plots or raw per-core wave timer traces are generated.",
        ],
    )


def environment(base=None):
    result = dict(os.environ if base is None else base)
    for key in CORE_ENV:
        result.pop(key, None)
    result.setdefault("OMP_NUM_THREADS", "1")
    return result


def execute(argv, cwd, logfile, timeout):
    with logfile.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(argv=argv, cleared_environment=list(CORE_ENV))) + "\n")
        stream.flush()
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                pass
            raise
    if status:
        raise RuntimeError(f"BS case failed (status={status}); inspect {logfile}")


def positive(value, label, zero=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"Invalid {label}: {value}")
    return value


def read_case(args, bs):
    directory = args.output_dir / f"bs_{bs}"
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("prototype") != "staged_hierarchical_dedup_fp32_partial":
        raise ValueError("Unknown hierarchy result schema")
    expected_config = {
        name: getattr(args, name)
        for name in (
            "hidden",
            "intermediate",
            "experts",
            "topk",
            "routing",
            "input",
            "warmup",
            "iters",
            "seed",
            "relative_rmse",
            "max_abs",
        )
    }
    expected_config["bs"] = bs
    if any(report["config"].get(key) != value for key, value in expected_config.items()):
        raise ValueError("Hierarchy result config differs from requested case")
    checks = report["accuracy"]
    if sorted(item["rank"] for item in checks) != list(range(16)) or not all(
        item["passed"] and item["counts_equal"] for item in checks
    ):
        raise ValueError("Missing or failed full-output/count comparison ranks")
    repeats = report["repeat_accuracy"]
    if len(repeats) != 16 or not all(item["passed"] for item in repeats):
        raise ValueError("Missing or failed repeated-call comparison")
    dynamic = report["dynamic_accuracy"]
    if len(dynamic) != 16 or not all(item["passed"] for item in dynamic):
        raise ValueError("Missing or failed changed-input/weight comparison")
    if report["timer"] is not False or len(report["plan_wall_ms"]) != 16:
        raise ValueError("Missing plan times or unexpected timer-enabled run")
    for value in report["plan_wall_ms"]:
        positive(value, "plan wall time", zero=True)
    for mode in ("direct", "hierarchy"):
        if report[mode]["ranks"] != 16 or report[mode]["rounds"] != args.iters:
            raise ValueError("Timing ranks/rounds differ from requested case")
        for metric in ("acl_span_us", "wall_us"):
            positive(report[mode][metric]["rank_max_mean"], f"{mode} {metric}")
    for stage in STAGES:
        for metric in ("mean", "minimum", "maximum"):
            positive(report["stage_payload_gbps"][stage][metric], f"{stage} {metric}", zero=True)
    for rank in range(16):
        local = json.loads((directory / f"rank_{rank}" / "results.json").read_text(encoding="utf-8"))
        if (
            local["rank"] != rank
            or len(local["hierarchy_samples"]) != args.iters
            or len(local["direct_samples"]) != args.iters
        ):
            raise ValueError("Missing rank/iteration samples")
        for row in local["hierarchy_samples"]:
            positive(row["acl_span_us"], "rank ACL span")
            positive(row["wall_us"], "rank wall span")
            for name, expected in local["oracle"].items():
                if name.endswith("_bytes") and row["stages"].get(name) != expected:
                    raise ValueError(f"rank={rank}: repeated native payload differs from oracle for {name}")
    return report


def write_summary(path, launch_plan, reports):
    (path / "summary.json").write_text(
        json.dumps(dict(plan=launch_plan, cases=reports), indent=2) + "\n", encoding="utf-8"
    )
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "bs",
            "routing",
            "input",
            "direct_acl_span_us",
            "hierarchy_acl_span_us",
            "acl_span_speedup",
            "direct_wall_us",
            "hierarchy_wall_us",
            "plan_max_ms",
            "rebuild_every_call_estimate_us",
            *(name + "_stage_GBps" for name in STAGES),
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            direct = report["direct"]["acl_span_us"]["rank_max_mean"]
            hierarchy = report["hierarchy"]["acl_span_us"]["rank_max_mean"]
            row = dict(
                bs=report["config"]["bs"],
                routing=report["config"]["routing"],
                input=report["config"]["input"],
                direct_acl_span_us=direct,
                hierarchy_acl_span_us=hierarchy,
                acl_span_speedup=direct / hierarchy,
                direct_wall_us=report["direct"]["wall_us"]["rank_max_mean"],
                hierarchy_wall_us=report["hierarchy"]["wall_us"]["rank_max_mean"],
                plan_max_ms=max(report["plan_wall_ms"]),
                rebuild_every_call_estimate_us=report["rebuild_every_call_estimate_us"],
            )
            row.update({name + "_stage_GBps": report["stage_payload_gbps"][name]["mean"] for name in STAGES})
            writer.writerow(row)


def print_case(report):
    bs = report["config"]["bs"]
    direct, hierarchy = (report[name]["acl_span_us"]["rank_max_mean"] for name in ("direct", "hierarchy"))
    print(
        f"[CASE] BS={bs} direct={direct:.3f}us hierarchy={hierarchy:.3f}us "
        f"speedup={direct / hierarchy:.3f}x (ACL span, all kernels + stage gaps); "
        f"hierarchy wall={report['hierarchy']['wall_us']['rank_max_mean']:.3f}us "
        f"plan max={max(report['plan_wall_ms']):.3f}ms",
        flush=True,
    )
    for name in STAGES:
        value = report["stage_payload_gbps"][name]
        print(
            f"[STAGE] {name} GB/s/card mean={value['mean']:.3f} "
            f"min/max={value['minimum']:.3f}/{value['maximum']:.3f} "
            "(payload / synchronous stage wall time)",
            flush=True,
        )


def main(argv=None):
    args = parse_args(argv)
    launch_plan = plan(args)
    if args.plan:
        print(json.dumps(launch_plan, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("Run on Linux with 16 allocated UBX devices; --plan requires no device")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.json").write_text(json.dumps(launch_plan, indent=2) + "\n", encoding="utf-8")
    snapshot = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, check=True)
    (args.output_dir / "npu_smi.txt").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
    print(
        f"[RESOURCE] Read-only NPU snapshot: {args.output_dir / 'npu_smi.txt'}; all 16 devices must be allocated to this job.",
        flush=True,
    )
    reports = []
    for bs in args.bs_values:
        log = args.output_dir / f"bs_{bs}.log"
        print(f"[RUN] BS={bs}; {log}", flush=True)
        execute(command(args, bs), args.output_dir, log, args.timeout)
        report = read_case(args, bs)
        reports.append(report)
        write_summary(args.output_dir, launch_plan, reports)
        print_case(report)
    print(f"[PASSED] {len(reports)} BS cases with all 16 ranks; {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
