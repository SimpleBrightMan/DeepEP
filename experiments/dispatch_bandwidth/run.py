# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Launch one native process per local NPU for concurrent or CROSS-only Dispatch."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

if __package__:
    from . import report
else:
    import report

ROOT = Path(__file__).resolve().parents[2]
CROSS_LAUNCH_GROUPS = 28


def _csv_integers(value):
    try:
        if not value or any(not part.strip() for part in value.split(",")):
            raise ValueError
        return [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "build-dispatch-bandwidth/experiments/dispatch_bandwidth/dispatch_bandwidth_bench",
    )
    parser.add_argument(
        "--devices", type=_csv_integers, default=list(range(16)), help="Native ACL device IDs, one per rank"
    )
    parser.add_argument("--bs-values", type=_csv_integers, default=[4096])
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4, help="Contiguous rank IDs in one direct-link domain")
    parser.add_argument("--cores", type=int, default=28)
    parser.add_argument("--traffic", choices=("all", "cross"), default="all")
    parser.add_argument(
        "--workers", type=_csv_integers, help="Dedicated LOCAL,GROUP,CROSS cores; CROSS-only defaults to 0,0,cores"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--buffer-count", type=int, default=6)
    parser.add_argument("--scope", choices=("all", "first-expert"), default="all")
    parser.add_argument("--metadata-mode", choices=("scalar", "dma", "preload"), default="dma")
    parser.add_argument("--timer", action="store_true", help="Export per-rank worker CSV and Chrome trace JSON")
    parser.add_argument("--timeout", type=float, default=600, help="Maximum seconds for each fresh BS process group")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark-runs/dispatch_bandwidth",
        help="Parent directory under which a unique run directory is created",
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print native commands without writes, ports or NPU queries"
    )
    args = parser.parse_args(argv)
    if args.workers is None:
        args.workers = [0, 0, args.cores] if args.traffic == "cross" else [2, 5, 21]
    world = len(args.devices)
    if not 4 <= world <= 16 or len(set(args.devices)) != world or any(device < 0 for device in args.devices):
        parser.error("--devices must contain 4..16 distinct nonnegative ACL device IDs")
    if not 1 < args.group_size < world or world % args.group_size:
        parser.error("--group-size must divide world and leave nonempty GROUP and CROSS classes")
    minimum_cores = 1 if args.traffic == "cross" else 3
    maximum_cores = CROSS_LAUNCH_GROUPS if args.traffic == "cross" else 32
    if not minimum_cores <= args.cores <= maximum_cores or len(args.workers) != 3 or sum(args.workers) != args.cores:
        parser.error(f"--workers must sum to --cores, with --cores in [{minimum_cores},{maximum_cores}]")
    if args.traffic == "cross":
        if args.workers != [0, 0, args.cores]:
            parser.error("--traffic cross requires --workers 0,0,cores")
    elif any(count < 1 for count in args.workers):
        parser.error("--traffic all requires three positive worker counts")
    if (
        not args.bs_values
        or len(set(args.bs_values)) != len(args.bs_values)
        or any(bs < 1 or bs > 65536 for bs in args.bs_values)
    ):
        parser.error("--bs-values requires distinct values in [1,65536]")
    if not 1024 <= args.hidden <= 8192 or args.hidden % 64:
        parser.error("--hidden must be divisible by 64 and in [1024,8192]")
    if not 256 <= args.intermediate <= 4096 or args.intermediate % 128:
        parser.error("--intermediate must be divisible by 128 and in [256,4096]")
    if (
        not world <= args.experts <= 16384
        or args.experts % world
        or args.experts // world > 128
        or not 1 <= args.topk <= min(32, args.experts)
    ):
        parser.error("Experts must divide world; top-k must be in [1,min(32,experts)]")
    if args.warmup < 0 or min(args.iters, args.repeat) < 1 or not 2 <= args.buffer_count <= 6:
        parser.error("warmup >= 0, iters/repeat >= 1 and buffer-count in [2,6] are required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    return args


def native_command(args, *, rank, bs, endpoint, directory):
    values = {
        "rank": rank,
        "world": len(args.devices),
        "device": args.devices[rank],
        "endpoint": endpoint,
        "output": Path(directory) / f"rank_{rank}.json",
        "bs": bs,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "experts": args.experts,
        "topk": args.topk,
        "group-size": args.group_size,
        "cores": args.cores,
        "workers": ",".join(map(str, args.workers)),
        "buffer-count": args.buffer_count,
        "repeat": args.repeat,
        "scope": args.scope,
        "traffic": args.traffic,
        "metadata-mode": args.metadata_mode,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    command = [str(args.binary.resolve())]
    for name, value in values.items():
        command.extend(("--" + name, str(value)))
    if args.timer:
        command.append("--timer")
    return command


def make_plan(args, directory):
    cases = []
    for bs in args.bs_values:
        case = Path(directory) / f"bs_{bs}"
        endpoint = "tcp://127.0.0.1:<fresh-free-port>"
        cases.append(
            dict(
                batch=bs,
                endpoint=endpoint,
                commands=[
                    native_command(args, rank=rank, bs=bs, endpoint=endpoint, directory=case)
                    for rank in range(len(args.devices))
                ],
            )
        )
    return dict(
        mode="cross_only" if args.traffic == "cross" else "concurrent_local_group_cross",
        traffic_selection=args.traffic,
        metadata_mode=args.metadata_mode,
        world=len(args.devices),
        devices=args.devices,
        workers=dict(zip(report.TRAFFIC, args.workers)),
        output_dir=str(directory),
        cases=cases,
        note=(
            "Only CROSS workers execute in each kernel."
            if args.traffic == "cross"
            else "One kernel concurrently executes all three classes."
        )
        + " Each BS starts fresh native processes.",
    )


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _endpoint():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _stop_owned(processes):
    """Only Popen objects created by this launcher are eligible for termination."""
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in alive:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _resource_snapshot(directory):
    try:
        completed = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=False)
        snapshot = f"exit_code={completed.returncode}\n{completed.stdout}\n{completed.stderr}"
    except (OSError, subprocess.TimeoutExpired) as error:
        snapshot = f"npu-smi info unavailable: {error}\n"
    path = Path(directory) / "npu_smi.txt"
    path.write_text(snapshot, encoding="utf-8")
    print(f"[RESOURCE] Read-only npu-smi snapshot: {path}; selected devices must be allocated to this job.", flush=True)


def run_case(args, bs, directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    endpoint = _endpoint()
    commands = [
        native_command(args, rank=rank, bs=bs, endpoint=endpoint, directory=directory)
        for rank in range(len(args.devices))
    ]
    _write_json(
        directory / "launch.json",
        dict(batch=bs, metadata_mode=args.metadata_mode, endpoint=endpoint, commands=commands),
    )
    processes, logs = [], []
    started = time.monotonic()
    try:
        for rank, command in enumerate(commands):
            stream = (directory / f"rank_{rank}.log").open("w", encoding="utf-8")
            logs.append(stream)
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT))
        while True:
            exits = [process.poll() for process in processes]
            failed = [(rank, code) for rank, code in enumerate(exits) if code is not None and code != 0]
            if failed:
                raise RuntimeError(f"Native worker(s) failed: {failed}; inspect {directory}/rank_*.log")
            if all(code is not None for code in exits):
                break
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(f"BS={bs} exceeded {args.timeout:g}s; inspect {directory}/rank_*.log")
            time.sleep(0.1)
    finally:
        _stop_owned(processes)
        for stream in logs:
            stream.close()
    reports = report.load_reports(directory, len(args.devices))
    expected = dict(
        world=len(args.devices),
        cores=args.cores,
        group_size=args.group_size,
        batch=bs,
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        topk=args.topk,
        scope=args.scope,
        traffic_selection=args.traffic,
        metadata_mode=args.metadata_mode,
        buffer_count=args.buffer_count,
        repeats=args.repeat,
        worker_counts=dict(zip(report.TRAFFIC, args.workers)),
    )
    for rank, raw in enumerate(reports):
        if raw.get("device") != args.devices[rank] or any(raw.get(key) != value for key, value in expected.items()):
            raise report.ReportError(f"rank {rank}: report does not match the requested native launch")
    summary = report.summarize_reports(reports, expected_world=len(args.devices), expected_iterations=args.iters)
    _write_json(directory / "summary.json", summary)
    if args.timer:
        for raw in reports:
            report.export_timers(raw, directory / "timer")
    return summary


def print_summary(summary):
    config = summary["configuration"]
    label = "CROSS_ONLY" if config["traffic_selection"] == "cross" else "CONCURRENT"
    activity = (
        "Only CROSS workers run." if config["traffic_selection"] == "cross" else "LOCAL/GROUP/CROSS run in one kernel."
    )
    print(
        f"[{label}] BS={config['batch']} scope={config['scope']} metadata_mode={config['metadata_mode']}; {activity}",
        flush=True,
    )
    for traffic in report.TRAFFIC:
        value = summary["classes"][traffic]
        rates, times, amounts = (
            value["rank_mean_gbps_per_card"],
            value["envelope_us_per_card"],
            value["bytes_per_round_per_card"],
        )
        if rates is None:
            print(f"{traffic.upper():5} GB/s/card=N/A (zero rows); workers={value['workers']}", flush=True)
            continue
        rounds = value["round_rank_gbps"]["mean"]
        print(
            f"{traffic.upper():5} GB/s/card mean={rates['mean']:.3f} rank-min/max={rates['min']:.3f}/{rates['max']:.3f} "
            f"round-mean p50/p95={rounds['p50']:.3f}/{rounds['p95']:.3f} "
            f"bytes/card/round={amounts['min']:g}..{amounts['max']:g} "
            f"class-envelope-us mean={times['mean']:.3f} min/max={times['min']:.3f}/{times['max']:.3f} "
            f"workers={value['workers']}",
            flush=True,
        )


def main(argv=None):
    args = parse_args(argv)
    run_id = (
        "ep"
        + str(len(args.devices))
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    directory = args.output_dir.resolve() / run_id
    plan = make_plan(args, directory)
    if args.plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not sys.platform.startswith("linux"):
        raise SystemExit("Native NPU execution requires Linux; --plan is available on CPU hosts")
    if not args.binary.is_file():
        raise SystemExit(f"Native binary not found: {args.binary}; build it first or pass --binary")
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "plan.json", plan)
    _resource_snapshot(directory)
    combined = dict(
        schema_version=3,
        mode=plan["mode"],
        metadata_mode=args.metadata_mode,
        status="running",
        run_dir=str(directory),
        requested_ranks=len(args.devices) * len(args.bs_values),
        successful_ranks=0,
        successful_rank_rounds=0,
        cases=[],
    )
    exit_code = 0
    try:
        for bs in args.bs_values:
            summary = run_case(args, bs, directory / f"bs_{bs}")
            print_summary(summary)
            combined["cases"].append(dict(batch=bs, status="passed", summary=summary))
            combined["successful_ranks"] += summary["successful_ranks"]
            combined["successful_rank_rounds"] += summary["successful_rank_rounds"]
            _write_json(directory / "summary.json", combined)
        combined["status"] = "passed"
    except KeyboardInterrupt:
        combined.update(status="interrupted", error="Interrupted by user")
        exit_code = 130
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        combined.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(f"[FAILED] {combined['error']}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        _write_json(directory / "summary.json", combined)
    print(
        f"[{combined['status'].upper()}] accepted ranks={combined['successful_ranks']}/{combined['requested_ranks']} "
        f"rank-rounds={combined['successful_rank_rounds']}; {directory / 'summary.json'}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
