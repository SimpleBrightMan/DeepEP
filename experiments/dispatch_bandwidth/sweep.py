# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Run a sequential CROSS-only core-count sweep on all selected local NPUs."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import uuid

if __package__:
    from . import report, run
else:
    import report
    import run

DEFAULT_HIDDEN = [3712, 4096, 6144, 8192]
DEFAULT_CORES = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28]


class SweepError(ValueError):
    """The expected experiment matrix or its measurements are incomplete."""


def csv_integers(value):
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must be nonempty and distinct")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-values", type=csv_integers, default=DEFAULT_HIDDEN)
    parser.add_argument("--core-values", type=csv_integers, default=DEFAULT_CORES)
    parser.add_argument("--bs", type=int, default=4096)
    parser.add_argument("--devices", type=csv_integers, default=list(range(16)))
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--metadata-mode", choices=("scalar", "dma", "preload"), default="dma")
    parser.add_argument("--buffer-count", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--binary",
        type=Path,
        default=run.ROOT / "build-dispatch-bandwidth/experiments/dispatch_bandwidth/dispatch_bandwidth_bench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run.ROOT / "benchmark-runs/dispatch_cross_sweep",
        help="Parent for a new unique sweep directory",
    )
    parser.add_argument("--svg", action="store_true", help="Save SVG alongside each PNG")
    parser.add_argument("--target-gbps", type=float, default=134.0, help="Reference target line; not a measured point")
    parser.add_argument(
        "--plan", action="store_true", help="Print all case commands; no writes, NPU or plotting imports"
    )
    args = parser.parse_args(argv)
    if len(args.devices) != 16:
        parser.error("The CROSS sweep requires exactly 16 distinct local devices (EP16)")
    if any(hidden < 1024 or hidden > 8192 or hidden % 64 for hidden in args.hidden_values):
        parser.error("Hidden values must be multiples of 64 in [1024,8192]")
    if any(cores < 1 or cores > 32 for cores in args.core_values):
        parser.error("Core values must be in [1,32]")
    if not math.isfinite(args.target_gbps) or args.target_gbps <= 0:
        parser.error("--target-gbps must be finite and positive")
    for hidden in args.hidden_values:
        for cores in args.core_values:
            case_args(args, hidden, cores)
    return args


def case_argv(args, hidden, cores):
    return [
        "--binary",
        str(args.binary),
        "--devices",
        ",".join(map(str, args.devices)),
        "--bs-values",
        str(args.bs),
        "--hidden",
        str(hidden),
        "--intermediate",
        str(args.intermediate),
        "--experts",
        str(args.experts),
        "--topk",
        str(args.topk),
        "--group-size",
        str(args.group_size),
        "--traffic",
        "cross",
        "--cores",
        str(cores),
        "--workers",
        f"0,0,{cores}",
        "--metadata-mode",
        args.metadata_mode,
        "--buffer-count",
        str(args.buffer_count),
        "--repeat",
        str(args.repeat),
        "--scope",
        "all",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--timeout",
        str(args.timeout),
        "--timer",
    ]


def case_args(args, hidden, cores):
    return run.parse_args(case_argv(args, hidden, cores))


def expected_configuration(args):
    return dict(
        hidden_values=list(args.hidden_values),
        core_values=list(args.core_values),
        batch=args.bs,
        world=len(args.devices),
        devices=list(args.devices),
        group_size=args.group_size,
        intermediate=args.intermediate,
        experts=args.experts,
        topk=args.topk,
        metadata_mode=args.metadata_mode,
        buffer_count=args.buffer_count,
        repeats=args.repeat,
        warmup=args.warmup,
        iterations=args.iters,
        scope="all",
        traffic_selection="cross",
    )


def make_plan(args):
    cases = []
    for hidden in args.hidden_values:
        for cores in args.core_values:
            cases.append(
                dict(
                    hidden=hidden,
                    cores=cores,
                    case_dir=f"h_{hidden}/cores_{cores}",
                    command=[
                        sys.executable,
                        "-m",
                        "experiments.dispatch_bandwidth.run",
                        *case_argv(args, hidden, cores),
                    ],
                )
            )
    return dict(
        schema_version=1,
        mode="cross_bandwidth_sweep",
        expected=expected_configuration(args),
        target_gbps=args.target_gbps,
        expected_cases=len(cases),
        cases=cases,
        note="Commands are equivalent manual launcher configurations; collection calls run_case directly.",
    )


def _require(condition, message):
    if not condition:
        raise SweepError(message)


def _finite(value, name, *, positive=True):
    _require(
        type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
        f"{name} must be finite and {'positive' if positive else 'nonnegative'}",
    )
    return value


def _same(actual, expected, name):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{name}: expected an object")
        for key, value in expected.items():
            _require(key in actual, f"{name}: missing {key}")
            _same(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"{name}: inconsistent list")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{name}[{index}]")
    elif type(expected) in (int, float):
        _finite(actual, name, positive=False)
        _require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), f"{name}: inconsistent value")
    else:
        _require(actual == expected, f"{name}: identity mismatch")


def _stats(value, name, *, positive=True):
    _require(isinstance(value, dict), f"{name}: missing distribution")
    for key in ("min", "mean", "max", "p50", "p95"):
        _finite(value.get(key), f"{name}.{key}", positive=positive)
    _require(value["min"] <= value["mean"] <= value["max"], f"{name}: mean outside range")
    _require(value["min"] <= value["p50"] <= value["p95"] <= value["max"], f"{name}: invalid percentiles")
    return value


def cross_rows(expected, rank):
    items, world, group = expected["batch"] * expected["topk"], expected["world"], expected["group_size"]
    return sum(
        (((peer + 1) * items - 1 - rank) // world - (peer * items - 1 - rank) // world)
        for peer in range(world)
        if peer // group != rank // group
    )


def validate_case(case, expected):
    _require(isinstance(case, dict) and case.get("status") == "passed", "Case did not pass")
    hidden, cores = case.get("hidden"), case.get("cores")
    _require(type(hidden) is int and hidden in expected["hidden_values"], "Unexpected case hidden size")
    _require(type(cores) is int and cores in expected["core_values"], "Unexpected case core count")
    relative = Path(case.get("case_dir", ""))
    _require(not relative.is_absolute() and ".." not in relative.parts and str(relative) != ".", "Invalid case_dir")
    _require(relative == Path(f"h_{hidden}/cores_{cores}"), "case_dir does not match case identity")
    summary = case.get("summary")
    _require(isinstance(summary, dict), "Missing native case summary")
    _same(summary.get("schema_version"), 3, "summary.schema_version")
    _same(summary.get("mode"), "cross_only", "summary.mode")
    _same(summary.get("metric"), "rank_local_class_payload_gbps", "summary.metric")
    required = {
        name: expected[name]
        for name in (
            "world",
            "group_size",
            "batch",
            "intermediate",
            "experts",
            "topk",
            "metadata_mode",
            "buffer_count",
            "repeats",
            "scope",
            "traffic_selection",
        )
    }
    required.update(
        hidden=hidden,
        cores=cores,
        worker_counts=dict(local=0, group=0, cross=cores),
        scale_elements=hidden // 32,
        row_bytes=(hidden + 255) // 256 * 256 + (hidden // 32 + 31) // 32 * 32,
    )
    _same(summary.get("configuration"), required, "summary.configuration")
    world, iterations = expected["world"], expected["iterations"]
    _same(summary.get("successful_ranks"), world, "successful_ranks")
    _same(summary.get("successful_rank_rounds"), world * iterations, "successful_rank_rounds")
    _require(summary.get("timing_validation", {}).get("status") == "consistent", "Timer/ACL validation did not pass")
    ranks = summary.get("per_rank")
    _require(isinstance(ranks, list) and len(ranks) == world, "Missing per-rank measurements")
    _require(
        all(isinstance(item, dict) and type(item.get("rank")) is int for item in ranks), "Invalid native rank identity"
    )
    _require(sorted(item.get("rank", -1) for item in ranks) == list(range(world)), "Missing or duplicate native rank")
    rank_rates, expected_bytes = [], []
    for item in ranks:
        rank = item["rank"]
        _same(item.get("device"), expected["devices"][rank], "rank.device")
        validation = item.get("validation", {})
        _require(validation.get("passed") is True, "Native rank output validation failed")
        rows = cross_rows(expected, rank)
        _same(validation.get("checked_rows"), rows, "rank.checked_rows")
        classes = item.get("classes", {})
        _require(
            classes.get("local") is None and classes.get("group") is None, "CROSS-only rank contains other traffic"
        )
        rank_rates.append(_stats(classes.get("cross"), "rank.CROSS"))
        expected_bytes.append(rows * expected["repeats"] * required["row_bytes"])
    _same(summary.get("checked_rows"), sum(cross_rows(expected, rank) for rank in range(world)), "checked_rows")
    classes = summary.get("classes", {})
    for traffic in ("local", "group"):
        _require(
            classes.get(traffic, {}).get("workers") == 0 and classes.get(traffic, {}).get("gbps_per_card") is None,
            "CROSS-only case contains other workers",
        )
    cross = classes.get("cross", {})
    _same(cross.get("workers"), cores, "CROSS.workers")
    _same(
        cross.get("rank_mean_gbps_per_card"),
        report.distribution([value["mean"] for value in rank_rates]),
        "rank mean GB/s",
    )
    _same(
        cross.get("rows_per_round_per_card"),
        report.distribution([cross_rows(expected, rank) * expected["repeats"] for rank in range(world)]),
        "CROSS.rows",
    )
    _same(cross.get("bytes_per_round_per_card"), report.distribution(expected_bytes * iterations), "CROSS.bytes")
    rounds = summary.get("per_round")
    _require(isinstance(rounds, list) and len(rounds) == iterations, "Missing measured round")
    _require(sorted(item.get("iteration", -1) for item in rounds) == list(range(iterations)), "Duplicate/missing round")
    round_rates, round_times = [], []
    for measured in rounds:
        value = measured.get("classes", {}).get("cross", {})
        _same(value.get("active_ranks"), world, "round.active_ranks")
        round_rates.append(_stats(value.get("gbps_per_card"), "round.CROSS.GB/s"))
        round_times.append(_stats(value.get("envelope_us_per_card"), "round.CROSS.envelope"))
        _same(value.get("bytes_per_card"), report.distribution(expected_bytes), "round.CROSS.bytes")
    for stat in ("min", "mean", "max"):
        _same(
            cross.get("round_rank_gbps", {}).get(stat),
            report.distribution([value[stat] for value in round_rates]),
            f"round rank {stat}",
        )
    rates = _stats(cross.get("gbps_per_card"), "CROSS.GB/s")
    _same(rates["mean"], statistics.mean(value["mean"] for value in rank_rates), "CROSS rank mean")
    _same(rates["mean"], statistics.mean(value["mean"] for value in round_rates), "CROSS round mean")
    _same(rates["min"], min(value["min"] for value in rank_rates), "CROSS minimum")
    _same(rates["max"], max(value["max"] for value in rank_rates), "CROSS maximum")
    envelope = _stats(cross.get("envelope_us_per_card"), "CROSS.envelope")
    _same(envelope["mean"], statistics.mean(value["mean"] for value in round_times), "CROSS envelope mean")
    return case


def validate_sweep(data, *, hidden_values=None, core_values=None):
    _require(isinstance(data, dict) and data.get("schema_version") == 1, "Unsupported sweep schema")
    _require(
        data.get("mode") == "cross_bandwidth_sweep" and data.get("status") == "passed", "Sweep is incomplete or failed"
    )
    _require(data.get("data_kind") in ("measured", "synthetic_test"), "Unknown data provenance")
    expected = data.get("expected", {})
    for name in ("hidden_values", "core_values", "devices"):
        values = expected.get(name)
        _require(
            isinstance(values, list)
            and values
            and all(type(n) is int for n in values)
            and len(set(values)) == len(values),
            f"Invalid expected {name}",
        )
    _require(expected.get("world") == len(expected["devices"]), "Expected world/device mismatch")
    _require(
        all(1024 <= hidden <= 8192 and hidden % 64 == 0 for hidden in expected["hidden_values"]),
        "Invalid expected hidden size",
    )
    _require(all(1 <= cores <= 32 for cores in expected["core_values"]), "Invalid expected core count")
    _require(all(device >= 0 for device in expected["devices"]), "Invalid expected device")
    _require(
        expected.get("metadata_mode") in ("scalar", "dma", "preload")
        and expected.get("traffic_selection") == "cross"
        and expected.get("scope") == "all",
        "Mislabeled sweep mode",
    )
    for name in (
        "batch",
        "world",
        "group_size",
        "intermediate",
        "experts",
        "topk",
        "buffer_count",
        "repeats",
        "iterations",
    ):
        _require(type(expected.get(name)) is int and expected[name] > 0, f"Invalid expected {name}")
    _require(
        expected["world"] % expected["group_size"] == 0 and 1 < expected["group_size"] < expected["world"],
        "Invalid expected domain layout",
    )
    _require(
        expected["world"] == 16
        and expected["experts"] % expected["world"] == 0
        and expected["experts"] // expected["world"] <= 128
        and expected["topk"] <= min(32, expected["experts"])
        and expected["batch"] <= 65536
        and 2 <= expected["buffer_count"] <= 6,
        "Invalid expected native configuration",
    )
    if hidden_values is not None:
        _require(expected["hidden_values"] == list(hidden_values), "Hidden matrix does not match requested plot")
    if core_values is not None:
        _require(expected["core_values"] == list(core_values), "Core matrix does not match requested plot")
    expected_pairs = {(hidden, cores) for hidden in expected["hidden_values"] for cores in expected["core_values"]}
    cases = data.get("cases")
    _require(isinstance(cases, list) and len(cases) == len(expected_pairs), "Missing sweep case(s)")
    _require(all(isinstance(case, dict) for case in cases), "Malformed sweep case")
    pairs = [(case.get("hidden"), case.get("cores")) for case in cases]
    _require(len(set(pairs)) == len(pairs) and set(pairs) == expected_pairs, "Duplicate or missing sweep case")
    for case in cases:
        validate_case(case, expected)
    return data


def load_sweep(path, **kwargs):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_sweep(data, **kwargs)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SweepError(f"Invalid sweep {path}: {error}") from error


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(data, path):
    fields = (
        "data_kind",
        "sweep_status",
        "hidden",
        "cores",
        "batch",
        "world",
        "metadata_mode",
        "mean_gbps_card",
        "rank_min_gbps_card",
        "rank_max_gbps_card",
        "round_mean_p50_gbps_card",
        "round_mean_p95_gbps_card",
        "mean_envelope_us",
        "min_envelope_us",
        "max_envelope_us",
        "mean_bytes_card_round",
        "case_dir",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in data["cases"]:
            if case["status"] != "passed":
                continue
            validate_case(case, data["expected"])
            value = case["summary"]["classes"]["cross"]
            rates, rounds, envelope = (
                value["rank_mean_gbps_per_card"],
                value["round_rank_gbps"]["mean"],
                value["envelope_us_per_card"],
            )
            writer.writerow(
                dict(
                    data_kind=data["data_kind"],
                    sweep_status=data["status"],
                    hidden=case["hidden"],
                    cores=case["cores"],
                    batch=data["expected"]["batch"],
                    world=data["expected"]["world"],
                    metadata_mode=data["expected"]["metadata_mode"],
                    mean_gbps_card=rates["mean"],
                    rank_min_gbps_card=rates["min"],
                    rank_max_gbps_card=rates["max"],
                    round_mean_p50_gbps_card=rounds["p50"],
                    round_mean_p95_gbps_card=rounds["p95"],
                    mean_envelope_us=envelope["mean"],
                    min_envelope_us=envelope["min"],
                    max_envelope_us=envelope["max"],
                    mean_bytes_card_round=value["bytes_per_round_per_card"]["mean"],
                    case_dir=case["case_dir"],
                )
            )


def collect_sweep(args, *, runner=None, data_kind="measured"):
    """Collect measurements only; main checks plotting dependencies before calling this."""
    parent = args.output_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ_") + uuid.uuid4().hex[:8]
    directory = parent / ("cross_" + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    plan = make_plan(args)
    _write_json(directory / "plan.json", plan)
    data = dict(
        schema_version=1,
        mode="cross_bandwidth_sweep",
        data_kind=data_kind,
        status="running",
        expected=plan["expected"],
        target_gbps=args.target_gbps,
        cases=[],
        run_dir=str(directory),
        plots=[],
    )
    path = directory / "sweep.json"
    _write_json(path, data)
    runner = runner or run.run_case
    current = None
    try:
        run._resource_snapshot(directory)
        for item in plan["cases"]:
            current = dict(hidden=item["hidden"], cores=item["cores"], case_dir=item["case_dir"], status="running")
            target = directory / item["case_dir"]
            target.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[SWEEP {len(data['cases']) + 1}/{len(plan['cases'])}] H={item['hidden']} CROSS cores={item['cores']}",
                flush=True,
            )
            summary = runner(case_args(args, item["hidden"], item["cores"]), args.bs, target)
            current.update(status="passed", summary=summary)
            validate_case(current, data["expected"])
            data["cases"].append(current)
            current = None
            _write_json(path, data)
            write_csv(data, directory / "sweep.csv")
        data["status"] = "passed"
        validate_sweep(data)
    except (Exception, KeyboardInterrupt) as error:
        if current is not None:
            current.update(status="failed", error=f"{type(error).__name__}: {error}")
            data["cases"].append(current)
        data.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        _write_json(path, data)
        write_csv(data, directory / "sweep.csv")
        if isinstance(error, KeyboardInterrupt):
            raise
        raise SweepError(f"Sweep failed; partial results retained at {path}: {error}") from error
    _write_json(path, data)
    write_csv(data, directory / "sweep.csv")
    return path


def main(argv=None):
    args = parse_args(argv)
    if args.plan:
        print(json.dumps(make_plan(args), indent=2))
        return 0
    try:
        # Check before any native launch, not after forty completed cases.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401 -- fail before launching if the rendering components are unavailable

        if __package__:
            from .plot_sweep import plot_sweep
        else:
            from plot_sweep import plot_sweep
        path = collect_sweep(args)
        try:
            outputs = plot_sweep(
                path,
                svg=args.svg,
                target_gbps=args.target_gbps,
                hidden_values=args.hidden_values,
                core_values=args.core_values,
                verify_raw=True,
            )
        except Exception as error:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(status="failed", error=f"Plotting failed: {error}")
            _write_json(path, data)
            write_csv(data, path.with_suffix(".csv"))
            raise SweepError(f"Plotting failed; measurements retained at {path}: {error}") from error
        data = load_sweep(path)
        data["plots"] = [str(output.relative_to(path.parent)) for output in outputs]
        _write_json(path, data)
        print(f"[PASSED] {len(data['cases'])} CROSS-only cases; {len(outputs)} plots; {path}", flush=True)
        return 0
    except ImportError as error:
        print(
            f"Plot dependency unavailable before launch: {error}; install requirements-sweep.txt in this environment.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
