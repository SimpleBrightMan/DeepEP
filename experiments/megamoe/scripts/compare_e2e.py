"""Compare two existing full-EP8 batch_launch runs; never launches an NPU task."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measurements(directory, batches):
    plan = read(directory / "launch.json")
    if len(plan["hosts"]) != 1 or plan["mode"] != "perf" or plan.get("measurement", "full_moe") != "full_moe":
        raise ValueError(f"{directory}: need a full-MoE EP8 performance run")
    if not set(batches).issubset(plan["config"]["bs_values"]):
        raise ValueError(f"{directory}: requested BS missing from plan")
    data = {}
    for bs in batches:
        source = directory / f"bs_{bs}/rank_0/bs_{bs}/profiling.json"
        report = read(source)
        if report["metric"] != "megamoe_kernel_duration_us" or report.get("timer_pass") is not False:
            raise ValueError(f"{source}: expected full-MoE timer-off performance samples")
        rows = report["per_rank_rounds_us"]
        if len(rows) != 8 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f"{source}: need eight equally sampled ranks")
        if any(not math.isfinite(x) or x <= 0 for row in rows for x in row):
            raise ValueError(f"{source}: invalid duration")
        slow = statistics.mean(max(values) for values in zip(*rows))
        fast = statistics.mean(min(values) for values in zip(*rows))
        if not math.isclose(slow, report["rank_max_mean_us"], rel_tol=1e-9) or not math.isclose(fast, report["mean_us"], rel_tol=1e-9):
            raise ValueError(f"{source}: saved rank aggregation does not match samples")
        data[bs] = dict(rank_max=slow, rank_min=fast, rounds=len(rows[0]),
                        recorded=report["recorded_rounds"], selection=report["iteration_selection"])
    return plan, read(directory / "artifacts.json"), data


def compare(before, after, batches):
    left, la, a = measurements(before, batches)
    right, ra, b = measurements(after, batches)
    for key in ("config", "input", "routing", "timer"):
        if left[key] != right[key]:
            raise ValueError(f"Cannot compare different {key}")
    if la["cann"] != ra["cann"]:
        raise ValueError("Cannot compare different CANN builds")
    for key in ("torch", "torch_npu"):
        if la["build"]["megamoe"][key] != ra["build"]["megamoe"][key]:
            raise ValueError(f"Cannot compare different {key}")
    lines = [f"Before commit: {la['commit']}; wheel SHA256: {la['wheel_sha256']}",
             f"After commit: {ra['commit']}; wheel SHA256: {ra['wheel_sha256']}",
             "Metric: full MegaMoE kernel Duration (not Python wall time), timer-off; trace collected separately.",
             "", "| BS | Before rank-max μs | After rank-max μs | Reduction μs | Reduction % | Before/after rank-min μs |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for bs in batches:
        if any(a[bs][k] != b[bs][k] for k in ("rounds", "recorded", "selection")):
            raise ValueError(f"BS={bs}: different sample selection")
        delta = a[bs]["rank_max"] - b[bs]["rank_max"]
        lines.append(f"| {bs} | {a[bs]['rank_max']:.3f} | {b[bs]['rank_max']:.3f} | {delta:.3f} | "
                     f"{100*delta/a[bs]['rank_max']:.3f}% | {a[bs]['rank_min']:.3f} / {b[bs]['rank_min']:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--bs-values", default="32,96,256,1024,4096")
    args = parser.parse_args()
    batches = [int(x) for x in args.bs_values.split(",")]
    if not batches or len(set(batches)) != len(batches) or any(x <= 0 for x in batches):
        parser.error("BS values must be positive and distinct")
    print(compare(args.before, args.after, batches))


if __name__ == "__main__":
    main()
