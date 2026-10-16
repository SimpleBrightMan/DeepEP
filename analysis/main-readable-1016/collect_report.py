"""Validate four complete EP8 runs and summarize the unchanged measurement contract."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root):
    stages = ("baseline_a", "candidate_a", "candidate_b", "baseline_b")
    shapes = (32, 96, 256, 1024, 4096)
    rows = []
    for bs in shapes:
        profiles, hashes = {}, {}
        for stage in stages:
            reports, outputs = [], []
            for rank in range(8):
                case = root / "runs" / stage / f"rank_{rank}" / f"bs_{bs}"
                profile = load(case / "profiling.json")
                output = load(case / "output_regression.json")
                samples = load(case / "kernel_samples.json")
                if len(samples) != 20 or profile["recorded_rounds"] != 20 or profile["rounds"] != 10:
                    raise ValueError(f"{stage} rank={rank} BS={bs}: incomplete sampling")
                if profile["measurement"] != "full_moe" or profile["aggregation"] != "mean_of_per_round_min_rank":
                    raise ValueError(f"{stage} rank={rank} BS={bs}: wrong measurement")
                if not output["finite"] or output["rank"] != rank or output["shape"] != [bs, 7168]:
                    raise ValueError(f"{stage} rank={rank} BS={bs}: invalid output")
                reports.append(profile)
                outputs.append(output["sha256"])
            if any(report != reports[0] for report in reports[1:]):
                raise ValueError(f"{stage} BS={bs}: ranks disagree on gathered performance")
            profiles[stage] = reports[0]
            hashes[stage] = outputs
        equal = all(hashes[stage] == hashes[stages[0]] for stage in stages[1:])
        baseline = [profiles[s]["mean_us"] for s in ("baseline_a", "baseline_b")]
        candidate = [profiles[s]["mean_us"] for s in ("candidate_a", "candidate_b")]
        a, b = statistics.mean(baseline), statistics.mean(candidate)
        rows.append(dict(bs=bs, baseline_us=baseline, candidate_us=candidate,
                         baseline_mean_us=a, candidate_mean_us=b,
                         candidate_latency_change_pct=(b / a - 1) * 100,
                         output_bitwise_equal_by_sha256=equal, output_hashes=hashes,
                         baseline_rank_max_us=[profiles[s]["rank_max_mean_us"] for s in ("baseline_a", "baseline_b")],
                         candidate_rank_max_us=[profiles[s]["rank_max_mean_us"] for s in ("candidate_a", "candidate_b")]))
    artifacts = {}
    for name in ("baseline", "candidate"):
        wheels = list((root / f"{name}-wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected one {name} wheel, found {len(wheels)}")
        artifacts[name] = dict(wheel=str(wheels[0]), wheel_sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
                               source_tar_sha256=hashlib.sha256((root / f"{name}.tar").read_bytes()).hexdigest(),
                               build=load(root / "runs" / f"{name}_a" / "rank_0" / "manifest.json")["build"])
    compatibility = []
    for kind, batches in (("random", (33, 257, 4104)), ("shared", (32, 257))):
        for bs in batches:
            pairs = []
            for rank in range(8):
                paths = [root / "runs" / f"{name}_{kind}" / f"rank_{rank}" for name in ("baseline", "candidate")]
                manifests = [load(path / "manifest.json") for path in paths]
                if any(m["config"]["model"]["num_shared_experts"] != int(kind == "shared") for m in manifests):
                    raise ValueError(f"{kind} rank={rank}: wrong shared configuration")
                if any(m["routing"] != ("random" if kind == "random" else "balanced") for m in manifests):
                    raise ValueError(f"{kind} rank={rank}: wrong routing")
                outputs = [load(path / f"bs_{bs}" / "output_regression.json") for path in paths]
                pairs.append(all(out["finite"] for out in outputs) and outputs[0]["sha256"] == outputs[1]["sha256"])
            compatibility.append(dict(kind=kind, bs=bs, ranks=8, all_outputs_equal=all(pairs)))
    return dict(schema="megamoe_readable_main_abba_v1", stages=stages, ep_size=8,
                baseline_commit="bef32bca3", candidate_device_commit="0f90f5333",
                measurement="full communication MegaMoE kernel Duration; timer off; last 10 of 20",
                aggregation="mean of per-round minimum rank; rank-max also retained",
                output_regression="complete BF16 output hash for every rank/shape/run; not independent Golden",
                independent_golden="not_run", rows=rows, artifacts=artifacts, compatibility=compatibility,
                all_outputs_equal=all(row["output_bitwise_equal_by_sha256"] for row in rows)
                    and all(case["all_outputs_equal"] for case in compatibility))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = collect(args.root)
    (args.root / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, indent=2))
    if not result["all_outputs_equal"]:
        raise SystemExit("Output regression mismatch; do not promote to main")
