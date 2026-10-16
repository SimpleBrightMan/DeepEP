"""Strictly validate complete EP8 output/performance reports; no missing-as-zero."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, reference, shared_not_run_reason=None):
    stages = ("baseline_a", "candidate_a", "candidate_b", "baseline_b")
    rows = []
    for bs in (32, 96, 256, 1024, 4096):
        profiles, outputs = {}, {}
        for stage in stages:
            gathered, hashes = [], []
            for rank in range(8):
                case = root / "runs" / stage / f"rank_{rank}" / f"bs_{bs}"
                profile, output = load(case / "profiling.json"), load(case / "output_regression.json")
                samples = load(case / "kernel_samples.json")
                if len(samples) != 20 or profile["recorded_rounds"] != 20 or profile["rounds"] != 10:
                    raise ValueError(f"{stage} rank={rank} BS={bs}: incomplete sampling")
                if profile["measurement"] != "full_moe" or profile["aggregation"] != "mean_of_per_round_min_rank":
                    raise ValueError(f"{stage} rank={rank} BS={bs}: wrong measurement")
                if not output["finite"] or output["rank"] != rank or output["shape"] != [bs, 7168]:
                    raise ValueError(f"{stage} rank={rank} BS={bs}: invalid output")
                gathered.append(profile)
                hashes.append(output["sha256"])
            if any(p != gathered[0] for p in gathered[1:]):
                raise ValueError(f"{stage} BS={bs}: ranks disagree on gathered profile")
            profiles[stage], outputs[stage] = gathered[0], hashes
        a = [profiles[s]["mean_us"] for s in ("baseline_a", "baseline_b")]
        b = [profiles[s]["mean_us"] for s in ("candidate_a", "candidate_b")]
        rows.append(dict(bs=bs, baseline_us=a, candidate_us=b,
            baseline_mean_us=statistics.mean(a), candidate_mean_us=statistics.mean(b),
            latency_change_pct=(statistics.mean(b) / statistics.mean(a) - 1) * 100,
            outputs_equal=all(outputs[s] == outputs[stages[0]] for s in stages), output_hashes=outputs,
            rank_max_us={s: profiles[s]["rank_max_mean_us"] for s in stages}))
    a8w8 = []
    for rank in range(8):
        pair = [load(root / "a8w8" / name / f"rank_{rank}.json") for name in ("baseline", "candidate")]
        if any(p["rank"] != rank or p["ep"] != 8 or p["mma"] != "fp8xfp8" for p in pair):
            raise ValueError(f"A8W8 rank={rank}: wrong configuration")
        expected = [(bs, kind) for bs in (1, 17, 96, 257, 4096, 17) for kind in ("bf16", "fp8")]
        if any([(c["bs"], c["input"]) for c in p["cases"]] != expected for p in pair):
            raise ValueError(f"A8W8 rank={rank}: incomplete cases")
        a8w8.append(dict(rank=rank, outputs_equal=pair[0]["cases"] == pair[1]["cases"]))
    layout = [root / f"layout-{name}.csv" for name in ("baseline", "candidate")]
    if layout[0].read_bytes() != layout[1].read_bytes() or len(layout[0].read_text().splitlines()) != 288:
        raise ValueError("Workspace baseline/candidate allocation regression")
    artifacts = {}
    for name, directory in (("baseline", reference / "candidate-wheels"), ("candidate", root / "wheels")):
        wheels = list(directory.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected one {name} wheel")
        artifacts[name] = dict(wheel_sha256=digest(wheels[0]),
            build=load(root / "runs" / f"{name}_a" / "rank_0" / "manifest.json")["build"])
    compatibility = []
    for kind, batches in (("random", (33, 257, 4104)), ("shared", (32, 257))):
        if kind == "shared" and shared_not_run_reason:
            if any((root / "runs" / f"{name}_shared").exists() for name in ("baseline", "candidate")):
                raise ValueError("Shared results exist: inspect them instead of overriding with not_run")
            compatibility.extend(dict(kind=kind, bs=bs, ranks=8, status="not_run_resource_busy",
                outputs_equal=None, reason=shared_not_run_reason) for bs in batches)
            continue
        for bs in batches:
            equal = []
            for rank in range(8):
                paths = [root / "runs" / f"{name}_{kind}" / f"rank_{rank}" for name in ("baseline", "candidate")]
                manifests = [load(p / "manifest.json") for p in paths]
                if any(m["config"]["model"]["num_shared_experts"] != int(kind == "shared") for m in manifests):
                    raise ValueError(f"{kind} rank={rank}: wrong shared configuration")
                if any(m["routing"] != ("random" if kind == "random" else "balanced") for m in manifests):
                    raise ValueError(f"{kind} rank={rank}: wrong routing")
                outputs = [load(p / f"bs_{bs}" / "output_regression.json") for p in paths]
                equal.append(all(o["finite"] for o in outputs) and outputs[0]["sha256"] == outputs[1]["sha256"])
            compatibility.append(dict(kind=kind, bs=bs, ranks=8, status="complete", outputs_equal=all(equal)))
    tested_equal = all(row["outputs_equal"] for row in rows) and all(r["outputs_equal"] for r in a8w8)
    tested_equal = tested_equal and all(r["outputs_equal"] for r in compatibility if r["status"] == "complete")
    return dict(schema="megamoe_maintainable_1016_v1", reference_git="18ada75b1",
        reference_device_code="0f90f5333", candidate_commit="c7a774c27", ep=8, stages=stages,
        source_overlay_sha256="3051f8cf86a35f70bc7ad03802f82f3f113b7e505cf5ccf967c8365610fc6ac9",
        final_source_note="Compared with compiled overlay, final C++ edits only correct three comment lines",
        measurement="complete communication kernel Duration, timer-off, last 10 of 20",
        aggregation="mean of per-round rank-min; rank-max separately retained",
        independent_golden="not_run", rows=rows, a8w8=a8w8,
        workspace_layout_cases=288, workspace_layout_sha256=digest(layout[0]), artifacts=artifacts,
        compatibility=compatibility,
        device_matrix_complete=shared_not_run_reason is None, all_tested_outputs_equal=tested_equal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--shared-not-run-reason", help="Explicit incomplete status; never an accuracy PASS")
    args = parser.parse_args()
    report = collect(args.root, args.reference, args.shared_not_run_reason)
    (args.root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(rows=report["rows"], all_tested_outputs_equal=report["all_tested_outputs_equal"],
                          device_matrix_complete=report["device_matrix_complete"]), indent=2))
    if not report["all_tested_outputs_equal"]:
        raise SystemExit("Output mismatch: do not promote candidate")
    if not report["device_matrix_complete"]:
        raise SystemExit("Incomplete device matrix: shared-expert regression was not run")
