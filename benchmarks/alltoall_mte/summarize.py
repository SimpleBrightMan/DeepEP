# Copyright (c) 2026, SimpleBrightMan

"""Aggregate per-rank All-to-All JSON reports into the requested bandwidth table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from . import plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory containing rank_XX.json")
    parser.add_argument("--output", type=Path, default=None, help="defaults to --input")
    parser.add_argument("--title", default="")
    return parser.parse_args(argv)


def load_reports(directory: Path):
    reports = []
    for path in sorted(directory.glob("rank_*.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        raise SystemExit(f"no rank_*.json under {directory}")
    ranks = sorted(report["rank"] for report in reports)
    if ranks != list(range(len(ranks))):
        raise SystemExit(f"rank files are not contiguous from 0: {ranks}")
    first = reports[0]
    for report in reports[1:]:
        for key in ("world_size", "group_size", "token_bytes", "iters", "warmup"):
            if report[key] != first[key]:
                raise SystemExit(f"rank {report['rank']} disagrees on {key}")
        if report.get("timing", "event") != first.get("timing", "event"):
            raise SystemExit(f"rank {report['rank']} disagrees on timing")
    return reports


def selected_ms(case):
    """Use the baseline-corrected host wall clock when selected, else NPU events."""
    if case.get("timing") == "host":
        return (
            float(case.get("host_net_median_ms", case.get("host_median_ms", case["median_ms"]))),
            float(case.get("host_net_min_ms", case.get("host_min_ms", case["min_ms"]))),
        )
    return float(case["median_ms"]), float(case["min_ms"])


def aggregate(reports):
    cases = {}
    for report in reports:
        for case in report["cases"]:
            key = (int(case["tokens"]), str(case["mode"]))
            cases.setdefault(key, {"case": case, "rank_medians": {}, "rank_mins": {}})
            median_ms, min_ms = selected_ms(case)
            cases[key]["rank_medians"][report["rank"]] = median_ms
            cases[key]["rank_mins"][report["rank"]] = min_ms
    output = []
    for (tokens, mode), item in sorted(cases.items(), key=lambda entry: (entry[0][0], plan.MODES.index(entry[0][1]))):
        medians = list(item["rank_medians"].values())
        max_ms = max(medians)
        mean_ms = statistics.fmean(medians)
        payload = int(item["case"]["bytes"])
        output.append(
            {
                "tokens": tokens,
                "mode": mode,
                "label": plan.MODE_LABELS[mode],
                "peers": int(item["case"]["peers"]),
                "bytes_per_rank": payload,
                "bytes_total": payload * len(medians),
                "timing": item["case"].get("timing", "event"),
                "max_rank_median_ms": max_ms,
                "mean_rank_median_ms": mean_ms,
                "min_rank_median_ms": min(medians),
                "bandwidth_gbps": plan.bandwidth_gbps(payload, max_ms),
                "mean_rank_bandwidth_gbps": plan.bandwidth_gbps(payload, mean_ms),
                "per_destination_gbps": plan.bandwidth_gbps(payload, max_ms) / int(item["case"]["peers"]),
                "rank_medians_ms": {str(rank): ms for rank, ms in sorted(item["rank_medians"].items())},
            }
        )
    return output


def write_outputs(args, reports, cases):
    destination = args.output or args.input
    destination.mkdir(parents=True, exist_ok=True)
    timings = sorted({case["timing"] for case in cases})
    selected_timing = timings[0] if len(timings) == 1 else "mixed"
    table = plan.build_table(cases)
    per_link = plan.build_per_link_table(cases)
    summary = {
        "schema": 1,
        "kind": "alltoall-mte",
        "title": args.title,
        "world_size": reports[0]["world_size"],
        "group_size": reports[0]["group_size"],
        "token_bytes": reports[0]["token_bytes"],
        "ranks": [report["rank"] for report in reports],
        "timing": selected_timing,
        "checked": all(report.get("checked", False) for report in reports),
        "cases": cases,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tokens",
                "mode",
                "label",
                "peers",
                "bytes_per_rank",
                "timing",
                "max_rank_median_ms",
                "mean_rank_median_ms",
                "bandwidth_gbps",
                "mean_rank_bandwidth_gbps",
                "per_destination_gbps",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow({name: case[name] for name in writer.fieldnames})

    markdown = [
        f"# MTE All-to-All: {args.title}" if args.title else "# MTE All-to-All",
        "",
        f"World {reports[0]['world_size']}, group {reports[0]['group_size']}, "
        f"token {reports[0]['token_bytes']} B, ranks {len(reports)}, "
        f"timing={selected_timing}, checked={summary['checked']}.",
        "",
        "Timing: " + (
            "automatic per-case selection (event when it agrees with host, otherwise launch-subtracted host); "
            "see the timing column in summary.csv."
            if selected_timing == "mixed"
            else "launch-subtracted host wall clock around a device-wide synchronize."
            if selected_timing == "host"
            else "NPU event delta on Runtime.stream (excludes host launch; verify against host on new SDKs)."
        ),
        "",
        "",
        "Aggregate per-rank egress bandwidth (`max-rank median`). "
        "`组内下行` excludes the local card; `全量` includes it.",
        "",
        table,
        "",
        "Per-destination share:",
        "",
        per_link,
        "",
    ]
    (destination / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return destination


def main(argv=None):
    args = parse_args(argv)
    reports = load_reports(args.input)
    cases = aggregate(reports)
    destination = write_outputs(args, reports, cases)
    print(f"summary written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
