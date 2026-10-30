#!/usr/bin/env python3
"""Read existing rank-local files; never launch a kernel or modify results."""

import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing benchmark output directory containing rank_* folders")
    parser.add_argument("--bs", type=int, default=1024)
    args = parser.parse_args()
    cases = sorted(args.run.glob(f"rank_*/bs_{args.bs}/kernel_samples.json"),
                   key=lambda p: int(p.parent.parent.name.removeprefix("rank_")))
    if not cases:
        parser.error("No matching kernel_samples.json files")
    world = read(cases[0].parent.parent / "manifest.json")["world"]
    if [int(p.parent.parent.name.removeprefix("rank_")) for p in cases] != list(range(world)):
        parser.error(f"Expected all {world} ranks; refusing a partial cross-rank comparison")
    rows = []
    print("rank  formal-mean  diagnostic-kernel  raw-envelope  visible-span  hidden-end  records")
    for path in cases:
        rank = int(path.parent.parent.name.removeprefix("rank_"))
        samples = read(path)
        trace = read(path.parent / "urma_trace.json")
        diagnostic = samples.get("diagnostic_sample")
        if diagnostic is None:
            parser.error(f"rank {rank} has no diagnostic invocation")
        events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        end = max(e["ts"] + e["dur"] for e in events)
        visible = end - min(e["ts"] for e in events)
        envelope = trace["metadata"]["envelope_us"]
        formal = samples["durations_us"]
        duration = diagnostic["duration_us"]
        rows.append((rank, formal, duration, envelope, visible))
        print(f"{rank:4d} {statistics.mean(formal):12.3f} {duration:18.3f} {envelope:13.3f} "
              f"{visible:13.3f} {max(0.0, envelope - end):11.3f} {trace['metadata']['record_count']:8d}")
    if len({len(row[1]) for row in rows}) != 1:
        parser.error("Ranks have different formal iteration counts")
    print("All values are microseconds; aggregates cover the rank files found above.")
    print(f"formal round-max-mean: {statistics.mean(max(v) for v in zip(*(r[1] for r in rows))):.3f}")
    for column, label in [(2, "last diagnostic kernel"), (3, "last raw trace envelope"), (4, "last visible span")]:
        row = max(rows, key=lambda r: r[column])
        print(f"max {label}: {row[column]:.3f} (rank {row[0]})")
    print("Compare diagnostic-kernel/raw-envelope/visible-span on the SAME rank; formal mean is a different sample set.")


if __name__ == "__main__":
    main()
