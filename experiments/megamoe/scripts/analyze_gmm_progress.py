"""Classify observed AIC progression; gaps are NOT proof of hardware idle or lifecycle cost."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def normalize(data):
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    origin = min((e["start_cycle"] for e in events if "start_cycle" in e), default=0)
    result = []
    for event in events:
        match = re.fullmatch(r"AIC_(GMM[12])(?:_(ISSUE|BEGIN|END|FIX_DRAIN))?(?:_(\d+))?(?:_part\d+)?", event.get("name", ""))
        if not match:
            continue
        args = event.get("args", {})
        group = event.get("group_id", args.get("group_id"))
        track = re.fullmatch(r"group_(\d+)_AIC_(\d+)", str(event.get("tid", "")), re.IGNORECASE)
        if group is None and track:
            group = int(track[1])
        if group is None:
            raise ValueError("GMM event has no group_id; cannot infer core identity")
        if "start_cycle" in event:
            # 先在整数域减原点，避免绝对cycle转float损失纳秒精度。
            start, end = (event["start_cycle"] - origin) / 1000, (event["end_cycle"] - origin) / 1000
        else:
            if event.get("ph") != "X":
                continue
            start, end = event["ts"], event["ts"] + event["dur"]
        if end <= start:
            raise ValueError("GMM event must have positive duration")
        result.append(dict(rank=event.get("rank", args.get("rank", event.get("pid", 0))),
            group=group, sub=event.get("sub_id", args.get("sub_id", int(track[2]) if track else 0)),
            phase=match[1], semantics="issue" if match[2] == "ISSUE" else "native_boundary",
            boundary_kind=match[2] if match[2] in ("BEGIN", "END", "FIX_DRAIN") else None,
            sequence=match[3],
            start=start, end=end,
            task={k: event.get(k, args.get(k)) for k in ("expert_id", "row_start", "row_count", "wave_id")}))
    if not result:
        raise ValueError("No AIC GMM intervals")
    return result


def boundary(a, b):
    if a["phase"] != b["phase"]:
        return a["phase"] + "_to_" + b["phase"]
    if a["sequence"] is not None and a["sequence"] == b["sequence"]:
        return "same_task_fragment_resume"
    # 坐标相等不是 pipeline 生命周期相同；这里只分类已知的任务位置。
    ta, tb = a["task"], b["task"]
    if any(ta[k] is None or tb[k] is None for k in ("expert_id", "row_start", "row_count")):
        return "same_phase_unknown_task"
    if ta["expert_id"] != tb["expert_id"]:
        return "same_phase_expert_change"
    if ta != tb:
        return "same_phase_slice_or_wave_change"
    return "same_phase_same_task_coordinates"


def stats(values):
    ordered = sorted(values)
    return dict(count=len(values), total_us=sum(values), mean_us=statistics.mean(values),
                median_us=statistics.median(values), p95_us=ordered[(len(ordered) * 95 - 1) // 100],
                max_us=max(values))


def analyze(data):
    lanes = defaultdict(list)
    for event in normalize(data):
        lanes[(event["rank"], event["group"], event["sub"])].append(event)
    reports = []
    for lane, events in sorted(lanes.items(), key=lambda item: str(item[0])):
        lifecycle = defaultdict(list)
        for event in events:
            if event["boundary_kind"]:
                lifecycle[event["phase"] + ":" + event["boundary_kind"]].append(event["end"] - event["start"])
        events = [e for e in events if e["boundary_kind"] is None]
        if not events:
            raise ValueError(f"Only lifecycle markers without GMM tasks on lane {lane}")
        events.sort(key=lambda e: e["start"])
        durations, gaps, cadence = defaultdict(list), defaultdict(list), defaultdict(list)
        sequences = defaultdict(set)
        for event in events:
            durations[event["phase"] + ":" + event["semantics"]].append(event["end"] - event["start"])
            if event["sequence"] is not None:
                sequences[event["phase"]].add(event["sequence"])
        for a, b in zip(events, events[1:]):
            gap = b["start"] - a["end"]
            if gap < -1e-6:
                raise ValueError(f"Overlapping GMM intervals on lane {lane}; cannot add a serial ledger")
            key = a["phase"] + ":" + boundary(a, b)
            gaps[key].append(max(0, gap))
            cadence[key].append(b["start"] - a["start"])
        reports.append(dict(rank=lane[0], group_id=lane[1], sub_id=lane[2],
            first_us=events[0]["start"], last_us=events[-1]["end"],
            interval_stats={k: stats(v) for k, v in durations.items()},
            task_sequence_counts={k: len(v) for k, v in sequences.items()},
            native_lifecycle_stats={k: stats(v) for k, v in lifecycle.items()},
            gap_stats={k: stats(v) for k, v in gaps.items()},
            start_to_start_stats={k: stats(v) for k, v in cadence.items()}))
    return dict(semantics="Observed scalar intervals and progression, not MMAD busy time. Task coordinates are not Begin/End IDs. Do not sum parallel lanes or subtract unmatched task durations.", lanes=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{len(report['lanes'])} AIC lanes -> {args.output}")


if __name__ == "__main__":
    main()
