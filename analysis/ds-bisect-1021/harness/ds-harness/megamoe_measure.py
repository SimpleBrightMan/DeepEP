"""Match Yancy/DeepEP a859c17's aligned kernel-Duration measurement, not its backend.

Copyright (c) 2026, Lu Lu. Adapted from experiments/megamoe/performance.py.
"""
import csv
import math
from pathlib import Path
import statistics

DEFAULT_ALIGNMENT = dict(rows=100000, cols=30000, repeats=20)


def enqueue_batch(call, torch, prefix, iterations, timer=None, *, timer_every_iteration=False):
    for _ in range(DEFAULT_ALIGNMENT["repeats"]):
        torch.exp(prefix, out=prefix)
    for iteration in range(iterations):
        call(timer=timer if timer_every_iteration or iteration + 1 == iterations else None)


def read_samples(directory, iterations):
    samples = []
    for path in sorted(Path(directory).rglob("kernel_details*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                row = {key.strip(): value for key, value in row.items() if key is not None}
                name = " ".join(row.get(key, "") for key in ("Name", "Type", "Kernel Name"))
                if "mega_moe_impl" not in name:
                    continue
                start, duration = float(row["Start Time(us)"]), float(row["Duration(us)"])
                if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid MegaMoE sample: {path}: {row}")
                samples.append(dict(start_us=start, duration_us=duration, name=name,
                                    stream=row.get("Stream ID"), device=row.get("Device_id")))
    samples.sort(key=lambda row: row["start_us"])
    if len(samples) != iterations or len({(s["device"], s["stream"], s["start_us"]) for s in samples}) != iterations:
        raise ValueError(f"{directory}: expected {iterations} unique kernels; found {len(samples)}; inspect export")
    return samples


def summarize(per_rank, *, timer_pass=False):
    if len(per_rank) != 8 or not per_rank[0] or any(len(r) != len(per_rank[0]) for r in per_rank):
        raise ValueError("Require eight equal-length nonempty rank sample lists")
    if any(not math.isfinite(t) or t <= 0 for row in per_rank for t in row):
        raise ValueError("Nonpositive/nonfinite timing")
    recorded = len(per_rank[0])
    rows = [row[recorded // 2:] for row in per_rank]
    return dict(metric="megamoe_kernel_duration_us", backend="deepseek", timer_pass=timer_pass,
                aggregation="mean_of_per_round_min_rank", iteration_selection="tail_half_ceil",
                recorded_rounds=recorded, rounds=len(rows[0]), first_selected_iteration=recorded // 2,
                mean_us=statistics.mean(min(r) for r in zip(*rows)),
                rank_max_mean_us=statistics.mean(max(r) for r in zip(*rows)),
                per_rank_rounds_us=rows, alignment=DEFAULT_ALIGNMENT)


def compare_timer_overhead(baseline, instrumented, budget_pct):
    """同一 profiler/对齐/后半轮口径；rank-max 是保守的单次 kernel 跨 rank 指标。"""
    if not math.isfinite(budget_pct) or budget_pct < 0:
        raise ValueError("Timer overhead budget must be finite and nonnegative")
    if baseline["timer_pass"] or not instrumented["timer_pass"]:
        raise ValueError("Require timer-off baseline and timer-on measurement")
    for field in ("metric", "aggregation", "iteration_selection", "recorded_rounds", "alignment"):
        if baseline[field] != instrumented[field]:
            raise ValueError(f"Timer measurement mismatch: {field}")
    result = dict(metric=baseline["metric"], budget_pct=budget_pct,
                  acceptance_metric="rank_max_mean_us", scope="kernel only; excludes prelaunch memset and host work")
    for metric in ("mean_us", "rank_max_mean_us"):
        off, on = baseline[metric], instrumented[metric]
        if not all(math.isfinite(t) and t > 0 for t in (off, on)):
            raise ValueError(f"Invalid timer comparison: {metric} off={off} on={on}")
        result[metric] = dict(timer_off_us=off, timer_on_us=on, overhead_us=on - off,
                              overhead_pct=(on / off - 1) * 100)
    result["within_budget"] = result["rank_max_mean_us"]["overhead_pct"] <= budget_pct
    return result


def collect(call, torch, dist, prefix, iterations, directory, *, timer=None):
    import torch_npu
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(directory)),
        profile_memory=True, with_stack=False, with_flops=False, with_modules=False,
        experimental_config=p._ExperimentalConfig(export_type=p.ExportType.Text,
            aic_metrics=p.AiCMetrics.PipeUtilization, profiler_level=p.ProfilerLevel.Level1,
            l2_cache=False, data_simplification=False),
    ) as prof:
        torch.npu.synchronize()
        dist.barrier()
        # 开销测量必须每轮带 timer，不能拿“仅最后一轮插桩”的批次平均值稀释开销。
        # 每次 kernel 的局部计数从零开始，buffer 最终保留最后一轮，容量之外旧槽由 count 排除。
        enqueue_batch(call, torch, prefix, iterations, timer, timer_every_iteration=timer is not None)
        torch.npu.synchronize()
        prof.step()
    return read_samples(directory, iterations)
