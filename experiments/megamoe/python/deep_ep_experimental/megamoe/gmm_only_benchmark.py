"""Single-NPU CATLASS replay of BS2048/4096/8192 EP16 GMM work, without external flags.

The logical rank controls the original routing/segmented row layout, NOT an
actual communication rank. No SHMEM, HCCL or torch.distributed initialization.
"""

import argparse
import csv
import importlib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import statistics
import sys
import uuid

MODES = {"both": 0, "gmm1": 1, "gmm2": 2}
KERNEL = re.compile(
    r"(?:^|::|\s)mega_moe_gmm_only_kernel(?=\s*(?:$|\())"
    r"|^_Z24mega_moe_gmm_only_kernelN11MegaMoeUrma9ArgumentsEj$"
)


def load_gmm_extension():
    """Load the standalone kernel without requiring a communication runtime."""
    name = __package__ + "._C"
    try:
        extension = importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f"Cannot load {name} with Python {sys.executable}; runner={__file__}. "
            "Install the megamoe-gmm-only-bs2048 wheel into this Python environment "
            "and configure its matching Torch, torch_npu, CANN and SHMEM libraries. "
            f"Native loader error: {error}"
        ) from error
    if getattr(extension, "gmm_only_protocol", None) != 2 or not hasattr(extension, "GmmOnlyPlan"):
        raise RuntimeError(
            f"GMM-only Python/native versions differ: extension={getattr(extension, '__file__', name)}; "
            f"Python={sys.executable}. Rebuild/install the megamoe-gmm-only-bs2048 branch wheel."
        )
    return extension


def route_counts(bs=2048):
    """Call the repository's actual balanced generator for all 16 sources."""
    import numpy as np
    from .benchmark import inputs, load_config

    if bs not in (2048, 4096, 8192):
        raise ValueError("BS must be 2048, 4096 or 8192")
    config = load_config("urma", str(bs))
    counts = []
    for source in range(16):
        _, ids, _ = inputs(config, source, bs, "balanced", world=16)
        counts.append(np.bincount(ids.reshape(-1), minlength=384))
    return np.asarray(counts, dtype=np.int64)


def workload(counts, rank, bs=2048):
    if type(rank) is not int or not 0 <= rank < 16:
        raise ValueError("logical rank must be in [0,16)")
    if bs not in (2048, 4096, 8192):
        raise ValueError("BS must be 2048, 4096 or 8192")
    if counts.shape != (16, 384) or (counts < 0).any() or (counts.sum(axis=1) != bs * 8).any():
        raise ValueError(f"invalid BS{bs} EP16 routing counts")
    expert_rows = counts[:, rank * 24 : (rank + 1) * 24].sum(axis=0)
    rows = int(expert_rows.sum())
    groups = int(((expert_rows + 255) // 256).sum())
    return dict(
        bs=bs,
        logical_rank=rank,
        ep=16,
        receive_rows=rows,
        m_groups=groups,
        expert_rows=expert_rows.tolist(),
        gmm1_tiles=groups * 5,
        gmm2_tiles=groups * 14,
        gmm1_flops=4 * rows * 3584 * 1280,
        gmm2_flops=2 * rows * 3584 * 1280,
    )


def summarize(durations, flops, peak_tflops):
    if not durations or any(not math.isfinite(x) or x <= 0 for x in durations):
        raise ValueError("profile durations must be finite and positive")
    if not math.isfinite(peak_tflops) or peak_tflops <= 0 or flops <= 0:
        raise ValueError("positive FLOPs and peak TFLOPS required")
    mean = statistics.mean(durations)
    return dict(
        mean_us=mean,
        min_us=min(durations),
        max_us=max(durations),
        rounds=len(durations),
        flops=flops,
        effective_tflops=flops / mean / 1e6,
        mfu_percent=100 * flops / mean / 1e6 / peak_tflops,
        peak_tflops=peak_tflops,
        formula="MFU = effective FLOPs / (mean kernel us * 1e-6) / (peak TFLOPS * 1e12)",
    )


def read_samples(directory, expected):
    samples = []
    for path in sorted(Path(directory).rglob("kernel_details*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for raw in csv.DictReader(stream):
                row = {k.strip(): (v or "").strip() for k, v in raw.items() if k is not None}
                name = next((row[k] for k in ("Name", "Kernel Name", "Op Name") if KERNEL.search(row.get(k, ""))), None)
                if name is None:
                    continue
                try:
                    start, duration = Decimal(row["Start Time(us)"]), float(row["Duration(us)"])
                except (KeyError, InvalidOperation, ValueError) as error:
                    raise ValueError(f"Invalid profiler row in {path}: {row}") from error
                if not start.is_finite() or start < 0 or not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid GMM-only timing in {path}")
                samples.append(
                    dict(
                        start_us=str(start),
                        duration_us=duration,
                        name=name,
                        file=str(path),
                        device=row.get("Device_id") or row.get("Device ID"),
                        stream=row.get("Stream ID") or row.get("Stream Id"),
                    )
                )
    samples.sort(key=lambda item: Decimal(item["start_us"]))
    keys = [(s["device"], s["stream"], Decimal(s["start_us"])) for s in samples]
    if len(set(keys)) != len(keys) or len({(s["device"], s["stream"]) for s in samples}) > 1:
        raise ValueError("Duplicate or multi-device/stream profile rows")
    if len(samples) != expected:
        raise ValueError(f"Expected {expected} GMM-only device tasks, found {len(samples)} under {directory}")
    return samples


def profile(plan, mode, warmup, iters, directory):
    import torch
    import torch_npu

    directory.mkdir(parents=True, exist_ok=False)
    profiler = torch_npu.profiler
    torch.npu.synchronize()  # All generated data/weights and descriptors are ready.
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            data_simplification=False,
        ),
    ) as capture:
        for _ in range(warmup + iters):
            plan.forward(MODES[mode])
        torch.npu.synchronize()
        capture.step()
    samples = read_samples(directory, warmup + iters)
    return samples, [row["duration_us"] for row in samples[warmup:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bs", type=int, choices=(2048, 4096, 8192), default=2048)
    parser.add_argument("--logical-rank", type=int, default=0)
    parser.add_argument("--cv-rows", type=int, default=256)
    parser.add_argument("--gmm1-ahead-mgroups", type=int, default=12)
    parser.add_argument("--modes", default="both,gmm1,gmm2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--peak-tflops", type=float, default=765.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true", help="CPU routing/FLOP report; no NPU access")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if len(modes) != len(set(modes)) or any(m not in MODES for m in modes):
        parser.error("--modes must be distinct entries from both,gmm1,gmm2")
    if (
        args.device < 0
        or args.warmup < 0
        or args.iters < 1
        or not math.isfinite(args.peak_tflops)
        or args.peak_tflops <= 0
    ):
        parser.error("invalid device, iteration count or peak TFLOPS")
    if (
        not 0 <= args.logical_rank < 16
        or not 128 <= args.cv_rows <= 4096
        or args.cv_rows % 128
        or not 0 <= args.gmm1_ahead_mgroups <= 65536
    ):
        parser.error("invalid logical rank, CV rows or lookahead")
    counts = route_counts(args.bs)
    expected = workload(counts, args.logical_rank, args.bs)
    if args.plan_only:
        print(json.dumps(expected, indent=2))
        return

    import torch
    import torch_npu  # noqa: F401
    from .api import _check_fp4_weight_format, transform_weights_for_mega_moe
    from .reference import make_weights

    torch.npu.set_device(args.device)
    ext = load_gmm_extension()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = args.output_dir or Path("benchmark-runs/gmm_only") / stamp
    output.mkdir(parents=True, exist_ok=False)
    raw = make_weights(24, 3584, 1280, 1234 + args.logical_rank)

    def pair(value):
        data, scale = value
        return torch.from_numpy(data).npu(), torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu)

    first, second = transform_weights_for_mega_moe(*map(pair, raw))
    w1, s1 = first.select(False)  # Same planar prefill layout as UrmaBuffer.
    w2, s2 = second
    _check_fp4_weight_format(w1, "w1")
    _check_fp4_weight_format(w2, "w2")
    plan = ext.GmmOnlyPlan(
        torch.from_numpy(counts), w1, w2, s1, s2, args.logical_rank, args.cv_rows, args.gmm1_ahead_mgroups, args.bs
    )
    native = plan.describe()
    for key in ("bs", "receive_rows", "m_groups", "gmm1_tiles", "gmm2_tiles", "gmm1_flops", "gmm2_flops"):
        if native[key] != expected[key]:
            raise RuntimeError(f"Native/original routing work mismatch: {key}: {native[key]} != {expected[key]}")
    del raw
    report = dict(
        workload=expected,
        native=native,
        device=args.device,
        dimensions=dict(hidden=3584, intermediate=1280, experts=384, topk=8),
        base="origin/megamoe-urma-tailCover 60642fb",
        timing="profiler device kernel duration; no Event/internal timer",
        scope="one NPU replay of one logical EP16 rank; no communication or accuracy validation",
        inputs="independent, prepared finite FP8 inputs for both GMMs; GMM2 does not consume GMM1 output",
        retained="CATLASS/AIV0 FP4 prologue, M256/N256, row segments, expert schedule, internal pipeline dependencies",
        removed="external GM flags, per-tile GMM1 publication drain, activation, dispatch/combine, pack/reduce",
        warmup=args.warmup,
        iters=args.iters,
        results={},
    )
    try:
        for mode in modes:
            samples, durations = profile(plan, mode, args.warmup, args.iters, output / mode / "profile")
            flops = sum(native[f"{stage}_flops"] for stage in ("gmm1", "gmm2") if mode in ("both", stage))
            result = summarize(durations, flops, args.peak_tflops)
            result.update(samples=samples, formal_durations_us=durations)
            report["results"][mode] = result
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(
                f"[GMM-ONLY] BS={args.bs} logical-rank={args.logical_rank} mode={mode} rows={native['receive_rows']} "
                f"mean={result['mean_us']:.3f}us min/max={result['min_us']:.3f}/{result['max_us']:.3f}us "
                f"TFLOPS={result['effective_tflops']:.3f} MFU={result['mfu_percent']:.2f}% "
                f"peak={args.peak_tflops:g}TFLOPS",
                flush=True,
            )
    finally:
        torch.npu.synchronize()
    print(f"[COMPLETED] {output / 'summary.json'} (performance only; no accuracy verdict)")


if __name__ == "__main__":
    main()
