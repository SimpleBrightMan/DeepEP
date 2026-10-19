# Copyright (c) 2026, SimpleBright_Man

"""Communication-only MegaMoE simulator: real routing, Dispatch and Combine, no GMM."""

import argparse
from datetime import datetime, timezone
from importlib.resources import files
import json
import os
from pathlib import Path

from .benchmark import load_config
from .performance import collect_profile, enqueue_batch, summarize_rounds


def synthetic_input(rank, tokens, hidden):
    """Finite FP8 bit patterns and E8M0 scales identifying source/token/lane."""
    import numpy as np

    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1, 1)
    lanes = np.arange(hidden)
    data = ((tokens * 7 + rank * 13 + lanes) % 120).astype(np.uint8)
    # Repeated identity lanes distinguish every source/token, including tokens
    # separated by the hash period. All bit patterns remain finite FP8 values.
    data[:, lanes % 8 == 0] = rank
    data[:, lanes % 8 == 1] = tokens % 120
    data[:, lanes % 8 == 2] = (tokens // 120) % 120
    data[:, lanes % 8 == 3] = (tokens // 14400) % 120
    scales = (123 + (tokens + rank + np.arange(hidden // 32)) % 5).astype(np.uint8)
    return data, scales


def synthetic_payload(rank, rows, hidden):
    """Exactly representable BF16 values; generated once outside measured calls."""
    import numpy as np

    shifts = (np.arange(hidden) % 5) * 5
    return (((np.asarray(rows).reshape(-1, 1) >> shifts) & 31) + rank * 2).astype(np.float32)


def balanced_ids(rank, bs, experts, topk, world=16):
    import numpy as np

    routes = np.arange(rank * bs * topk, (rank + 1) * bs * topk, dtype=np.int64)
    return (
        ((routes % world) * (experts // world) + (routes // world) % (experts // world))
        .reshape(bs, topk)
        .astype(np.int32)
    )


def expected_layout(config, rank, bs):
    """Independent balanced-route oracle; never reads device metadata to build expectations."""
    import numpy as np
    from .wave_oracle import plan, route_queues

    model = config["model"]
    queues = route_queues(bs, model["num_topk"], model["num_experts"], rank)
    counts = np.array([sum(map(len, peers)) for peers in queues], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    mapping = np.full((int(offsets[-1]), 3), -1, dtype=np.int32)
    schedule = plan(bs, model["hidden"], model["intermediate_hidden"], model["num_experts"], model["num_topk"], rank)
    for item in schedule["transfers"]:
        if item.operation == "DISPATCH":
            mapping[int(offsets[item.expert]) + item.row] = item.source_rank, item.source_token, item.topk_slot
    if np.any(mapping < 0):
        raise AssertionError("Independent oracle left unassigned expert rows")
    return mapping, counts


def expected_output(config, rank, bs):
    """Reverse all destination maps, then weight/reduce the synthetic expert outputs."""
    import numpy as np
    from .reference import bf16

    model = config["model"]
    sums = np.zeros((bs, 5), dtype=np.int64)
    seen = np.zeros((bs, model["num_topk"]), dtype=np.int32)
    for destination in range(16):
        mapping, _ = expected_layout(config, destination, bs)
        rows = np.flatnonzero(mapping[:, 0] == rank)
        tokens, slots = mapping[rows, 1], mapping[rows, 2]
        values = ((rows[:, None] >> (np.arange(5) * 5)) & 31) + destination * 2
        np.add.at(sums, tokens, values)
        np.add.at(seen, (tokens, slots), 1)
    if not np.all(seen == 1):
        raise AssertionError("Independent Combine oracle must cover every token/top-k exactly once")
    return bf16(sums[:, np.arange(model["hidden"]) % 5].astype(np.float32) / model["num_topk"])


def validate_payload(config, rank, bs, snapshot, output, counts, expected):
    """Check every copied token/scale, route identity, expert count and reduced element."""
    import numpy as np

    mapping, expected_counts = expected_layout(config, rank, bs)
    rows, hidden = len(mapping), config["model"]["hidden"]
    data, scales, metadata = (snapshot[name] for name in ("data", "scales", "metadata"))
    if data.shape != (rows, hidden) or scales.shape != (rows, hidden // 32) or metadata.shape != (rows, 8):
        raise ValueError("Communication snapshot dimensions do not match expert-flat rows")
    data_equal = scales_equal = True
    for source in range(16):
        selected = np.flatnonzero(mapping[:, 0] == source)
        # Bound temporary expected arrays rather than retaining 16 source matrices.
        for begin in range(0, len(selected), 512):
            indices = selected[begin : begin + 512]
            expected_data, expected_scale = synthetic_input(source, mapping[indices, 1], hidden)
            data_equal &= bool(np.array_equal(data[indices], expected_data))
            scales_equal &= bool(np.array_equal(scales[indices], expected_scale))
    report = dict(
        rank=rank,
        rows=rows,
        dispatch_data_equal=data_equal,
        dispatch_scales_equal=scales_equal,
        metadata_equal=bool(np.array_equal(metadata[:, :3], mapping)),
        counts_equal=bool(np.array_equal(counts, expected_counts)),
        combine_output_equal=bool(np.array_equal(output, expected)),
        reference="independent_balanced_route_and_synthetic_payload_oracle",
    )
    report["passed"] = all(
        report[key]
        for key in (
            "dispatch_data_equal",
            "dispatch_scales_equal",
            "metadata_equal",
            "counts_equal",
            "combine_output_equal",
        )
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-timer-raw", action="store_true")
    parser.add_argument(
        "--dispatch-metadata-mode",
        choices=("dma", "scalar"),
        default="dma",
        help="Dispatch metadata via local GM DMA, or original scalar stores for comparison",
    )
    parser.add_argument(
        "--combine-buffer-count",
        type=int,
        choices=(2, 4, 6),
        default=2,
        help="Communication-only Prefill Combine pipeline slots; original production count is 2",
    )
    parser.add_argument(
        "--no-validation", action="store_true", help="Measure communication only; skip payload checks and task audit"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model = config["model"]
    if model != dict(hidden=3584, intermediate_hidden=1280, num_experts=384, num_topk=8, num_shared_experts=0):
        parser.error("Initial communication simulator supports H3584/I1280/E384/K8/S0")
    if any(bs not in (2048, 4096, 8192) for bs in config["bs_values"]):
        parser.error("Initial communication simulator supports BS2048/4096/8192")
    return args, config


def run(args, config, group):
    import numpy as np
    import torch
    from .api import SymmBuffer, communication_forward
    from .comm_report import rank_report, read_events, read_transfers, summarize, timer_rank_report, write_rank_detail
    from .timer.breakdown import compact_events, rank_breakdown, write_breakdown
    from .validation import collective_check

    rank, world = group.rank(), group.size()
    if world != 16 or int(os.environ["LOCAL_WORLD_SIZE"]) != 16:
        raise ValueError("Communication simulator requires one EP16 machine")
    collective_check(
        group,
        "communication-only configuration",
        None,
        (config, args.no_timer_raw, args.no_validation, args.dispatch_metadata_mode, args.combine_buffer_count),
    )
    rank_dir = args.output_dir / f"rank_{rank}"
    error = None
    try:
        rank_dir.mkdir(parents=True, exist_ok=False)
    except OSError as failure:
        error = str(failure)
    collective_check(group, "communication-only output directory", error)
    model = config["model"]
    manifest = dict(
        config=config,
        input="fp8",
        mode="perf",
        routing="balanced",
        workload="communication_only",
        validation="skipped" if args.no_validation else "strict",
        dispatch_metadata_mode=args.dispatch_metadata_mode,
        combine_buffer_count=args.combine_buffer_count,
        ep_size=world,
        rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        timer_ranks=list(range(16)),
        host_coordination="tcpstore",
        device_communication="shmem_mte",
        api="communication_forward",
        numerical_contract="synthetic_expert_bf16_uniform_route_weight_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        build=json.loads(files("deep_ep").joinpath("_build_info.json").read_text()),
        comm_core_split={
            name: os.environ.get(name)
            for name in ("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", "DEEPEP_MEGAMOE_COMBINE_INTRA_CORES")
        },
    )
    (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    buffer = SymmBuffer(group, num_max_tokens_per_rank=max(config["bs_values"]), ranks_per_node=16, **model)
    if not buffer.timer_enabled:
        raise RuntimeError("Rebuild with DEEPEP_BUILD_MEGAMOE=ON and DEEPEP_MEGAMOE_TIMER=ON")
    for bs in config["bs_values"]:
        case = rank_dir / f"bs_{bs}"
        case.mkdir()
        data, scales = synthetic_input(rank, np.arange(bs), model["hidden"])
        buffer.x[:bs].copy_(torch.from_numpy(data).npu().view(torch.float8_e4m3fn))
        buffer.x_sf[:bs].copy_(torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu))
        buffer.topk_idx[:bs].copy_(torch.from_numpy(balanced_ids(rank, bs, model["num_experts"], model["num_topk"])))
        buffer.topk_weights[:bs].fill_(1 / model["num_topk"])
        rows = bs * model["num_topk"]
        payload = torch.empty((rows, model["hidden"]), device="npu", dtype=torch.bfloat16)
        for begin in range(0, rows, 512):
            end = min(rows, begin + 512)
            payload[begin:end].copy_(torch.from_numpy(synthetic_payload(rank, np.arange(begin, end), model["hidden"])))
        buffer.prepare_communication(
            bs,
            combine_payload=payload,
            dispatch_metadata_mode=args.dispatch_metadata_mode,
            combine_buffer_count=args.combine_buffer_count,
        )
        torch.npu.synchronize()
        del payload, data, scales
        expected = None if args.no_validation else expected_output(config, rank, bs)
        result = torch.empty((bs, model["hidden"]), device="npu", dtype=torch.bfloat16)
        counts = None

        def call(timer=None):
            nonlocal counts
            _, counts = communication_forward(buffer, bs=bs, timer=timer, out=result)

        alignment = config["alignment"]
        prefix = torch.zeros((alignment["rows"], alignment["cols"]), dtype=torch.float32, device="npu")
        torch.exp(prefix, out=prefix)
        for _ in range(config["warmup"]):
            call()
        torch.npu.synchronize()
        group.barrier(f"communication warmup BS={bs}")
        samples = collect_profile(call, group, torch, config, prefix, case / "profiler", bs)
        (case / "kernel_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
        per_rank = group.all_gather([item["duration_us"] for item in samples], f"communication profiling BS={bs}")
        profile = summarize_rounds(per_rank, 16)
        profile.update(
            metric="communication_only_kernel_duration_us", workload="communication_only", alignment=alignment
        )
        (case / "profiling.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        uninstrumented = None if args.no_validation else result.float().cpu().numpy()

        def validate(label, expected, uninstrumented):
            snapshot = buffer.communication_snapshot(rows)
            arrays = {
                name: value.view(torch.uint8).cpu().numpy() if name != "metadata" else value.cpu().numpy()
                for name, value in snapshot.items()
            }
            actual = result.float().cpu().numpy()
            report = validate_payload(config, rank, bs, arrays, actual, counts.cpu().numpy(), expected)
            report["timer_off_output_equal"] = bool(np.array_equal(actual, uninstrumented))
            report["passed"] &= report["timer_off_output_equal"]
            reports = group.all_gather(report, f"communication payload {label} BS={bs}")
            (case / label).write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
            if not all(item["passed"] for item in reports):
                raise AssertionError(f"Communication payload verification failed: {case / label}")

        if not args.no_validation:
            validate("payload_accuracy.json", expected, uninstrumented)
        timer_dir = args.output_dir / f"bs_{bs}" / "timer"
        timer_dir.mkdir(parents=True, exist_ok=True)
        timer = buffer.allocate_timer()
        torch.npu.synchronize()
        group.barrier(f"communication timer warmup BS={bs}")
        for _ in range(config["warmup"]):
            call()
        torch.npu.synchronize()
        group.barrier(f"communication timer batch BS={bs}")
        enqueue_batch(call, torch, prefix, alignment, config["iters"], timer)
        torch.npu.synchronize()
        if not args.no_validation:
            validate("timer_accuracy.json", expected, uninstrumented)
        raw = timer.cpu().numpy().astype("<i8", copy=False).reshape(-1)
        events = read_events(raw, rank)
        breakdown = rank_breakdown(compact_events(raw, rank), rank)
        if not args.no_timer_raw:
            raw.tofile(timer_dir / f"rank_{rank}_npu_time.bin")
        del raw, timer
        breakdowns = group.all_gather(breakdown, f"communication timer breakdown BS={bs}")
        if rank == 0:
            write_breakdown(breakdowns, timer_dir, f"communication only BS={bs}")

        if args.no_validation:
            report = timer_rank_report(events, rank, model["hidden"])
        else:
            # Audit writes occur in a separate invocation; they never enter measured bandwidth.
            audit = buffer.allocate_timer(audit=True)
            torch.npu.synchronize()
            group.barrier(f"communication task audit BS={bs}")
            call(audit)
            torch.npu.synchronize()
            validate("communication_audit_accuracy.json", expected, uninstrumented)
            raw = audit.cpu().numpy().astype("<i8", copy=False).reshape(-1)
            transfers = read_transfers(raw, rank)
            if not args.no_timer_raw:
                raw.tofile(timer_dir / f"rank_{rank}_communication_audit.bin")
            del raw, audit
            report = rank_report(
                events,
                rank,
                bs,
                model["hidden"],
                model["num_topk"],
                int(os.environ.get("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", "0")),
                int(os.environ.get("DEEPEP_MEGAMOE_COMBINE_INTRA_CORES", "0")),
                experts=model["num_experts"],
                intermediate=model["intermediate_hidden"],
                ranks_per_node=16,
                transfers=transfers,
            )
        report["dispatch_metadata_mode"] = args.dispatch_metadata_mode
        report["combine_buffer_count"] = args.combine_buffer_count
        report = write_rank_detail(report, timer_dir)
        reports = group.all_gather(report, f"communication bandwidth report BS={bs}")
        if rank == 0:
            summary = summarize(reports, require_validation=not args.no_validation)
            summary["workload"] = "communication_only"
            summary["dispatch_metadata_mode"] = args.dispatch_metadata_mode
            summary["combine_buffer_count"] = args.combine_buffer_count
            (timer_dir / "communication.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            validation_status = (
                "validation skipped" if args.no_validation else "Dispatch/Combine payload and wave/core audit passed"
            )
            print(
                f"[COMM_ONLY] BS={bs} dispatch_metadata_mode={args.dispatch_metadata_mode} "
                f"combine_buffer_count={args.combine_buffer_count} "
                f"kernel rank-max-mean={profile['rank_max_mean_us']:.3f}us; "
                f"{validation_status}; no GMM executed",
                flush=True,
            )
        del prefix, expected, uninstrumented
        group.barrier(f"communication case complete BS={bs}")
    group.barrier("communication before SHMEM teardown")
    buffer.destroy()
    group.close()


def main():
    args, config = parse_args()
    import torch
    import torch_npu  # noqa: F401
    from .api import StoreGroup

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env()
    try:
        run(args, config, group)
    except BaseException as error:
        group.abort(f"communication benchmark: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
