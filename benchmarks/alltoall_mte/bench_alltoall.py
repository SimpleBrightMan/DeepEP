# Copyright (c) 2026, SimpleBrightMan

"""MTE All-to-All microbenchmark for an EP16 / 4-card-group UBX machine.

Each rank sends ``tokens`` tokens of ``token_bytes`` bytes to every selected
peer. ``local`` / ``intra`` / ``inter`` isolate the three bandwidth scopes
requested by the dispatch/combine investigation, and ``full`` runs the
complete 16-rank All-to-All as a cross-check.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from . import plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "16")))
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--device", type=int, default=0, help="NPU index inside ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--token-bytes", type=int, default=3584)
    parser.add_argument(
        "--tokens",
        default="2048",
        help="tokens per rank-to-rank pair; one value or a comma-separated list, e.g. 1024,2048,4096",
    )
    parser.add_argument("--group-size", type=int, default=4, help="physical UBX group size (4 cards)")
    parser.add_argument("--cores", type=int, default=64, help="AIV cores used by the copy kernel")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--capacity-mib", type=int, default=0, help="0 = derive from the largest payload")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-check", action="store_true", help="skip the one-shot payload check")
    parser.add_argument("--no-full", action="store_true", help="do not run the full All-to-All case")
    return parser.parse_args(argv)


def payload(rank: int, destination: int, tokens: int, token_bytes: int) -> np.ndarray:
    """Deterministic per-(source,destination,token) block so routing is checkable."""
    token = np.arange(tokens, dtype=np.uint32)[:, None]
    column = np.arange(token_bytes, dtype=np.uint32)[None, :]
    block = ((token * 7 + column * 3 + rank * 11 + destination * 5) & 0xFF).astype(np.uint8)
    block[:, 0] = rank & 0xFF
    block[:, 1] = destination & 0xFF
    block[:, 2] = token[:, 0] & 0xFF
    block[:, 3] = (token[:, 0] >> 8) & 0xFF
    return block


def make_send(rank: int, world: int, tokens: int, token_bytes: int, device: int):
    import torch

    host = np.empty((world, tokens, token_bytes), dtype=np.uint8)
    for destination in range(world):
        host[destination] = payload(rank, destination, tokens, token_bytes)
    return torch.from_numpy(host).to(f"npu:{device}")


def timed_case(native, runtime, send, tokens: int, token_bytes: int, group: int, mode: int, cores: int,
               warmup: int, iters: int):
    import torch

    stream = runtime.stream
    stream_address = stream.npu_stream
    for _ in range(warmup):
        native.alltoall(runtime.native, send, tokens, token_bytes, group, mode, cores, stream_address)
    stream.synchronize()
    samples = []
    for _ in range(iters):
        runtime.barrier()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record(stream)
        native.alltoall(runtime.native, send, tokens, token_bytes, group, mode, cores, stream_address)
        end.record(stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    runtime.barrier()
    return samples


def check_case(native, runtime, send, tokens: int, token_bytes: int, group: int, cores: int, rank: int,
               world: int, device: int):
    import torch

    out = torch.empty((world, tokens, token_bytes), dtype=torch.uint8, device=f"npu:{device}")
    native.alltoall(runtime.native, send, tokens, token_bytes, group, 3, cores, runtime.stream.npu_stream)
    # The sender drains only its own MTE queue. Every peer must finish its
    # writes before any rank reads the shared inbox, otherwise the drain can
    # observe a partially written source region.
    runtime.barrier()
    native.alltoall_drain(runtime.native, out, tokens, token_bytes, cores, runtime.stream.npu_stream)
    runtime.stream.synchronize()
    runtime.barrier()
    host = out.cpu().numpy()
    del out
    for source in range(world):
        expected = payload(source, rank, tokens, token_bytes)
        if not np.array_equal(host[source], expected):
            mismatch = int(np.count_nonzero(host[source] != expected))
            raise AssertionError(
                f"payload mismatch source={source} destination={rank} tokens={tokens}: {mismatch} bytes"
            )
    return True


def run(args):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    from deep_ep._native import require_native
    from deep_ep._runtime import Runtime

    native = require_native("AllToAll")
    tokens_list = plan.parse_token_counts(args.tokens)
    plan.validate_shape(args.world_size, args.group_size, args.token_bytes)
    if not 1 <= args.world_size <= 128:
        raise ValueError("world size must be in [1, 128]")
    if not 1 <= args.cores <= 64:
        raise ValueError("cores must be in [1, 64]")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be >= 0 and iters >= 1")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank is outside the world")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = max(plan.workspace_bytes(args.world_size, tokens, args.token_bytes) for tokens in tokens_list)
    capacity = args.capacity_mib << 20
    if capacity == 0:
        capacity = max(2 << 20, (required + (2 << 20) - 1) // (2 << 20) * (2 << 20))
    if capacity < required or capacity % (2 << 20):
        raise ValueError("capacity must cover the payload and be a 2 MiB multiple")

    runtime = None
    started = time.time()
    try:
        runtime = Runtime(None, capacity, timeout=args.timeout, rank=args.rank, world_size=args.world_size)
        runtime.barrier()
        results = []
        for tokens in tokens_list:
            send = make_send(args.rank, args.world_size, tokens, args.token_bytes, args.device)
            checked = False
            if not args.no_check:
                checked = check_case(native, runtime, send, tokens, args.token_bytes, args.group_size, args.cores,
                                     args.rank, args.world_size, args.device)
            modes = list(plan.MODES) if not args.no_full else list(plan.MODES[:-1])
            for mode in modes:
                peers = plan.peers_for(args.rank, args.world_size, args.group_size, mode)
                if not peers:
                    continue
                samples = timed_case(native, runtime, send, tokens, args.token_bytes, args.group_size,
                                     plan.MODES.index(mode), args.cores, args.warmup, args.iters)
                results.append(
                    {
                        "tokens": tokens,
                        "mode": mode,
                        "peers": len(peers),
                        "bytes": plan.payload_bytes(tokens, args.token_bytes, len(peers)),
                        "times_ms": samples,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                    }
                )
                print(
                    f"[rank {args.rank}] tokens={tokens} mode={mode} peers={len(peers)} "
                    f"median={statistics.median(samples):.3f} ms",
                    flush=True,
                )
            del send
        report = {
            "schema": 1,
            "rank": args.rank,
            "world_size": args.world_size,
            "local_rank": args.local_rank,
            "device": args.device,
            "group_size": args.group_size,
            "token_bytes": args.token_bytes,
            "cores": args.cores,
            "warmup": args.warmup,
            "iters": args.iters,
            "checked": not args.no_check,
            "elapsed_s": time.time() - started,
            "cases": results,
        }
        destination = args.output_dir / f"rank_{args.rank:02d}.json"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[rank {args.rank}] wrote {destination}", flush=True)
    finally:
        if runtime is not None:
            runtime.close()


def main(argv=None):
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
