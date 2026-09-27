"""Installed-wheel expanded dispatch/combine functional precision test.

NPU: python tests/elastic/test_combine.py --ep-size 2 --nproc-per-node 2 --output /tmp/combine-check
"""

import argparse
import json
from pathlib import Path
import runpy

_dispatch = runpy.run_path(str(Path(__file__).with_name("test_dispatch.py")))
_ROUTES = ("normal", "duplicates", "invalid", "empty", "local_duplicates", "local_experts")


def payload(source, tokens, generation):
    import torch

    return ((torch.arange(tokens)[:, None] + torch.arange(7168)[None, :] + source + generation) % 7 - 3).to(
        torch.bfloat16
    )


def expected_output(routes, experts, rank, generation):
    import torch

    base = payload(rank, len(routes[rank]), generation).float()
    out = torch.zeros_like(base)
    for token, selected in enumerate(routes[rank]):
        for slot, expert in enumerate(selected):
            if 0 <= expert < experts:
                out[token] += base[token] + (expert % 3 - 1) + (slot % 3 - 1)
    return out.to(torch.bfloat16)


def worker(local_rank, args):
    import torch
    import torch_npu  # noqa: F401

    deep_ep = _dispatch["_import_installed_deep_ep"]()
    from deep_ep.buffers._elastic_layout import elastic_size_hint

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    if "elastic-combine-expanded-v1" not in info.get("capabilities", []):
        raise RuntimeError("Installed wheel does not contain expanded Combine; rebuild/install first")
    torch.npu.set_device(local_rank)
    rank, world = _dispatch["_worker_identity"](local_rank, args)
    log = _dispatch["_progress_logger"]("combine", rank)
    log("worker.begin")
    modes = _ROUTES if args.route == "all" else (args.route,)
    capacity = 0
    for mode in modes:
        log(f"route={mode} capacity_golden.begin (CPU, before buffer initialization)")
        routes = _dispatch["make_routes"](world, args.tokens, args.topk, args.experts, 0, mode)
        plan = _dispatch["build_notify_golden"](routes, args.experts, 256)
        capacity = max(
            capacity,
            elastic_size_hint(
                world,
                args.tokens,
                args.topk,
                args.experts,
                max(1, max(plan["rank_rows"])),
                max(1, max(plan["gather_rows"])),
            ),
        )
        log(f"route={mode} capacity_golden.done session_bytes={capacity}")
    del plan
    log(f"buffer.init.begin bytes={capacity}")
    buffer = deep_ep.ElasticBuffer(
        rank=rank,
        world_size=world,
        num_bytes=capacity,
        num_max_tokens_per_rank=args.tokens,
        hidden=7168,
        num_topk=args.topk,
        explicitly_destroy=True,
    )
    log("buffer.init.done")
    checks, retained = 0, None
    try:
        for mode in modes:
            log(f"route={mode} routes_and_expert_rows.begin (CPU)")
            routes = _dispatch["make_routes"](world, args.tokens, args.topk, args.experts, 0, mode)
            entries = _dispatch["expert_rows"](routes, args.experts, rank)
            log(f"route={mode} routes_and_expert_rows.done rows={len(entries)}")
            handle = None
            for generation in range(args.repeat):
                context = f"route={mode} iter={generation + 1}/{args.repeat}"
                log(f"{context} inputs.begin")
                indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu") if handle is None else None
                x = payload(rank, args.tokens, generation).npu()
                log(f"{context} dispatch.enter mode={'fresh' if handle is None else 'cached'}")
                recv, _, _, handle, event = buffer.dispatch(
                    x,
                    topk_idx=indices,
                    num_experts=args.experts if handle is None else None,
                    handle=handle,
                    do_expand=True,
                    async_with_compute_stream=True,
                )
                log(f"{context} dispatch.return (async; device completion not yet confirmed)")
                event.current_stream_wait()
                log(f"{context} dispatch_golden.begin (CPU)")
                sources = torch.tensor([s + t for s, t, _ in entries], dtype=torch.int32)[:, None]
                expected_recv = ((sources + torch.arange(7168)[None, :] + generation) % 7 - 3).to(torch.bfloat16)
                log(f"{context} dispatch_golden.done")
                _dispatch["assert_bytes"](recv, expected_recv, f"{context} dispatch input to experts", progress=log)
                # Producer stream is deliberately distinct; Combine must honor its event.
                log(f"{context} expert_compute.enqueue.begin")
                producer = torch.npu.Stream()
                with torch.npu.stream(producer):
                    event.current_stream_wait()
                    bias = torch.tensor(
                        [routes[s][t][k] % 3 - 1 + k % 3 - 1 for s, t, k in entries], dtype=torch.float32, device="npu"
                    )[:, None]
                    expert = (recv.float() + bias).to(torch.bfloat16)
                    before = buffer.capture()
                log(f"{context} expert_compute.enqueued; combine.enter")
                output, values, completed = buffer.combine(
                    expert, handle, previous_event=before, async_with_compute_stream=True
                )
                log(f"{context} combine.return (async; device completion not yet confirmed)")
                completed.current_stream_wait()
                assert values is None
                log(f"{context} combine_golden.begin (CPU)")
                golden = expected_output(routes, args.experts, rank, generation)
                log(f"{context} combine_golden.done")
                _dispatch["assert_bytes"](output, golden, f"{context} combine", progress=log)
                # The same handle and workspace must support consecutive combines.
                log(f"{context} repeated_combine.enter")
                again, _, completed = buffer.combine(expert, handle)
                log(f"{context} repeated_combine.return (device completion not yet confirmed)")
                completed.current_stream_wait()
                _dispatch["assert_bytes"](again, golden, f"{context} repeated combine", progress=log)
                log(f"{context} retained_output_and_negative_checks.begin")
                if retained is not None:
                    _dispatch["assert_bytes"](*retained, "retained output")
                retained = (output, golden)
                corrupt = golden.clone()
                corrupt[0, 0] += 1
                try:
                    _dispatch["assert_bytes"](output, corrupt, "negative control")
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Comparator accepted corrupted golden")
                checks += 1
                log(f"{context} iteration.passed")
            print(f"[combine][rank={rank}] route={mode} precision passed", flush=True)
    except BaseException:
        _dispatch["_log_failure"](log)
        raise
    finally:
        log("buffer.destroy.begin (may wait for device work/peers)")
        buffer.destroy()
        log("buffer.destroy.done")
    if retained is not None:
        _dispatch["assert_bytes"](*retained, "output after destroy")
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                checks=checks,
                passed=True,
                tokens=args.tokens,
                topk=args.topk,
                experts=args.experts,
                route=args.route,
                source_sha256=info["source_sha256"],
            ),
            indent=2,
        )
        + "\n"
    )


def test_local_routes_require_reduction_without_remote_gather():
    for mode in ("local_duplicates", "local_experts"):
        routes = _dispatch["make_routes"](16, 3, 4, 32, 0, mode)
        for source, tokens in enumerate(routes):
            for token, row in enumerate(tokens):
                assert {expert // 2 for expert in row} == {(source + token) % 16}
                assert len(set(row)) == (1 if mode == "local_duplicates" else 2)
        golden = _dispatch["build_notify_golden"](routes, 32, 256)
        assert sum(golden["rank_rows"]) == 16 * 3 * 4
        assert not any(golden["gather_rows"])
        for rank, plan in enumerate(golden["plans"]):
            assert not any(plan["chunk_masks"])
            for records in plan["forward_list"]:
                for _, _, primary, peer, row, gather in records:
                    assert peer == rank and row != primary and gather == -1


def test_combine_reduction_predicates(tmp_path):
    """Execute the kernel's actual predicates on Host; does not test NPU synchronization."""
    import importlib.util
    import shutil
    import subprocess
    import sys

    import pytest

    root = Path(__file__).resolve().parents[2]
    source = root / "csrc/kernels/combine/combine.cpp"
    if not source.is_file():
        pytest.skip("Kernel source test runs before packaging")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("Host predicate test requires a C++17 compiler")
    code = source.read_text(encoding="utf-8")
    begin = code.index("__aicore__ inline bool ChunkNeedsGatewayReduce(")
    end = code.index("__aicore__ inline void AddBf16(", begin)
    harness = tmp_path / "combine_predicates.cpp"
    harness.write_text(
        r"""
#include <cassert>
#include <vector>
#include "kernels/combine/tiling.hpp"
#define __aicore__
using CombineTiling = ascend_deepep::PutCombineTiling;
template<class T> struct GlobalTensor {
    std::vector<T> values;
    unsigned reads = 0;
    T GetValue(uint64_t index) { ++reads; return values.at(index); }
};
"""
        + code[begin:end]
        + r"""
int main() {
    auto t = ascend_deepep::MakeCombineTiling(3, 8, 257, 2, 514, 514);
    GlobalTensor<int32_t> forward{std::vector<int32_t>(8 * 514 * 6, -1)};
    GlobalTensor<int32_t> ranges{std::vector<int32_t>(8 * 2 * 2, 0)};
    GlobalTensor<int32_t> masks{std::vector<int32_t>(8 * 2, 0)};
    const uint32_t source = 2;
    const uint64_t base = uint64_t(source) * t.forward_row_capacity;
    const uint64_t chunk = uint64_t(source) * t.chunks_per_group;
    auto needed = [&](uint32_t c) { return ChunkNeedsGatewayReduce(forward, ranges, masks, t, source, c); };
    auto source_needed = [&]() { return SourceNeedsGatewayReduce(forward, ranges, masks, t, source); };
    // Empty chunks must not read any forward record or wait for reduction.
    assert(!needed(0) && !source_needed() && forward.reads == 0);
    ranges.values[chunk * 2] = 2;
    ranges.values[chunk * 2 + 1] = 5;
    assert(!needed(0)); // Three singleton records.
    // A contributor outside this chunk must not affect the decision.
    forward.values[(base + 5) * 6 + 3] = t.rank_id;
    assert(!needed(0));
    // Local duplicate after a singleton: gather mask stays zero but we must wait.
    forward.values[(base + 3) * 6 + 3] = t.rank_id;
    assert(needed(0) && source_needed() && masks.values[chunk] == 0);
    forward.values[(base + 3) * 6 + 3] = -1;
    // Only the final, partial chunk needs reduction; source-wide check must find it.
    ranges.values[(chunk + 1) * 2] = 5;
    ranges.values[(chunk + 1) * 2 + 1] = 6;
    assert(!needed(0) && needed(1) && source_needed());
    assert(!SourceNeedsGatewayReduce(forward, ranges, masks, t, 1));
    // Remote contribution fast path must avoid scanning ranges/forward metadata.
    masks.values[chunk] = 1 << 4;
    forward.reads = ranges.reads = 0;
    assert(needed(0) && source_needed());
    assert(forward.reads == 0 && ranges.reads == 0);
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("combine_predicates.exe" if sys.platform == "win32" else "combine_predicates")
    result = subprocess.run(
        [*command, "-std=c++17", "-O2", "-I", str(root / "csrc"), str(harness), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _dispatch["_add_launch_args"](parser)
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--route", choices=("all", *_ROUTES), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ep_size > 128 or not 1 <= args.tokens <= 10240 or not 1 <= args.topk <= 16 or args.repeat < 1:
        parser.error("EP<=128, tokens1..10240, topk1..16, repeat>=1 required")
    if args.experts < args.ep_size or args.experts > 1024 or args.experts % args.ep_size:
        parser.error("experts must be divisible by EP and <=1024")
    args.output.mkdir(parents=True, exist_ok=True)
    _dispatch["_launch"](args, worker)


if __name__ == "__main__":
    main()
