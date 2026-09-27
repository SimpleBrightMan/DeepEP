"""Forward-list contract checks and installed-wheel, bitwise NPU dispatch validation.

CPU: python -m pytest tests/elastic/test_dispatch.py -q
NPU: python tests/elastic/test_dispatch.py --ep-size 2 --nproc-per-node 2 \
     --output /tmp/dispatch-check-001
"""

import argparse
import json
import os
from pathlib import Path
import importlib
from importlib.machinery import PathFinder
import sys
import sysconfig
import time
import traceback


def _progress_logger(operation, rank):
    started = previous = time.perf_counter()

    def log(message):
        nonlocal previous
        now = time.perf_counter()
        print(
            f"[{operation}][rank={rank}] {time.strftime('%H:%M:%S')} "
            f"elapsed={now - started:.3f}s delta={now - previous:.3f}s {message}",
            flush=True,
        )
        previous = now

    return log


def _log_failure(log):
    # Cleanup may wait for device work or peers; expose the original error first.
    log("FAILED: traceback before buffer.destroy")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()


def group_peer(rank, world, group, index):
    if world == 128:
        toggle, subround = divmod(group, 4)
        row, side = rank % 64 // 32, rank % 8 // 4
        peer_side = side ^ (subround % 2)
        peer_row = (row + subround // 2) % 2
        frame = rank // 64 ^ side ^ toggle
        return frame * 64 + (peer_row * 4 + index // 4) * 8 + peer_side * 4 + index % 4
    width, half = min(world, 16), world // 2
    half_width, groups = width // 2, world // width
    own = rank % half // half_width
    target = (own + group) % groups
    return index // half_width * half + target * half_width + index % half_width


def group_steps(rank, world):
    steps = [0] * world
    for group in range(world // min(world, 16)):
        for index in range(min(world, 16)):
            steps[group_peer(rank, world, group, index)] = group
    return steps


def build_notify_golden(routes, experts, chunk_tokens):
    world = len(routes)
    if world not in (2, 4, 8, 16, 32, 64, 128, 256) or experts <= 0 or experts % world:
        raise ValueError("unsupported Notify shape")
    tokens = len(routes[0])
    topk = len(routes[0][0]) if tokens else 0
    if (
        chunk_tokens <= 0
        or not 1 <= topk <= 16
        or any(len(rows) != tokens or any(len(row) != topk for row in rows) for rows in routes)
    ):
        raise ValueError("routes must have a common [tokens, topk] shape")
    local_experts = experts // world
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens

    counts = [0] * experts
    for rows in routes:
        for row in rows:
            for expert in row:
                if 0 <= expert < experts:
                    counts[expert] += 1
    starts = [0] * experts
    rank_rows = [0] * world
    for expert, count in enumerate(counts):
        peer = expert // local_experts
        starts[expert] = rank_rows[peer]
        rank_rows[peer] += count
    stride = max(1, max(rank_rows, default=0))
    invalid = -(1 << 31)
    plans = []
    for source in range(world):
        plans.append(
            {
                "server_mask": [0] * tokens,
                "dst": [invalid] * (tokens * topk),
                "forward_list": [[] for _ in range(world)],
                "backward_list": [[] for _ in range(min(world, 8))],
                "chunk_ranges": [[(0, 0) for _ in range(chunks)] for _ in range(world)],
                "chunk_masks": [0] * (world * chunks),
            }
        )

    expert_cursors = [0] * experts
    gathers = [0] * world
    for source, rows in enumerate(routes):
        for token, route in enumerate(rows):
            by_server = {}
            for slot, expert in enumerate(route):
                if not 0 <= expert < experts:
                    continue
                peer = expert // local_experts
                output_row = starts[expert] + expert_cursors[expert]
                expert_cursors[expert] += 1
                server = peer // 8
                plans[source]["server_mask"][token] |= 1 << server
                by_server.setdefault(server, []).append((slot, peer, output_row))
            for records in by_server.values():
                primary_slot, gateway, primary_row = records[0]
                for slot, peer, output_row in records:
                    encoded = peer * stride + output_row
                    value = encoded if slot == primary_slot else ~encoded
                    plans[source]["dst"][token * topk + slot] = value
                forward = plans[gateway]["forward_list"][source]
                if len(records) == 1:
                    forward.append((source, token, primary_row, -1, -1, -1))
                    continue
                for _, peer, output_row in records[1:]:
                    reduce_row = -1
                    if peer != gateway:
                        reduce_row = gathers[gateway]
                        gathers[gateway] += 1
                        plans[peer]["backward_list"][gateway % 8].append((source, output_row, reduce_row, token))
                    forward.append((source, token, primary_row, peer, output_row, reduce_row))

    for gateway, plan in enumerate(plans):
        for source, block in enumerate(plan["forward_list"]):
            cursor = 0
            for chunk in range(chunks):
                begin = cursor
                token_end = min((chunk + 1) * chunk_tokens, tokens)
                while cursor < len(block) and block[cursor][1] < token_end:
                    peer = block[cursor][3]
                    if peer >= 0 and peer != gateway:
                        plan["chunk_masks"][source * chunks + chunk] |= 1 << (peer % 8)
                    cursor += 1
                plan["chunk_ranges"][source][chunk] = (begin, cursor)
        for lane, block in enumerate(plan["backward_list"]):
            gateway_peer = gateway // 8 * 8 + lane
            gateway_steps = group_steps(gateway_peer, world)
            ordered = list(enumerate(block))
            ordered.sort(
                key=lambda item: (
                    item[1][0] // 8 == gateway_peer // 8,
                    gateway_steps[item[1][0]],
                    item[1][0],
                    item[1][3] // chunk_tokens,
                    item[0],
                )
            )
            block[:] = [item[1] for item in ordered]
            for index, (source, row, reduce_row, token) in enumerate(block):
                key = source * chunks + token // chunk_tokens + 1
                next_key = 0
                if index + 1 == len(block):
                    next_key = key
                else:
                    next_source, _, _, next_token = block[index + 1]
                    candidate = next_source * chunks + next_token // chunk_tokens + 1
                    if candidate != key:
                        next_key = key
                block[index] = (source, row, reduce_row | (next_key << 32))
        plan["forward_counts"] = [len(block) for block in plan["forward_list"]]
        plan["backward_counts"] = [len(block) for block in plan["backward_list"]]
    for plan in plans:
        plan["rank_rows"] = rank_rows
        plan["gather_rows"] = gathers
    return {"plans": plans, "rank_rows": rank_rows, "gather_rows": plans[0]["gather_rows"], "address_stride": stride}


def make_routes(world, tokens, topk, experts, generation, mode):
    result = []
    for source in range(world):
        rows = []
        for token in range(tokens):
            row = [(source * 17 + token * 7 + slot * 131 + generation) % experts for slot in range(topk)]
            if mode == "duplicates" and topk > 1:
                row[1] = row[0]
            if mode == "invalid":
                row = [
                    (experts + 17 if (source + token + slot) % 2 else -1) if (source + token + slot) % 3 == 0 else e
                    for slot, e in enumerate(row)
                ]
            if mode == "empty":
                row = [-1] * topk
            if mode == "hotspot":
                row = [experts - 1] * topk
            if mode in ("local_duplicates", "local_experts"):
                local = experts // world
                peer = (source + token) % world
                row = [peer * local + (slot % local if mode == "local_experts" else 0) for slot in range(topk)]
            rows.append(row)
        result.append(rows)
    return result


def check_handle(handle, golden, rank, routes, experts):
    import torch

    plan = golden["plans"][rank]
    if handle.metadata_only or handle.metadata_version != 1:
        raise AssertionError("Expected expanded dispatch handle v1")
    for name in (
        "server_mask",
        "dst",
        "forward_counts",
        "backward_counts",
        "chunk_ranges",
        "chunk_masks",
        "rank_rows",
        "gather_rows",
    ):
        actual = getattr(handle, name)
        if actual.device.type != "npu":
            raise AssertionError(f"{name} is not an NPU tensor")
        expected = torch.tensor(plan[name], dtype=actual.dtype).reshape(actual.shape)
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        actual = getattr(handle, name).cpu()
        for row, count in enumerate(plan[counts]):
            expected = torch.tensor(plan[name][row], dtype=actual.dtype).reshape(count, actual.shape[-1])
            if not torch.equal(actual[row, :count], expected):
                raise AssertionError(f"rank {rank}: {name}[{row}] mismatch")
    totals = [0] * experts
    for source in routes:
        for row in source:
            for expert in row:
                if 0 <= expert < experts:
                    totals[expert] += 1
    local = experts // len(routes)
    expected_counts = totals[rank * local : (rank + 1) * local]
    if handle.num_recv_tokens_per_expert_list != expected_counts:
        raise AssertionError("Expert counts mismatch")
    for name, expected in (
        ("expert_totals", torch.tensor(totals, dtype=torch.int32)),
        ("num_unaligned_recv_tokens_per_expert", torch.tensor(expected_counts, dtype=torch.int32)),
        (
            "psum_num_recv_tokens_per_expert",
            torch.tensor(expected_counts, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        ),
    ):
        if not torch.equal(getattr(handle, name).cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    if handle.num_recv_tokens is not None or handle.dst_buffer_slot_idx is not None:
        raise AssertionError("Expanded dispatch must leave ordinary-layout fields unset")
    if handle.address_stride != golden["address_stride"]:
        raise AssertionError("Destination stride mismatch")
    if handle.num_expanded_tokens != golden["rank_rows"][rank]:
        raise AssertionError("Expanded token count mismatch")


def _import_installed_deep_ep():
    # Launching from the checkout or inheriting PYTHONPATH must not shadow the wheel.
    locations = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    spec = PathFinder.find_spec("deep_ep", locations)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"deep_ep is not installed in {sys.executable}; "
            "run python -m pip install --no-build-isolation --no-deps . first"
        )
    expected = Path(spec.origin).resolve()
    sys.path.insert(0, str(Path(spec.origin).parent.parent))
    package = importlib.import_module("deep_ep")
    actual = Path(package.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"deep_ep was already loaded from {actual}; expected installed package {expected}")
    return package


def _add_launch_args(parser):
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8899)
    parser.add_argument("--shmem-ip-port", help="Defaults to DEEPEP_SHMEM_ENDPOINT or master-port + 1")


def _launch(args, worker):
    # torchrun may still supply process identities, but no process group is created.
    launched = [name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if any(launched) and not all(launched):
        raise ValueError("RANK, WORLD_SIZE and LOCAL_RANK must be supplied together")
    if all(launched):
        args.ep_size = int(os.environ["WORLD_SIZE"])
    elif (
        args.nproc_per_node < 1
        or args.ep_size % args.nproc_per_node
        or not 0 <= args.node_rank < args.ep_size // args.nproc_per_node
    ):
        raise ValueError("EP must be divisible by nproc-per-node; node-rank must be within the job")
    if args.ep_size not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError("Dispatch/combine EP must be a power of two from 2 to 128")
    endpoint = args.shmem_ip_port or os.environ.get("DEEPEP_SHMEM_ENDPOINT")
    if not endpoint:
        master = os.environ.get("MASTER_ADDR", args.master_addr) if all(launched) else args.master_addr
        port = int(os.environ.get("MASTER_PORT", args.master_port)) if all(launched) else args.master_port
        endpoint = f"tcp://{master}:{port + 1}"
    os.environ["DEEPEP_SHMEM_ENDPOINT"] = endpoint
    if all(launched):
        worker(int(os.environ["LOCAL_RANK"]), args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.nproc_per_node, join=True)


def _worker_identity(local_rank, args):
    rank = int(os.environ.get("RANK", args.node_rank * args.nproc_per_node + local_rank))
    if not 0 <= rank < args.ep_size or local_rank < 0:
        raise ValueError("Invalid rank or local rank")
    return rank, args.ep_size


def expert_rows(routes, experts, rank):
    """Independent destination order: local expert, source rank, token, top-k slot."""
    local = experts // len(routes)
    return [
        (source, token, slot)
        for expert in range(rank * local, (rank + 1) * local)
        for source, tokens in enumerate(routes)
        for token, row in enumerate(tokens)
        for slot, selected in enumerate(row)
        if selected == expert
    ]


def test_forward_list_covers_unique_expert_rows():
    # Exercise both primary core maps, cross-server forwarding, same-rank
    # duplicates, singleton records, empty ranks and non-power-of-two strides.
    for world in (2, 8, 16, 64, 128):
        for mode in ("normal", "duplicates", "invalid", "empty", "hotspot"):
            tokens, topk, experts = 3, 6, world * 2
            routes = make_routes(world, tokens, topk, experts, 0, mode)
            golden = build_notify_golden(routes, experts, 256)
            stride = golden["address_stride"]
            output = [[None] * n for n in golden["rank_rows"]]

            def write(peer, row, value):
                assert output[peer][row] is None, "Two tasks write the same output row"
                output[peer][row] = value

            for source, plan in enumerate(golden["plans"]):
                for route, encoded in enumerate(plan["dst"]):
                    if encoded >= 0:
                        peer, row = divmod(encoded, stride)
                        write(peer, row, (source, route // topk))
            primary = [list(rows) for rows in output]
            workers = 16 if world == 128 else 24
            for gateway, plan in enumerate(golden["plans"]):
                for source, records in enumerate(plan["forward_list"]):
                    covered = []
                    for worker in range(workers):
                        begin, end = len(records) * worker // workers, len(records) * (worker + 1) // workers
                        covered.extend(range(begin, end))
                        for record in records[begin:end]:
                            src, token, primary_row, peer, row, _ = record
                            assert src == source and primary[gateway][primary_row] == (source, token)
                            if peer >= 0:
                                assert gateway // 8 == peer // 8
                                write(peer, row, primary[gateway][primary_row])
                    assert covered == list(range(len(records)))
            for rank in range(world):
                assert output[rank] == [(source, token) for source, token, _ in expert_rows(routes, experts, rank)]


def test_dispatch_regions_and_encoding_bounds():
    import pytest
    from deep_ep.buffers._elastic_layout import dispatch_layout, workspace_layout

    for world in (2, 8, 16, 64, 128):
        for fp8 in (False, True):
            for rows in (1, 3, world * 33 * 6):
                layout = dispatch_layout(world, 33, 6, 1024, rows, fp8)
                assert layout["sync"] == workspace_layout(world, 33, 6)["workspace_bytes"]
                assert rows <= layout["address_stride"] < 2 * rows
                offsets = [layout[name] for name in ("sync", "book", "input", "output", "weights", "scales", "inbox")]
                assert offsets == sorted(offsets) and all(n % 512 == 0 for n in offsets)
                row_bytes = 7168 if fp8 else 14336
                assert layout["weights"] - layout["output"] >= rows * row_bytes
                assert layout["workspace_bytes"] == layout["inbox"] + world * layout["inbox_stride"]
                assert layout["num_bytes"] >= layout["workspace_bytes"]
                assert layout["num_bytes"] % (2 << 20) == 0
    for arguments in ((256, 1, 1, 1024, 1), (2, 1, 1, 1024, 3), (2, 1, 1, 1024, 0)):
        with pytest.raises(ValueError):
            dispatch_layout(*arguments)


def assert_bytes(actual, expected, name, progress=None):
    import torch

    if progress:
        progress(f"{name}: readback.begin (waiting for device result)")
    actual = actual.cpu().contiguous()
    if progress:
        progress(f"{name}: readback.done; precision.begin")
    expected = expected.cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    a, b = actual.view(torch.uint8).flatten(), expected.view(torch.uint8).flatten()
    bad = torch.nonzero(a != b).flatten()
    if bad.numel():
        first = bad[:8]
        raise AssertionError(
            f"{name}: {bad.numel()} mismatched bytes; offsets={first.tolist()}, "
            f"actual={a[first].tolist()}, expected={b[first].tolist()}"
        )
    if progress:
        progress(f"{name}: precision.passed")


def payload(source, tokens, generation, fp8):
    import torch

    # Bounded independently of EP, token count and hidden size. Neither BF16
    # nor FP8 construction can overflow; no arithmetic runs in dispatch.
    rows = torch.arange(tokens, dtype=torch.int32)[:, None]
    columns = torch.arange(7168, dtype=torch.int32)[None, :]
    hidden = ((source * 13 + rows * 7 + columns * 3 + generation * 11) % 127 - 63).float() / 16
    hidden = hidden.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    scales = ((source + rows + torch.arange(56)[None, :] + generation) % 7 + 1).float() / 8
    return hidden, scales


def weights_for(source, tokens, topk, generation):
    import torch

    return ((torch.arange(tokens * topk).reshape(tokens, topk) + source * 3 + generation) % 31 + 1).float() / 32


def expected_payload(routes, experts, rank, generation, fp8):
    import torch

    entries = expert_rows(routes, experts, rank)
    tokens, topk = len(routes[0]), len(routes[0][0])
    source_payload = [payload(source, tokens, generation, fp8) for source in range(len(routes))]
    source_weights = [weights_for(source, tokens, topk, generation) for source in range(len(routes))]
    # Byte indexing supports CPU builds that do not implement FP8 index/stack.
    if entries:
        hidden_bytes = torch.stack([source_payload[s][0][t].view(torch.uint8) for s, t, _ in entries])
        hidden = hidden_bytes.view(source_payload[0][0].dtype)
        scales = torch.stack([source_payload[s][1][t] for s, t, _ in entries])
        weights = torch.stack([source_weights[s][t, k] for s, t, k in entries])
    else:
        hidden = torch.empty((0, 7168), dtype=source_payload[0][0].dtype)
        scales, weights = torch.empty((0, 56), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
    return hidden, scales, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _add_launch_args(parser)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp8", "both"), default="both")
    parser.add_argument(
        "--route", choices=("normal", "duplicates", "invalid", "empty", "hotspot", "all"), default="all"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE") or args.repeat < 2:
        parser.error("Enable assertions and use --repeat >=2")
    _launch(args, worker)


def worker(local_rank, args):
    rank, world = _worker_identity(local_rank, args)
    log = _progress_logger("dispatch", rank)
    log("worker.begin; import installed package")
    deep_ep = _import_installed_deep_ep()
    import torch
    import torch_npu  # noqa: F401
    from deep_ep.buffers._elastic_layout import dispatch_layout

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    if "elastic-dispatch-expanded-v1" not in info.get("capabilities", []):
        raise RuntimeError("Installed wheel does not contain payload dispatch")
    torch.npu.set_device(local_rank)
    modes = ("normal", "duplicates", "invalid", "empty", "hotspot") if args.route == "all" else (args.route,)
    dtypes = (False, True) if args.dtype == "both" else (args.dtype == "fp8",)
    if local_rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    # Small correctness runs reserve the exact all-to-one bound. The public
    # hint budgets four times average rows and is intentionally smaller at EP>4.
    log("capacity.begin")
    capacity = 2 * max(
        dispatch_layout(world, args.tokens, args.topk, args.experts, world * args.tokens * args.topk, fp8)["num_bytes"]
        for fp8 in dtypes
    )
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
    checks, previous_output = 0, None
    try:
        for fp8 in dtypes:
            for mode in modes:
                context = f"dtype={'fp8' if fp8 else 'bf16'} route={mode}"
                log(f"{context} routes_and_notify_golden.begin (CPU)")
                routes = make_routes(world, args.tokens, args.topk, args.experts, 0, mode)
                notify_golden = build_notify_golden(routes, args.experts, 256)
                log(f"{context} routes_and_notify_golden.done")
                for with_weights in (False, True):
                    handle = None
                    for generation in range(args.repeat):
                        use_weights = with_weights if generation % 2 == 0 else not with_weights
                        iteration = (
                            f"{context} weights_case={with_weights} weights={use_weights} "
                            f"iter={generation + 1}/{args.repeat}"
                        )
                        log(f"{iteration} inputs.begin")
                        local, scales = payload(rank, args.tokens, generation, fp8)
                        local_weights = weights_for(rank, args.tokens, args.topk, generation)
                        producer = torch.npu.Stream()
                        with torch.npu.stream(producer):
                            x = (local.npu(), scales.npu()) if fp8 else local.npu()
                            w = local_weights.npu() if use_weights else None
                            indices = (
                                torch.tensor(routes[rank], dtype=torch.int64, device="npu") if handle is None else None
                            )
                            before = buffer.capture()
                        old_handle = handle
                        log(f"{iteration} dispatch.enter mode={'fresh' if handle is None else 'cached'}")
                        output, index, weights, handle, event = buffer.dispatch(
                            x,
                            topk_idx=indices,
                            topk_weights=w,
                            num_experts=args.experts if handle is None else None,
                            handle=handle,
                            previous_event=before,
                            async_with_compute_stream=True,
                            do_expand=True,
                        )
                        log(f"{iteration} dispatch.return (async; device completion not yet confirmed)")
                        event.current_stream_wait()
                        if index is not None or handle.metadata_only:
                            raise AssertionError("Expected expanded payload output")
                        if old_handle is None:
                            log(f"{iteration} handle_check.begin (includes device readback)")
                            check_handle(handle, notify_golden, rank, routes, args.experts)
                            log(f"{iteration} handle_check.done")
                        if old_handle is not None and old_handle is not handle:
                            raise AssertionError("Cached dispatch replaced its routing handle")
                        log(f"{iteration} payload_golden.begin (CPU)")
                        expected, expected_scales, expected_weights = expected_payload(
                            routes, args.experts, rank, generation, fp8
                        )
                        log(f"{iteration} payload_golden.done")
                        data = output[0] if fp8 else output
                        assert_bytes(data, expected, f"{iteration} dispatch hidden", progress=log)
                        log(f"{iteration} remaining_checks.begin")
                        if fp8:
                            assert_bytes(output[1], expected_scales, "scales")
                        if use_weights:
                            assert_bytes(weights, expected_weights, "weights")
                        elif weights is not None:
                            raise AssertionError("Weights returned when none were supplied")
                        if previous_output is not None:
                            for prior, expected_prior in previous_output:
                                assert_bytes(prior, expected_prior, "retained output across cached/fresh calls")
                        previous_output = [(data, expected)]
                        if fp8:
                            previous_output.append((output[1], expected_scales))
                        if use_weights:
                            previous_output.append((weights, expected_weights))
                        if indices is not None:
                            indices.fill_(-1)
                            assert_bytes(handle.topk_idx, torch.tensor(routes[rank], dtype=torch.int64), "routing copy")
                        if expected.numel():
                            corrupt = expected.view(torch.uint8).clone()
                            corrupt.flatten()[0] ^= 1
                            try:
                                assert_bytes(data, corrupt.view(expected.dtype), "deliberate corruption")
                            except AssertionError:
                                pass
                            else:
                                raise AssertionError("Precision comparator failed to detect corruption")
                        checks += 1
                        log(f"{iteration} iteration.passed")
                    print(
                        f"rank={rank} dtype={'fp8' if fp8 else 'bf16'} route={mode} weights={with_weights} passed",
                        flush=True,
                    )
    except BaseException:
        _log_failure(log)
        raise
    finally:
        log("buffer.destroy.begin (may wait for device work/peers)")
        buffer.destroy()
        log("buffer.destroy.done")
    # Returned tensors must also remain valid after workspace destruction.
    if previous_output is not None:
        for prior, expected_prior in previous_output:
            assert_bytes(prior, expected_prior, "retained output after destroy")
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                checks=checks,
                passed=True,
                source_sha256=info["source_sha256"],
                tokens=args.tokens,
                topk=args.topk,
                experts=args.experts,
                dtype=args.dtype,
                route=args.route,
            ),
            indent=2,
        )
        + "\n"
    )


def test_direct_and_external_launcher(monkeypatch):
    from types import SimpleNamespace
    import pytest

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "DEEPEP_SHMEM_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    args = parser.parse_args(
        ["--ep-size", "64", "--nproc-per-node", "8", "--node-rank", "5", "--master-addr", "141.61.54.91"]
    )
    calls = []

    def spawn(worker, args, nprocs, join):
        assert nprocs == 8 and join
        for local_rank in range(nprocs):
            worker(local_rank, *args)

    mp = SimpleNamespace(spawn=spawn)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(multiprocessing=mp))
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)

    def worker(local_rank, args):
        calls.append(_worker_identity(local_rank, args))

    _launch(args, worker)
    assert calls == [(rank, 64) for rank in range(40, 48)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://141.61.54.91:8900"
    calls.clear()
    monkeypatch.setenv("RANK", "23")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("LOCAL_RANK", "7")
    _launch(args, worker)
    assert calls == [(23, 32)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://141.61.54.91:8900"
    monkeypatch.delenv("WORLD_SIZE")
    with pytest.raises(ValueError, match="supplied together"):
        _launch(args, worker)


def test_hand_computed_routes():
    golden = build_notify_golden([[[0, 2]], [[1, 2]]], 4, 256)
    assert golden["rank_rows"] == [2, 2]
    assert golden["plans"][0]["dst"] == [0, ~2]
    assert golden["plans"][1]["dst"] == [1, ~3]
    duplicate = build_notify_golden([[[0, 0]], [[-1, -1]]], 4, 256)
    assert duplicate["rank_rows"] == [2, 0]
    assert duplicate["plans"][0]["dst"] == [0, ~1]


def test_installed_import_ignores_checkout_and_pythonpath(tmp_path):
    import subprocess

    installed = tmp_path / "site-packages"
    checkout = tmp_path / "checkout"
    for root, marker in ((installed, "installed"), (checkout, "source")):
        (root / "deep_ep").mkdir(parents=True)
        (root / "deep_ep/__init__.py").write_text(f"marker = {marker!r}\n", encoding="utf-8")
    # A separate process keeps pytest's already-imported source package intact.
    script = """
import runpy, sys, sysconfig
namespace = runpy.run_path(sys.argv[1])
sysconfig.get_path = lambda name: sys.argv[2]
mode = sys.argv[3]
if mode == "cached":
    import deep_ep
    assert deep_ep.marker == "source"
try:
    package = namespace["_import_installed_deep_ep"]()
except RuntimeError as error:
    assert mode != "installed", str(error)
    assert ("already loaded" if mode == "cached" else "not installed") in str(error)
else:
    assert mode == "installed"
    assert package.marker == "installed"
"""
    for mode in ("installed", "cached", "missing"):
        location = installed if mode != "missing" else tmp_path / "empty-site-packages"
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve()), str(location), mode],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": str(checkout)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    main()
