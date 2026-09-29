"""Installed-wheel expanded dispatch/combine functional precision test.

NPU: python tests/elastic/test_combine.py --ep-size 2 --nproc-per-node 2 --output /tmp/combine-check
"""

import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys

_dispatch = runpy.run_path(str(Path(__file__).with_name("test_dispatch.py")))
_ROUTES = ("normal", "duplicates", "invalid", "empty", "local_duplicates", "local_experts")


def build_reference(routes, experts, rank, chunk_tokens=256):
    source = Path(__file__).resolve().parents[2] / "third-party/ascend-deepep/put_combine_plan.py"
    spec = importlib.util.spec_from_file_location("_deepep_combine_reference", source)
    planner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = planner
    spec.loader.exec_module(planner)
    # Replace only its synthetic route generator, not metadata construction.
    # Upstream represents every invalid expert with -1.
    planner.route_experts = lambda source, token, case: [
        expert if 0 <= expert < experts else -1 for expert in routes[source][token]
    ]
    world, tokens, topk = len(routes), len(routes[0]), len(routes[0][0])
    case = planner.CombineCase(world, tokens, topk, experts, chunk_tokens=chunk_tokens)
    plan = planner.build_plan(case, rank)
    result = {
        "forward_list": [list(plan.forward_rows(source)) for source in range(world)],
        "backward_list": [list(plan.backward_rows(lane)) for lane in range(plan.lanes)],
        "forward_counts": list(plan.forward_count),
        "backward_counts": list(plan.backward_count),
        "chunk_ranges": list(plan.chunk_ranges),
        "chunk_masks": list(plan.chunk_masks),
        "server_mask": list(plan.server_mask),
        "rank_rows": list(plan.rank_rows),
        "gather_rows": list(plan.gather_rows),
        "weight_return_meta": list(plan.route_ids),
    }
    return result


def assert_metadata_equal(actual, expected, label):
    """Report the first differing coordinate, including packed chunk markers."""
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: length actual={len(actual)}, expected={len(expected)}")
        for index, (value, reference) in enumerate(zip(actual, expected)):
            assert_metadata_equal(value, reference, f"{label}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def check_handle(handle, reference, rank):
    if handle.metadata_only or handle.metadata_version != 4:
        raise AssertionError(f"rank {rank}: expected payload dispatch metadata v4")
    # Check counts first, then copy/compare only valid rows of padded lists.
    for name in (
        "forward_counts",
        "backward_counts",
        "server_mask",
        "rank_rows",
        "gather_rows",
        "chunk_ranges",
        "chunk_masks",
    ):
        actual = getattr(handle, name).detach().cpu().reshape(-1).tolist()
        assert_metadata_equal(actual, reference[name], f"rank {rank}: {name}")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        for lane, count in enumerate(reference[counts]):
            actual = getattr(handle, name)[lane, :count].detach().cpu().tolist()
            assert_metadata_equal(actual, reference[name][lane], f"rank {rank}: {name}[{lane}]")
    assert_metadata_equal(
        handle.weight_return_meta.detach().cpu().tolist(),
        reference["weight_return_meta"],
        f"rank {rank}: weight_return_meta",
    )


def payload(source, tokens, generation):
    import torch

    return ((torch.arange(tokens)[:, None] + torch.arange(7168)[None, :] + source + generation) % 7 - 3).to(
        torch.bfloat16
    )


def expected_output(routes, experts, rank, generation):
    import torch

    base = payload(rank, len(routes[rank]), generation).float()
    selected = torch.tensor(routes[rank], dtype=torch.int64)
    valid = (selected >= 0) & (selected < experts)
    slots = torch.arange(selected.shape[1])[None, :]
    count = valid.sum(dim=1, keepdim=True)
    bias = ((selected % 3 - 1 + slots % 3 - 1) * valid).sum(dim=1, keepdim=True)
    # Bounded integer fixtures make this factored sum exact in FP32/BF16.
    out = base * count + bias
    return out.to(torch.bfloat16)


def weight_payload(ids, generation):
    import torch

    bits = 0x3E800000 + ((ids + generation * 104729) % (1 << 25))
    return bits.to(torch.int32).view(torch.float32)


def expected_weights(routes, experts, rank, generation):
    import torch

    selected = torch.tensor(routes[rank], dtype=torch.int64)
    tokens, topk = selected.shape
    ids = torch.arange(tokens * topk).reshape(tokens, topk) + rank * tokens * topk
    return weight_payload(ids, generation).masked_fill((selected < 0) | (selected >= experts), 0)


def worker(local_rank, args):
    import torch
    import torch_npu  # noqa: F401

    deep_ep = _dispatch["_import_installed_deep_ep"]()
    from deep_ep.buffers._elastic_layout import elastic_size_hint

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    if "elastic-combine-expanded-v3" not in info.get("capabilities", []):
        raise RuntimeError("Installed wheel does not contain expanded Combine; rebuild/install first")
    if "elastic-notify-metadata-v4" not in info.get("capabilities", []):
        raise RuntimeError("Rebuild/install the wheel with Notify metadata v4")
    torch.npu.set_device(local_rank)
    rank, world = _dispatch["_worker_identity"](local_rank, args)
    log = _dispatch["_progress_logger"]("combine", rank)
    log("worker.begin")
    modes = _ROUTES if args.route == "all" else (args.route,)
    capacity = 0
    references = {}
    for mode in modes:
        log(f"route={mode} capacity_golden.begin (CPU, before buffer initialization)")
        routes = _dispatch["make_routes"](world, args.tokens, args.topk, args.experts, 0, mode)
        plan = build_reference(routes, args.experts, rank)
        references[mode] = plan
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
    checks, retained, retained_weights = 0, None, None
    try:
        for mode in modes:
            log(f"route={mode} routes_and_expert_rows.begin (CPU)")
            routes = _dispatch["make_routes"](world, args.tokens, args.topk, args.experts, 0, mode)
            entries = _dispatch["expert_rows"](routes, args.experts, rank)
            log(f"route={mode} routes_and_expert_rows.done rows={len(entries)}")
            reference = references.pop(mode)
            log(f"route={mode} notify_reference.reused")
            sources = torch.tensor([s + t for s, t, _ in entries], dtype=torch.int32)[:, None]
            weight_ids = torch.tensor([(s * args.tokens + t) * args.topk + k for s, t, k in entries], dtype=torch.int64)
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
                log(f"{context} notify_metadata.check.begin")
                check_handle(handle, reference, rank)
                log(f"{context} notify_metadata.check.passed")
                log(f"{context} dispatch_golden.begin (CPU)")
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
                    weights = None if args.without_weights else weight_payload(weight_ids, generation).npu()
                    before = buffer.capture()
                log(f"{context} expert_compute.enqueued; combine.enter")
                output, values, completed = buffer.combine(
                    expert, handle, topk_weights=weights, previous_event=before, async_with_compute_stream=True
                )
                log(f"{context} combine.return (async; device completion not yet confirmed)")
                completed.current_stream_wait()
                weight_golden = None
                if weights is None:
                    assert values is None
                else:
                    weight_golden = expected_weights(routes, args.experts, rank, generation)
                    _dispatch["assert_bytes"](values, weight_golden, f"{context} weights", progress=log)
                log(f"{context} combine_golden.begin (CPU)")
                golden = expected_output(routes, args.experts, rank, generation)
                log(f"{context} combine_golden.done")
                _dispatch["assert_bytes"](output, golden, f"{context} combine", progress=log)
                if args.profile_op and generation == 0:

                    def invoke():
                        result = buffer.combine(expert, handle, topk_weights=weights, async_with_compute_stream=True)
                        result[-1].current_stream_wait()
                        return result

                    def validate(result):
                        if weight_golden is None:
                            assert result[1] is None
                        else:
                            _dispatch["assert_bytes"](result[1], weight_golden, "profile combine weights")
                        _dispatch["assert_bytes"](result[0], golden, "profile combine")

                    _dispatch["profile_operator"](
                        args,
                        rank,
                        f"{mode}-bf16-weights{int(weights is not None)}",
                        invoke,
                        validate,
                        info,
                        log,
                    )
                # The same handle and workspace must support consecutive combines.
                log(f"{context} repeated_combine.enter")
                again, no_weights, completed = buffer.combine(expert, handle)
                log(f"{context} repeated_combine.return (device completion not yet confirmed)")
                completed.current_stream_wait()
                assert no_weights is None
                _dispatch["assert_bytes"](again, golden, f"{context} repeated combine", progress=log)
                if weights is not None:
                    changed = weight_payload(weight_ids, generation + 1).npu()
                    again, changed_out, completed = buffer.combine(expert, handle, topk_weights=changed)
                    completed.current_stream_wait()
                    _dispatch["assert_bytes"](again, golden, f"{context} changed weights hidden")
                    _dispatch["assert_bytes"](
                        changed_out,
                        expected_weights(routes, args.experts, rank, generation + 1),
                        f"{context} changed weights",
                    )
                    _dispatch["assert_bytes"](values, weight_golden, f"{context} retained weights")
                log(f"{context} retained_output_and_negative_checks.begin")
                if retained is not None:
                    _dispatch["assert_bytes"](*retained, "retained output")
                retained = (output, golden)
                if retained_weights is not None:
                    _dispatch["assert_bytes"](*retained_weights, "retained weights from previous iteration")
                if values is not None:
                    retained_weights = (values, weight_golden)
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
    if retained_weights is not None:
        _dispatch["assert_bytes"](*retained_weights, "weights after destroy")
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
                with_weights=not args.without_weights,
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


def test_encoded_weight_metadata():
    for world in (2, 16):
        tokens, topk, experts = 7, 6, world * 2
        for mode in _ROUTES:
            routes = _dispatch["make_routes"](world, tokens, topk, experts, 0, mode)
            all_ids = []
            for rank in range(world):
                plan = build_reference(routes, experts, rank)
                ids = plan["weight_return_meta"]
                decoded = [(value // (tokens * topk), value // topk % tokens, value % topk) for value in ids]
                assert decoded == _dispatch["expert_rows"](routes, experts, rank)
                all_ids.extend(ids)
            expected = [
                (source * tokens + token) * topk + slot
                for source in range(world)
                for token in range(tokens)
                for slot, expert in enumerate(routes[source][token])
                if 0 <= expert < experts
            ]
            assert sorted(all_ids) == expected
            assert len(all_ids) == len(set(all_ids))


def test_sq_pressure_routes():
    routes = _dispatch["make_routes"](128, 8, 1, 128, 0, "sq_pressure")
    active = [rank for rank, rows in enumerate(routes) if rows[0] == [0]]
    assert active == [8, 20, 48, 52, 80, 84, 112, 116]
    assert all(row == ([0] if rank in active else [-1]) for rank, rows in enumerate(routes) for row in rows)
    reference = build_reference(routes, 128, 0)
    assert reference["rank_rows"] == [64] + [0] * 127


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

    def section(begin, end):
        return code[code.index(begin) : code.index(end, code.index(begin))]

    helpers = section("__aicore__ inline uint32_t CombineGroup(", "__aicore__ inline uint32_t CombinePeer(")
    helpers += section("__aicore__ inline uint32_t CrossPeerOwner(", "__aicore__ inline uint32_t SelfCopyDoneSlot(")
    helpers += section(
        "template <typename TaskReader>\n__aicore__ inline bool TokenNeedsGatewayReduce(",
        "template <bool EnableTrace>\n__aicore__ inline void ReduceGateway(",
    )
    helpers += section("constexpr uint32_t kWeightThreads =", "__aicore__ inline __gm__ uint64_t* WeightFlag(")
    credit_test = (
        r"""
namespace credit_test {
constexpr uint32_t kWqeBytes = 64, kWqeBatchScratchBytes = 128 * kWqeBytes;
constexpr uint32_t ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD = 10, EVENT_ID0 = 0;
namespace shm { constexpr uint32_t UDMA_SQ_BASKBLK_CNT = 32768; }
struct aclshmemi_udma_queue_state_t {
    uint32_t sq_head = 0, sq_tail = 0, cqe_cnt = 0, cq_tail = 0;
};
struct Context { uint32_t depth; uintptr_t state_addr, buf_addr; } contexts[2];
aclshmemi_udma_queue_state_t queues[2];
uint32_t completed_head[2]{}, polls[2]{}, copies = 0;
uint8_t rings[2][32768 * kWqeBytes]{}, scratch[2 * kWqeBatchScratchBytes]{};
bool stuck = false, expect_failure = false;
std::jmp_buf failure;
struct Address { uint8_t logicPos; uint64_t bufferAddr; };
template<class T> struct LocalTensor { Address address_{}; };
enum class TPosition { VECOUT };
int* aclshmemi_udma_qp_info_fetch() { return nullptr; }
Context* aclshmemi_udma_get_qp_ctx(int*, uint32_t, uint32_t qp) {
    assert(qp == 16 || qp == 17); return &contexts[qp - 16];
}
void aclshmemx_udma_qp_quiet(uint32_t, uint32_t qp) {
    auto lane = qp - 16; auto& q = queues[lane]; ++polls[lane];
    if (!stuck) { q.sq_tail = completed_head[lane]; q.cq_tail = q.cqe_cnt; }
}
template<class... Args> [[noreturn]] void aclshmemi_kernel_abort(const char*, Args...) {
    if (expect_failure) std::longjmp(failure, 1);
    std::abort();
}
void aclshmemi_udma_copy_wqe_from_ub(uint8_t* dst, LocalTensor<uint8_t> src, uint32_t bytes, uint32_t) {
    ++copies; std::memcpy(dst, reinterpret_cast<void*>(src.address_.bufferAddr), bytes);
}
void aclshmemi_udma_post_send_update_info(uint32_t head, Context*, aclshmemi_udma_queue_state_t* q) {
    assert(uint32_t(head - q->sq_tail) < shm::UDMA_SQ_BASKBLK_CNT); q->sq_head = head;
}
struct Batch {
    struct { uint32_t pending_count = 0; } state[2];
    uint32_t peer[2]{8, 8}, so_reserve = 16;
"""
        + section("    __aicore__ inline void FlushLane(", "    __aicore__ inline void Flush(")
        + r"""
};
void reset(uint32_t start = 0) {
    stuck = false; copies = 0;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        queues[lane] = {start, start, 0, 0}; completed_head[lane] = start; polls[lane] = 0;
        contexts[lane] = {32768, reinterpret_cast<uintptr_t>(&queues[lane]), reinterpret_cast<uintptr_t>(rings[lane])};
    }
}
void expect_abort(Batch& b) {
    const auto before = copies; expect_failure = true;
    if (setjmp(failure) == 0) { b.FlushLane(0, 8, scratch); assert(false); }
    expect_failure = false; assert(copies == before);
}
void simulate(uint32_t tokens, bool empty = false) {
    auto t = ascend_deepep::MakeCombineTiling(0, 128, tokens, 1, 8 * tokens, 1, 256);
    std::vector<uint32_t> peers;
    for (uint32_t p = 8; p < 128; ++p) if (CrossPeerAssigned(t, p, 8)) peers.push_back(p);
    assert((peers == std::vector<uint32_t>{8,20,48,52,80,84,112,116}));
    Batch b; b.so_reserve = 0;
    for (auto p:peers) for (uint32_t c = 0; c < t.chunks_per_group; ++c)
        if (CrossSoRequired(t, 0, p, c)) ++b.so_reserve;
    assert(b.so_reserve == (tokens <= 256 ? 8U : 16U));
    uint32_t payload[2]{};
    for (uint32_t c = 0; c < t.chunks_per_group; ++c) {
        auto begin = c ? std::min<uint64_t>(tokens, ChunkTokenEnd(t,c-1)) : 0;
        auto end = std::min<uint64_t>(tokens, ChunkTokenEnd(t,c));
        for (uint32_t g = 0; g < 2; ++g) {
            std::vector<uint32_t> group;
            for (auto p:peers) if (CombineGroup(0,128,p,t.group_size) == g) group.push_back(p);
            uint32_t left[2][8]{};
            if (!empty) for (uint32_t i = 0; i < group.size(); ++i)
                for (uint32_t n = 0; n < end - begin; ++n) ++left[n % 4 == 3 ? 1 : 0][i];
            bool remaining = true;
            while (remaining) {
                remaining = false;
                for (uint32_t lane = 0; lane < 2; ++lane) for (uint32_t i = 0; i < group.size(); ++i) {
                    auto n = std::min(left[lane][i], t.cross_tokens_per_turn); left[lane][i] -= n;
                    for (uint32_t j = 0; j < n; ++j) {
                        ++payload[lane]; b.peer[lane] = group[i];
                        if (++b.state[lane].pending_count == 32) b.FlushLane(lane,8,scratch);
                    }
                    remaining |= left[lane][i] != 0;
                }
            }
            for (uint32_t lane = 0; lane < 2; ++lane) b.FlushLane(lane,8,scratch);
            for (auto p:group) if (CrossSoRequired(t,0,p,c)) for (uint32_t lane = 0; lane < 2; ++lane) {
                auto& q = queues[lane];
                assert(uint32_t(q.sq_head - q.sq_tail) + 1 + 10 < 32768);
                ++q.sq_head; ++q.cqe_cnt; completed_head[lane] = q.sq_head;
            }
        }
    }
    if (tokens == 6000 && !empty) assert(payload[0] == 36000 && payload[1] == 12000);
    if (tokens <= 4096 || empty) assert(polls[0] == 0 && polls[1] == 0);
    else assert(polls[0] > 0 && polls[1] == 0);
    // Final quiet must still drain both lanes before the next invocation.
    for (uint32_t lane = 0; lane < 2; ++lane) {
        aclshmemx_udma_qp_quiet(8,16+lane);
        assert(queues[lane].sq_head == queues[lane].sq_tail);
        polls[lane] = 0;
    }
}
void run() {
    // Bound each no-SO interval for every supported token count at chunk=256.
    for (uint32_t tokens = 1; tokens <= 10240; ++tokens) {
        auto t = ascend_deepep::MakeCombineTiling(0,128,tokens,1,1,1,256);
        uint32_t previous = 0, lane0 = 0;
        for (uint32_t c = 0; c < t.chunks_per_group; ++c) {
            auto end = std::min<uint64_t>(tokens,ChunkTokenEnd(t,c));
            auto count = end - previous; previous = end;
            lane0 += count - count / 4;
            if (CrossSoRequired(t,0,8,c)) {
                assert(lane0 <= 3840);
                assert(uint64_t(lane0) * 8 + 32 + 16 + 10 < 32768); lane0 = 0;
            }
        }
        assert(lane0 == 0);
    }
    for (uint32_t world : {16U,32U,64U,128U}) for (uint32_t rank = 0; rank < world; ++rank) {
        auto t = ascend_deepep::MakeCombineTiling(rank,world,10240,1,1,1,256);
        for (uint32_t owner = 0; owner < 16; ++owner) {
            uint32_t peers = 0, so = 0;
            for (uint32_t p = 0; p < world; ++p) if (p/8 != rank/8 && CrossPeerAssigned(t,p,owner)) {
                ++peers;
                for (uint32_t c = 0; c < t.chunks_per_group; ++c)
                    if (CrossChunkOwner(t,rank,p,c) == owner && CrossSoRequired(t,rank,p,c)) ++so;
            }
            assert(peers <= 8 && so <= 16);
        }
    }
    reset(); Batch b;
    b.FlushLane(0,8,scratch); assert(copies == 0 && polls[0] == 0);
    queues[0].sq_head = 32712; b.state[0].pending_count = 32;
    expect_abort(b); assert(polls[0] == 0); // No CQE: never spin on quiet.
    queues[0].cqe_cnt = 1; completed_head[0] = 18440; stuck = true;
    expect_abort(b); assert(polls[0] == 1); // Recheck credit after polling.
    stuck = false; b.FlushLane(0,8,scratch);
    assert(queues[0].sq_tail == 18440 && queues[0].sq_head == 32744);
    assert(b.state[0].pending_count == 0 && queues[0].cqe_cnt == 1);
    reset(); contexts[0].depth = 1024; b.state[0].pending_count = 1;
    expect_abort(b); assert(polls[0] == 0);
    reset(32760); b.state[0].pending_count = 32;
    std::memset(scratch, 0x5a, 32*kWqeBytes); b.FlushLane(0,8,scratch);
    assert(copies == 2 && polls[0] == 0);
    for (uint32_t i = 0; i < 32*kWqeBytes; ++i) assert(rings[0][(32760*kWqeBytes+i)%sizeof(rings[0])] == 0x5a);
    reset(); simulate(256); simulate(4096); simulate(6000,true);
    reset(); simulate(6000); simulate(6000); // Second invocation crosses the 16-bit CQE index range.
    reset(UINT32_MAX-100); simulate(10240); simulate(10240);
}
} // namespace credit_test
"""
    )
    harness = tmp_path / "combine_predicates.cpp"
    harness.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <csetjmp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#define __aicore__
#define __simt_vf__
#define __gm__
#define __ubuf__
#define __launch_bounds__(n)
#include "kernels/common.hpp"
#include "kernels/combine/tiling.hpp"
#include "kernels/notify/tiling.hpp"
using CombineTiling = ascend_deepep::PutCombineTiling;
namespace combine_control = ascend_deepep::combine_control;
struct { uint32_t x; } threadIdx;
void asc_threadfence() {}
namespace simt {
    std::vector<std::vector<float>> receive;
    std::vector<std::vector<uint64_t>> done;
    std::vector<std::vector<uint32_t>> writes;
    void aclshmem_float_p(float* address, float value, int32_t peer) {
        const auto index = address - receive[0].data();
        assert(++writes.at(peer).at(index) == 1);
        receive.at(peer).at(index) = value;
    }
    void aclshmem_uint64_p(uint64_t* address, uint64_t value, int32_t peer) {
        done.at(peer).at(address - done[0].data()) = value;
    }
}
template<class T> struct GlobalTensor {
    std::vector<T> values;
    T GetValue(uint64_t index) { return values.at(index); }
};
"""
        + helpers
        + credit_test
        + r"""
int main() {
    credit_test::run();
    for (uint32_t world : {2U, 16U, 64U, 128U}) {
        constexpr uint32_t values = 7 * 6;
        simt::receive.assign(world, std::vector<float>(values, 0));
        simt::writes.assign(world, std::vector<uint32_t>(values, 0));
        simt::done.assign(world, std::vector<uint64_t>(world * 64, 0));
        std::vector<int32_t> ids;
        std::vector<float> weights;
        for (uint32_t id = world * values; id-- > 0;) {
            if (id % 5 == 0) continue;
            ids.push_back(id);
            weights.push_back(id * 0.125f + 0.25f);
        }
        const uint32_t reducers = world <= 8 ? 56 : 40;
        for (uint32_t reducer = 0; reducer < reducers; ++reducer) {
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x) {
                ScatterWeights(weights.data(), ids.data(), simt::receive[0].data(),
                               ids.size() * reducer / reducers, ids.size() * (reducer + 1) / reducers, values);
                ScatterWeights(nullptr, nullptr, simt::receive[0].data(), 0, 0, values);
            }
        }
        for (threadIdx.x = 0; threadIdx.x < 64; ++threadIdx.x) {
            NotifyWeightRanks(simt::done[0].data() + (world - 1) * 64, world);
        }
        for (uint32_t peer = 0; peer < world; ++peer) {
            assert(simt::done[peer][(world - 1) * 64] == 1);
            for (uint32_t slot = 0; slot < values; ++slot) {
                uint32_t id = peer * values + slot;
                assert(simt::receive[peer][slot] == (id % 5 == 0 ? 0 : id * 0.125f + 0.25f));
            }
        }
    }
    GlobalTensor<int32_t> tasks{std::vector<int32_t>(18, -1)};
    tasks.values[0] = 1; tasks.values[6] = tasks.values[12] = 3;
    assert(!TokenNeedsGatewayReduce(tasks, 0, 0));
    tasks.values[9] = 0; // local duplicate still needs reduction, even with zero gather mask
    assert(TokenNeedsGatewayReduce(tasks, 0, 1));
    assert(AlignTaskToTokenBoundary(tasks, 0, 0, 0) == 0);
    assert(AlignTaskToTokenBoundary(tasks, 0, 0, 1) == 1);
    assert(AlignTaskToTokenBoundary(tasks, 0, 0, 2) == 3);
    for (uint32_t world : {2U, 8U, 16U, 32U, 64U, 128U}) {
      for (uint32_t tokens : {1U, 3U, 63U, 256U, 257U, 1025U, 10240U}) {
        auto t = ascend_deepep::MakeCombineTiling(0, world, tokens, 2, 1, 1);
        auto n = ascend_deepep::MakeNotifyTiling(0, world, tokens, 2, world * 4, 256, 7168);
        std::cout << world << ' ' << tokens << ' ' << n.workspace_bytes << ' '
                  << n.backward_group_offsets << ' ' << n.source_chunk_masks << ' '
                  << n.expert_totals << ' ' << n.local_prefix_segments << ' '
                  << t.control_slot_num << ' ' << t.expert_input_offset_bytes << ' '
                  << t.gather_input_offset_bytes << ' ' << t.server_partial_offset_bytes << ' '
                  << t.returned_partial_offset_bytes << ' ' << t.output_offset_bytes << ' '
                  << t.weight_recv_offset_bytes << ' ' << t.weight_local_done_offset_bytes << ' '
                  << t.weight_done_offset_bytes << ' '
                  << ascend_deepep::CombineWorkspaceBytes(t) << '\n';
        assert(sizeof(t) == 168);
        assert(t.weight_topk == 2 && !t.with_weights && t.weight_rows == 0);
        assert(t.weight_recv_offset_bytes % 512 == 0);
        assert(t.weight_local_done_offset_bytes >= t.weight_recv_offset_bytes + uint64_t(tokens) * 2 * 4);
        assert(t.weight_done_offset_bytes == t.weight_local_done_offset_bytes + 64 * 512);
        uint32_t end = 0;
        for (uint32_t c = 0; c < t.chunks_per_group; ++c) {
          auto next = std::min<uint64_t>(tokens, ChunkTokenEnd(t, c));
          assert(next > end && next <= tokens); end = next;
          assert(SoWindowEnd(t, c) >= c && SoWindowEnd(t, c) < t.chunks_per_group);
          assert(CrossCompletionSlot(t, world - 1, c, 1) < t.control_slot_num - 64);
        }
        assert(end == tokens);
        if (world <= 8) continue;
        for (uint32_t rank = 0; rank < world; ++rank) {
          t.rank_id = rank;
          for (uint32_t peer = 0; peer < world; ++peer) {
            if (rank / 8 == peer / 8) continue;
            // Every owner that sends in a window must issue its own final SO.
            for (uint32_t begin = 0; begin < t.chunks_per_group;) {
              uint32_t last[16]; std::fill(last, last + 16, UINT32_MAX);
              const uint32_t finish = SoWindowEnd(t, begin);
              for (uint32_t c = begin; c <= finish; ++c) {
                const auto owner = CrossChunkOwner(t, rank, peer, c);
                assert(owner < 16 && CrossPeerAssigned(t, peer, owner));
                last[owner] = c;
              }
              for (uint32_t c = begin; c <= finish; ++c) {
                assert(CrossSoRequired(t, rank, peer, c) == (last[CrossChunkOwner(t, rank, peer, c)] == c));
              }
              begin = finish + 1;
            }
          }
        }
      }
    }
}
""",
        encoding="utf-8",
    )
    use_wasi = sys.platform == "win32" and importlib.util.find_spec("wasmtime") and not compiler
    executable = tmp_path / (
        "combine_predicates.wasm"
        if use_wasi
        else "combine_predicates.exe"
        if sys.platform == "win32"
        else "combine_predicates"
    )
    result = subprocess.run(
        [
            *command,
            *(["-target", "wasm32-wasi", "-fno-exceptions"] if use_wasi else []),
            "-std=c++17",
            "-O2",
            "-I",
            str(root / "csrc"),
            "-I",
            str(Path(__file__).parent),
            "-I",
            str(tmp_path),
            str(harness),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if use_wasi:
        import wasmtime

        engine = wasmtime.Engine()
        module = wasmtime.Module.from_file(engine, str(executable))
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        wasi = wasmtime.WasiConfig()
        output = tmp_path / "stdout"
        wasi.stdout_file = str(output)
        with wasmtime.Store(engine) as store:
            store.set_wasi(wasi)
            instance = linker.instantiate(store, module)
            try:
                instance.exports(store)["_start"](store)
            except wasmtime.ExitTrap as error:
                assert error.code == 0
        stdout = output.read_text(encoding="utf-8")
    else:
        stdout = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30).stdout
    from deep_ep.buffers._elastic_layout import combine_layout, workspace_layout

    for line in stdout.splitlines():
        world, tokens, *actual = map(int, line.split())
        notify = workspace_layout(world, tokens, 2, world * 4)
        combine = combine_layout(world, tokens, 2, 1, 1)
        expected = [
            notify[name]
            for name in (
                "workspace_bytes",
                "backward_group_offsets",
                "source_chunk_masks",
                "expert_totals",
                "local_prefix_segments",
            )
        ]
        expected += [
            combine[name]
            for name in (
                "control_slot_num",
                "expert_input",
                "gather_input",
                "server_partial",
                "returned_partial",
                "output",
                "weight_recv",
                "weight_local_done",
                "weight_done",
                "workspace_bytes",
            )
        ]
        assert actual == expected, (world, tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _dispatch["_add_launch_args"](parser)
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--without-weights", action="store_true", help="Disable weight return, including profiling")
    parser.add_argument("--route", choices=("all", *_ROUTES, "sq_pressure"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    _dispatch["add_profile_args"](parser, ("combine",))
    args = parser.parse_args()
    _dispatch["validate_profile_args"](parser, args)
    if args.ep_size > 128 or not 1 <= args.tokens <= 10240 or not 1 <= args.topk <= 16 or args.repeat < 1:
        parser.error("EP<=128, tokens1..10240, topk1..16, repeat>=1 required")
    if args.experts < args.ep_size or args.experts > 1024 or args.experts % args.ep_size:
        parser.error("experts must be divisible by EP and <=1024")
    if args.route == "sq_pressure" and (
        args.ep_size != 128 or args.topk != 1 or args.experts != 128 or args.tokens < 6000 or args.repeat < 2
    ):
        parser.error("sq_pressure requires EP128, topk=1, experts=128, tokens>=6000 and repeat>=2")
    args.output.mkdir(parents=True, exist_ok=True)
    _dispatch["_launch"](args, worker)


if __name__ == "__main__":
    main()
