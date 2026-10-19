"""Routing identity, expert boundaries and delayed-return lifecycle checks."""

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_routing import (
    CompletionLedger,
    cv_fragments,
    plan_routes,
    prepare_routes,
    remote_peer_for_worker,
    simulate_roundtrip,
)


def balanced(bs, rank, world=16, experts=384, topk=8):
    linear = np.arange(rank * bs * topk, (rank + 1) * bs * topk)
    return ((linear % world) * (experts // world) + (linear // world) % (experts // world)).reshape(bs, topk)


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
@pytest.mark.parametrize("cv_rows", [128, 256, 1024])
def test_prefill_full_route_coverage(bs, cv_rows):
    ids = [balanced(bs, rank) for rank in range(16)]
    routes = [plan_routes(value, 384, 16) for value in ids]
    for destination in range(16):
        fragments = cv_fragments(routes, destination, cv_rows)
        coverage = [np.zeros(int(r.peer_counts[destination]), dtype=np.uint8) for r in routes]
        for fragment in fragments:
            route = routes[fragment.source_rank]
            coverage[fragment.source_rank][fragment.peer_row : fragment.peer_row + fragment.rows] += 1
            flat = ids[fragment.source_rank].reshape(-1)
            original = route.order[fragment.source_expanded_row : fragment.source_expanded_row + fragment.rows]
            assert np.all(flat[original] == fragment.expert)
            assert fragment.cv_row + fragment.rows <= cv_rows
        assert all(np.all(value == 1) for value in coverage)
    for route in routes:
        np.testing.assert_array_equal(route.order[route.reduce_index.reshape(-1)], np.arange(bs * 8))


@pytest.mark.parametrize("pattern", ["random", "skewed", "duplicate", "empty_peers"])
@pytest.mark.parametrize("cv_rows", [1, 3, 16])
def test_return_put_and_weighted_reduce_match_direct_expert_evaluation(pattern, cv_rows):
    rng = np.random.default_rng(24)
    world, experts, bs, topk, hidden = 4, 12, 7, 3, 5
    xs = [rng.normal(size=(bs, hidden)).astype(np.float32) for _ in range(world)]
    ids, weights = [], []
    for rank in range(world):
        if pattern == "random":
            route = np.array([rng.choice(experts, topk, replace=False) for _ in range(bs)])
        elif pattern == "skewed":
            route = np.tile([0, 1, 2], (bs, 1))
        elif pattern == "duplicate":
            route = np.tile([rank, rank, rank], (bs, 1))
        else:
            route = np.tile([0, 3, 9], (bs, 1))
        ids.append(route)
        weights.append(rng.normal(size=(bs, topk)).astype(np.float32))

    def expert(expert_id, rows):
        return rows * (expert_id + 1) + expert_id / 2

    actual = simulate_roundtrip(xs, ids, weights, experts, cv_rows, expert)
    for rank in range(world):
        expected = np.zeros_like(xs[rank])
        for token in range(bs):
            for slot in range(topk):
                expected[token] += expert(ids[rank][token, slot], xs[rank][token]) * weights[rank][token, slot]
        np.testing.assert_allclose(actual[rank], expected, rtol=2e-6, atol=2e-6)


def test_stable_expansion_preserves_repeated_expert_instances():
    route = plan_routes(np.array([[2, 1, 2], [0, 1, 2]]), 4, 2)
    assert route.order.tolist() == [3, 1, 4, 0, 2, 5]
    assert route.reduce_index.tolist() == [[3, 1, 4], [0, 2, 5]]
    assert route.peer_counts.tolist() == [3, 3]


@pytest.mark.parametrize("rank", range(16))
def test_fifteen_workers_have_exclusive_remote_peers(rank):
    peers = [remote_peer_for_worker(rank, worker) for worker in range(15)]
    assert sorted(peers) == [peer for peer in range(16) if peer != rank]


def test_deferred_quiet_and_final_barrier_are_required_before_reduction():
    ledger = CompletionLedger(2)
    with pytest.raises(RuntimeError, match="Dispatch"):
        ledger.submit_return(0, 1, 0)
    ledger.finish_dispatch(0)
    ledger.finish_dispatch(1)
    ledger.submit_return(0, 1, 0)
    ledger.submit_return(0, 1, 1)
    with pytest.raises(RuntimeError, match="complete"):
        ledger.arrive_final_barrier(0)
    ledger.quiet(0)
    ledger.arrive_final_barrier(0)
    assert not ledger.can_reduce()
    ledger.arrive_final_barrier(1)
    assert ledger.can_reduce()
    with pytest.raises(RuntimeError, match="barrier"):
        ledger.submit_return(0, 1, 2)


def test_device_route_algorithm_matches_independent_numpy_sort():
    torch = pytest.importorskip("torch")
    ids = np.random.default_rng(19).integers(0, 24, size=(17, 4), dtype=np.int32)
    expected = plan_routes(ids, 24, 4)
    actual = prepare_routes(torch.from_numpy(ids), 24)
    for field in ("order", "reduce_index", "expert_counts", "expert_offsets"):
        np.testing.assert_array_equal(getattr(actual, field).numpy(), getattr(expected, field))


@pytest.mark.parametrize("ids", [np.array([[4]]), np.array([[-1]]), np.array([[1.5]]), np.array([], dtype=np.int32)])
def test_bad_routes_are_rejected(ids):
    with pytest.raises(ValueError):
        plan_routes(ids, 4, 2)
