"""Routing identity, expert boundaries and delayed-return lifecycle checks."""

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_routing import (
    CompletionLedger,
    cv_fragments,
    expert_m_groups,
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


@pytest.mark.parametrize("bs, expected_groups", [(2048, 72), (4096, 144), (8192, 264)])
@pytest.mark.parametrize("destination", [0, 7, 15])
@pytest.mark.parametrize("cv_rows", [128, 256, 1024])
def test_expert_m256_partition_is_independent_of_communication_cv(bs, expected_groups, destination, cv_rows):
    routes = [plan_routes(balanced(bs, rank), 384, 16) for rank in range(16)]
    groups = expert_m_groups(routes, destination, cv_rows)
    assert len(groups) == expected_groups
    assert [g.expert for g in groups] == sorted(g.expert for g in groups)
    written = np.zeros(bs * 8, dtype=np.uint8)
    compute_rows = 0
    for group in groups:
        assert group.row_begin == compute_rows
        assert 0 < group.rows <= 256
        compute_rows += group.rows
        local_coverage = np.zeros(group.rows, dtype=np.uint8)
        for segment in group.segments:
            route = routes[segment.source_rank]
            begin, end = map(int, route.expert_offsets[group.expert : group.expert + 2])
            assert begin <= segment.return_row < segment.return_row + segment.rows <= end
            peer_row = segment.return_row - int(route.peer_offsets[destination])
            assert peer_row // cv_rows == (peer_row + segment.rows - 1) // cv_rows == segment.cv_id
            local_coverage[segment.tile_row : segment.tile_row + segment.rows] += 1
            written[segment.receive_row : segment.receive_row + segment.rows] += 1
        assert np.all(local_coverage == 1)
    assert np.all(written == 1)
    assert compute_rows == bs * 8
    for expert in range(destination * 24, (destination + 1) * 24):
        expert_groups = [g for g in groups if g.expert == expert]
        assert all(g.rows == 256 for g in expert_groups[:-1])
        assert sum(g.rows for g in expert_groups) == sum(int(r.expert_counts[expert]) for r in routes)


def test_first_m256_needs_only_its_actual_sources_and_preserves_self_mapping():
    routes = [plan_routes(balanced(2048, rank), 384, 16) for rank in range(16)]
    first = expert_m_groups(routes, 0, 256)[0]
    assert [(s.source_rank, s.rows) for s in first.segments] == [(0, 43), (1, 43), (2, 42), (3, 43), (4, 43), (5, 42)]
    assert {s.cv_id for s in first.segments} == {0}
    assert first.rows == 256
    self_segment = first.segments[0]
    assert self_segment.return_row == int(routes[0].expert_offsets[0])


def test_expert_m256_roundtrip_really_combines_sources_before_expert_call():
    world, bs, topk = 4, 300, 2
    xs = [np.full((bs, 2), rank, dtype=np.float32) for rank in range(world)]
    ids = [np.tile([0, 1], (bs, 1)) for _ in range(world)]
    weights = [np.ones((bs, topk), dtype=np.float32) for _ in range(world)]
    calls = []

    def expert(expert_id, rows):
        calls.append((expert_id, len(rows), len(np.unique(rows[:, 0]))))
        return rows + expert_id

    result = simulate_roundtrip(xs, ids, weights, 8, 128, expert)
    assert [size for _, size, _ in calls] == [256, 256, 256, 256, 176] * 2
    assert any(sources > 1 for _, _, sources in calls)
    for rank, actual in enumerate(result):
        np.testing.assert_array_equal(actual, np.full_like(xs[rank], 2 * rank + 1))


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
