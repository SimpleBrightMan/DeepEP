"""CPU routing contracts, including exact agreement with the supplied algorithm."""

import random

import numpy as np
import pytest

from deep_ep_experimental.megamoe.benchmark import inputs, load_config
from deep_ep_experimental.megamoe.random_balanced import random_balanced_indices
from deep_ep_experimental.megamoe.urma_benchmark import parse_args


def reference_routes(bs, ep, experts, topk):
    remaining = [bs * topk] * ep
    occurrences = [0] * ep
    rng = random.Random(20261014 + bs * 1000 + 2)
    routes = []
    for token in range(ep * bs):
        left = ep * bs - token
        chosen = [r for r, n in enumerate(remaining) if n == left]
        while len(chosen) < topk:
            candidates = [r for r, n in enumerate(remaining) if n > 0 and r not in chosen]
            chosen.append(rng.choices(candidates, weights=[remaining[r] for r in candidates])[0])
        rng.shuffle(chosen)
        routes.append([r * (experts // ep) + occurrences[r] % (experts // ep) for r in chosen])
        for r in chosen:
            remaining[r] -= 1
            occurrences[r] += 1
    assert not any(remaining)
    return np.array(routes, dtype=np.int32)


@pytest.mark.parametrize("ep", [8, 16, 64])
def test_matches_supplied_default_script_exactly(ep):
    bs = 17
    actual = np.concatenate([random_balanced_indices(bs, ep, 384, 8, rank) for rank in range(ep)])
    np.testing.assert_array_equal(actual, reference_routes(bs, ep, 384, 8))


@pytest.mark.parametrize("ep,experts,topk", [(4, 384, 8), (8, 384, 8), (16, 384, 8), (3, 12, 8), (1, 8, 8)])
@pytest.mark.parametrize("bs", [1, 7, 31, 1024])
def test_exact_rank_quota_expert_balance_and_no_duplicate_experts(ep, experts, topk, bs):
    routes = np.concatenate([random_balanced_indices(bs, ep, experts, topk, rank) for rank in range(ep)])
    assert routes.dtype == np.int32 and routes.shape == (ep * bs, topk)
    assert routes.min() >= 0 and routes.max() < experts
    assert all(len(set(row)) == topk for row in routes)
    destinations = routes // (experts // ep)
    np.testing.assert_array_equal(np.bincount(destinations.ravel(), minlength=ep), [bs * topk] * ep)
    expert_counts = np.bincount(routes.ravel(), minlength=experts)
    assert expert_counts.max() - expert_counts.min() <= 1
    if ep >= topk:
        assert all(len(set(row)) == topk for row in destinations)
    elif topk % ep == 0:
        for row in destinations:
            np.testing.assert_array_equal(np.bincount(row, minlength=ep), [topk // ep] * ep)


def test_inputs_are_repeatable_and_caller_cannot_corrupt_global_cache():
    config = load_config("urma")
    config["model"]["hidden"] = 32
    first = inputs(config, 3, 31, "random-balanced", world=16)
    again = inputs(config, 3, 31, "random-balanced", world=16)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    first[1][:] = -1
    np.testing.assert_array_equal(inputs(config, 3, 31, "random-balanced", world=16)[1], again[1])
    ordinary = inputs(config, 3, 31, "balanced", world=16)
    np.testing.assert_array_equal(ordinary[0], again[0])
    np.testing.assert_array_equal(ordinary[2], again[2])
    assert not np.array_equal(ordinary[1], again[1])
    args = parse_args(["--routing", "random-balanced", "--bs-values", "1024", "--output-dir", "/tmp/test"])
    assert args.routing == "random-balanced"


@pytest.mark.parametrize("args", [(0, 4, 384, 8, 0), (1, 0, 384, 8, 0), (1, 5, 384, 8, 0), (1, 4, 384, 385, 0), (1, 4, 384, 8, 4)])
def test_invalid_dimensions_fail_before_generation(args):
    with pytest.raises(ValueError):
        random_balanced_indices(*args)
