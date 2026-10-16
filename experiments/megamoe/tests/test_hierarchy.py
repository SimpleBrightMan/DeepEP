# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Control protocol, independent network counts, and full-span reporting checks."""

import numpy as np
import pytest

from deep_ep_experimental.megamoe import hierarchy as h
from deep_ep_experimental.megamoe.benchmark import inputs
from deep_ep_experimental.megamoe.hierarchy_benchmark import arguments


def routes(batch=5, topk=8, experts=384, routing="balanced"):
    config = {"model": {"hidden": 32, "num_experts": experts, "num_topk": topk}, "seed": 42}
    return np.stack([inputs(config, rank, batch, routing, world=16)[1] for rank in range(16)])


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
def test_payload_matches_explicit_token_consumer_sets(routing):
    ids, hidden = routes(routing=routing), 3584
    stride = hidden + 128
    actual = [
        dict(
            dispatch_cross_bytes=0,
            dispatch_intra_bytes=0,
            dispatch_local_bytes=0,
            combine_intra_bytes=0,
            combine_cross_bytes=0,
            local_unique_tokens=0,
        )
        for _ in range(16)
    ]
    for source in range(16):
        for experts in ids[source]:
            consumers = set(int(e) // 24 for e in experts)
            groups = {consumer // 4 for consumer in consumers}
            for group in groups:
                proxy = group * 4 + source % 4
                if group != source // 4:
                    actual[proxy]["dispatch_cross_bytes"] += stride
                    actual[source]["combine_cross_bytes"] += hidden * 4
                for consumer in consumers:
                    if consumer // 4 != group:
                        continue
                    upstream = source if group == source // 4 else proxy
                    key = "dispatch_local_bytes" if upstream == consumer else "dispatch_intra_bytes"
                    actual[consumer][key] += stride
                    actual[consumer]["local_unique_tokens"] += 1
                    if consumer != proxy:
                        actual[proxy]["combine_intra_bytes"] += hidden * 4
    for rank in range(16):
        assert h.expected_payload_bytes(ids, 384, hidden, rank) == actual[rank]


def test_balanced_closed_form_all_ranks():
    batch, hidden = 12, 3584
    ids = routes(batch=batch)
    for rank in range(16):
        result = h.expected_payload_bytes(ids, 384, hidden, rank)
        assert result == dict(
            dispatch_cross_bytes=3 * batch // 2 * 3712,
            dispatch_intra_bytes=6 * batch * 3712,
            dispatch_local_bytes=2 * batch * 3712,
            combine_intra_bytes=6 * batch * hidden * 4,
            combine_cross_bytes=3 * batch // 2 * hidden * 4,
            local_unique_tokens=8 * batch,
        )


def test_local_expert_duplicates_are_deduplicated_but_not_duplicate_ids():
    ids = np.broadcast_to(np.array([0, 1, 2, 3], dtype=np.int32), (16, 3, 4)).copy()
    assert h.expected_payload_bytes(ids, 384, 3584, 0)["local_unique_tokens"] == 48
    assert h.expected_payload_bytes(ids, 384, 3584, 1)["local_unique_tokens"] == 0
    ids[0, 0, 1] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        h.validate_routes(ids, 384)


class ChunkGroup:
    def __init__(self, ids):
        self.ids = ids
        self.cursor = 0
        self.chunks = []

    def size(self):
        return 16

    def all_gather(self, value, label):
        if label == "hierarchy route shapes":
            return [list(self.ids.shape[1:])] * 16
        end = self.cursor + len(value)
        assert value == self.ids[0].reshape(-1)[self.cursor : end].tolist()
        packets = [rank.reshape(-1)[self.cursor : end].tolist() for rank in self.ids]
        self.chunks.append(len(value))
        self.cursor = end
        return packets


def test_chunked_control_exchange_retains_source_and_token_order():
    ids = routes(routing="random")
    group = ChunkGroup(ids)
    actual = h.gather_route_ids(group, ids[0], 384, chunk_elements=7)
    np.testing.assert_array_equal(actual, ids)
    assert group.chunks == [7, 7, 7, 7, 7, 5]


def test_control_exchange_rejects_divergent_shapes():
    group = ChunkGroup(routes())
    group.all_gather = lambda *args: [[5, 8]] * 15 + [[4, 8]]
    with pytest.raises(ValueError, match="identical"):
        h.gather_route_ids(group, group.ids[0], 384)


def test_total_uses_round_slowest_rank_not_rank_average():
    per_rank = [[dict(acl_span_us=10, wall_us=11), dict(acl_span_us=10, wall_us=11)] for _ in range(16)]
    per_rank[0][0] = dict(acl_span_us=100, wall_us=110)
    per_rank[1][1] = dict(acl_span_us=200, wall_us=210)
    report = h.summarize_measurements(per_rank)
    assert report["acl_span_us"]["rank_max_mean"] == 150
    assert report["wall_us"]["rank_max_mean"] == 160
    assert max(report["acl_span_us"]["per_rank_mean"]) == 105
    with pytest.raises(ValueError, match="16 ranks"):
        h.summarize_measurements(per_rank[:15])
    per_rank[0][0]["wall_us"] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        h.summarize_measurements(per_rank)


def test_stage_rates_use_bytes_and_ms_and_validate_native_counts():
    metrics = {}
    for byte_field, time_field in h.STAGE_BANDWIDTH.values():
        metrics[byte_field], metrics[time_field] = 100_000_000, 2
    assert set(h.stage_rates(metrics).values()) == {50.0}
    h.validate_payload_metrics(metrics, {"dispatch_cross_bytes": 100_000_000, "local_unique_tokens": 1})
    with pytest.raises(ValueError, match="mismatch"):
        h.validate_payload_metrics(metrics, {"dispatch_cross_bytes": 1})
    metrics["last_dispatch_cross_ms"] = 0
    with pytest.raises(ValueError, match="Invalid stage"):
        h.stage_rates(metrics)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--bs", "256"),
        ("--bs", "4097"),
        ("--hidden", "3712"),
        ("--experts", "383"),
        ("--topk", "0"),
        ("--iters", "0"),
        ("--relative-rmse", "nan"),
    ],
)
def test_runner_rejects_unsupported_case_before_npu_import(option, value):
    with pytest.raises(SystemExit):
        arguments(["--bs", "2048", "--output", "result", option, value])


def test_runner_defaults_match_requested_case():
    args = arguments(["--bs", "4096", "--output", "result"])
    assert (args.hidden, args.intermediate, args.experts, args.topk) == (3584, 1280, 384, 8)
