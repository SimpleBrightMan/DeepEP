# Copyright (c) 2026, SimpleBright_Man

import copy
import json

import numpy as np
import pytest

from deep_ep_experimental.megamoe import comm_benchmark as bench
from deep_ep_experimental.megamoe.reference import bf16


def small_config():
    return dict(model=dict(hidden=64, intermediate_hidden=128, num_experts=32, num_topk=8, num_shared_experts=0))


def make_payload_case():
    config, rank, bs = small_config(), 3, 8
    mapping, counts = bench.expected_layout(config, rank, bs)
    data = np.empty((len(mapping), 64), dtype=np.uint8)
    scales = np.empty((len(mapping), 2), dtype=np.uint8)
    for row, (source, token, _slot) in enumerate(mapping):
        data[row], scales[row] = bench.synthetic_input(int(source), [token], 64)
    metadata = np.zeros((len(mapping), 8), dtype=np.int32)
    metadata[:, :3] = mapping
    snapshot = dict(data=data, scales=scales, metadata=metadata)
    expected = bench.expected_output(config, rank, bs)
    return config, rank, bs, snapshot, expected, counts


def test_payload_validation_detects_each_data_plane_failure():
    config, rank, bs, snapshot, expected, counts = make_payload_case()
    assert bench.validate_payload(config, rank, bs, snapshot, expected, counts, expected)["passed"]
    for name, key in (
        ("data", "dispatch_data_equal"),
        ("scales", "dispatch_scales_equal"),
        ("metadata", "metadata_equal"),
    ):
        broken = copy.deepcopy(snapshot)
        broken[name][0, 0] += 1
        result = bench.validate_payload(config, rank, bs, broken, expected, counts, expected)
        assert not result["passed"] and not result[key]
    broken_output = expected.copy()
    broken_output[-1, -1] += 1
    assert not bench.validate_payload(config, rank, bs, snapshot, broken_output, counts, expected)["passed"]
    assert not bench.validate_payload(config, rank, bs, snapshot, expected, counts + 1, expected)["passed"]


def test_synthetic_combine_reverse_routes_matches_direct_scatter():
    config, rank, bs = small_config(), 3, 8
    # Independent direct scatter into [source,token,slot,H], then standard route reduction.
    slots = np.full((16, bs, 8, 64), np.nan, dtype=np.float32)
    for destination in range(16):
        mapping, _ = bench.expected_layout(config, destination, bs)
        payload = bench.synthetic_payload(destination, np.arange(len(mapping)), 64)
        for row, (source, token, topk) in enumerate(mapping):
            slots[source, token, topk] = payload[row]
    assert np.isfinite(slots).all()
    expected = bf16(slots[rank].sum(axis=1) / 8)
    assert np.array_equal(bench.expected_output(config, rank, bs), expected)


def test_balanced_ids_equal_global_rank_first_route_stream():
    for rank in range(16):
        ids = bench.balanced_ids(rank, 2048, 384, 8)
        assert ids.shape == (2048, 8) and ids.dtype == np.int32
        for token, topk in ((0, 0), (2047, 7), (101, 3)):
            route = (rank * 2048 + token) * 8 + topk
            assert ids[token, topk] == (route % 16) * 24 + (route // 16) % 24


def test_synthetic_rows_have_no_token_period_alias_and_are_finite():
    data, scales = bench.synthetic_input(15, np.arange(8192), 64)
    assert len(np.unique(data, axis=0)) == 8192
    assert data.max() < 127 and scales.max() < 255
    payload = bench.synthetic_payload(15, np.arange(65536), 64)
    assert len(np.unique(payload[:, :5], axis=0)) == 65536
    assert np.array_equal(payload, bf16(payload))


def test_initial_simulator_rejects_unsupported_shape_before_importing_torch(tmp_path):
    config = dict(schema_version=1, model=small_config()["model"], bs_values=[2048], warmup=0, iters=1, seed=42)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        bench.parse_args(["--config", str(path), "--output-dir", str(tmp_path / "output")])


@pytest.mark.parametrize("options,expected_mode", [([], "dma"), (["--dispatch-metadata-mode", "scalar"], "scalar")])
def test_benchmark_metadata_mode_defaults_to_dma_and_accepts_scalar(tmp_path, options, expected_mode):
    config = dict(
        schema_version=1,
        model=dict(hidden=3584, intermediate_hidden=1280, num_experts=384, num_topk=8, num_shared_experts=0),
        bs_values=[2048],
        warmup=0,
        iters=1,
        seed=42,
    )
    path = tmp_path / "model.json"
    path.write_text(json.dumps(config))
    args, _ = bench.parse_args(["--config", str(path), "--output-dir", str(tmp_path / "output"), *options])
    assert args.dispatch_metadata_mode == expected_mode
