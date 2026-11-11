# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026
from pathlib import Path
import shutil
import subprocess

import pytest


def test_shared_split_coverage_ownership_and_completion_counts(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "shared_schedule_test"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root / "experiments/megamoe/csrc"),
                    str(Path(__file__).with_name("shared_schedule_test.cpp")), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=15)


def test_disjoint_coarse_activation_intervals_decode_on_aiv0():
    import numpy as np
    from deep_ep_experimental.megamoe import urma_trace as trace

    capacity = 3
    stride = trace.HEADER_WORDS + capacity * trace.RECORD_WORDS
    raw = np.zeros((trace.ENGINE_COUNT, stride), dtype=np.uint64)
    for engine in range(trace.ENGINE_COUNT):
        raw[engine, :5] = [trace.TRACE_SCHEMA, capacity, 0, 0, engine]
    raw[1, 2] = 3  # AIV0: head activation, routed dequant, tail activation.
    for index, (start, end, event) in enumerate(((100, 110, 50), (115, 140, 47), (145, 180, 50))):
        offset = trace.HEADER_WORDS + index * trace.RECORD_WORDS
        raw[1, offset:offset + 3] = [start, end, event]
        raw[1, offset + 3:offset + 13] = trace.UNKNOWN
    decoded = trace.decode_trace(raw.reshape(-1), rank=0, clock_mhz=1000.0)
    events = [event for event in decoded["traceEvents"] if event["ph"] == "X"]
    assert [event["args"]["event_id"] for event in events] == [50, 47, 50]
    assert all(event["tid"] == 1 for event in events)
    assert decoded["metadata"]["instrumentation"] == "coarse"
