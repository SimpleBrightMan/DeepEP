# Copyright (c) 2026, SimpleBright_Man
"""CPU checks of the actual full-Prefill policy and communication planner.

These exercise allocation boundaries, metadata identities and token/ready
conservation. They do not replace an on-device numerical or pipeline test.
"""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest


def _compile_and_run(source, tmp_path):
    command = shlex.split(os.environ.get("CXX", ""))
    if not command:
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            pytest.skip("A C++17 host compiler is required for the full-Prefill contracts")
        command = [compiler]
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    executable = tmp_path / (source.stem + (".exe" if os.name == "nt" else ""))
    subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(csrc),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    return subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=120)


def test_actual_prefill_policy_metadata_and_wave_mapping(tmp_path):
    completed = _compile_and_run(Path(__file__).with_suffix(".cpp"), tmp_path)
    assert "profile allocation boundaries and metadata lifetime-independent identities passed" in completed.stdout
    assert "BS2048/4096/8192: EP16 XOR 4/24 ownership, M256 ready counts and inverse rows passed" in completed.stdout


def test_native_profile_requires_explicit_single_node_topology():
    source = (Path(__file__).resolve().parents[1] / "csrc/forward.cpp").read_text(encoding="utf-8")
    assignment = source.split("tiling_.prefillCommOptimized =", 1)[1].split(";", 1)[0]
    # The pure allocation helper cannot know physical rank placement. The Plan
    # must separately restrict its four local groups to one declared EP16 node.
    assert re.match(r"\s*ranks_per_node\s*==\s*16U?\s*&&\s*UsePrefillProfile\(", assignment)


def test_actual_combine_priority_probe_preserves_compute_team(tmp_path):
    tests = Path(__file__).parent
    production = (tests.parent / "csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp").read_text(encoding="utf-8")
    start = production.index("::TryRunReadyCombine(const AivTaskCursor& cursor)")
    opening = production.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (production[end] == "{") - (production[end] == "}")
        end += 1
    fixture = tests / "test_prefill_integration_scheduler.cpp"
    generated = tmp_path / "prefill_scheduler_actual.cpp"
    generated.write_text(
        fixture.read_text(encoding="utf-8").replace("// INSERT_ACTUAL_READY_PROBE", production[opening:end]),
        encoding="utf-8",
    )
    completed = _compile_and_run(generated, tmp_path)
    assert "nonblocking ready probe, original GMM2 producer team and full-path fallback passed" in completed.stdout


def test_actual_combine_split_keeps_one_row_ring_until_return(tmp_path):
    tests = Path(__file__).parent
    source = (tests.parent / "csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp").read_text(encoding="utf-8")
    start = source.index("__aicore__ inline void RunCommSplitCombineSlice(")
    opening = source.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    generated = tmp_path / "prefill_combine_actual.cpp"
    generated.write_text(
        (tests / "test_prefill_integration_combine.cpp")
        .read_text(encoding="utf-8")
        .replace("// INSERT_ACTUAL_COMBINE", source[start:end]),
        encoding="utf-8",
    )
    completed = _compile_and_run(generated, tmp_path)
    assert (
        "eight slots persist across metadata/peer chunks; one final drain also with timer off passed"
        in completed.stdout
    )
