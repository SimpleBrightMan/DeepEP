# Copyright (c) 2026, SimpleBright_Man

"""Execute production Combine metadata batching with deferred MTE2 copies."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


def _production_function(source, name):
    match = re.search(r"__aicore__ inline void " + name + r"\s*\(", source)
    assert match, f"Production function {name} is missing"
    opening = source.index("{", match.start())
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end]


def test_combine_metadata_batch_production_control_flow(tmp_path):
    tests = Path(__file__).parent
    source = (tests.parent / "csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp").read_text(encoding="utf-8")
    functions = [
        _production_function(source, name)
        for name in ("EnqueueWaveCombineMetaInfo", "PreloadWaveCombineMetaInfo", "RunCommSplitCombineSlice")
    ]
    # Compile the real function bodies, not a Python reimplementation of their
    # cursor loops. The mock only replaces device copies, events and row sends.
    fixture = (tests / "combine_metadata_batch_test.cpp").read_text(encoding="utf-8")
    generated = tmp_path / "combine_metadata_batch_production_test.cpp"
    generated.write_text(fixture.replace("// INSERT_PRODUCTION_FUNCTIONS", "\n\n".join(functions)), encoding="utf-8")
    compiler = os.environ.get("CXX")
    command = shlex.split(compiler) if compiler else []
    if not command:
        available = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not available:
            pytest.skip("A host C++17 compiler is required for the Combine metadata regression")
        command = [available]
    executable = generated.with_suffix(".exe" if os.name == "nt" else "")
    subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tests.parent / "csrc/kernels/mega_moe"),
            str(generated),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    completed = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=120)
    assert "metadata batching: delayed copies, mapping, capacity, zero tasks, 2/4/6 slots passed" in completed.stdout
    assert "BS2048/4096/8192 EP16: real planner, all 27 ratios, both group orders passed" in completed.stdout
