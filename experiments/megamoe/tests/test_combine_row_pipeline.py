# Copyright (c) 2026, SimpleBright_Man

"""Compile production Combine row-ring functions with independent DMA queues."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


def _function(source, name, *, template=False):
    match = re.search(r"__aicore__ inline [^\n]+\b" + name + r"\s*\(", source)
    assert match, f"Production function {name} is missing"
    start = source.rfind("template <", 0, match.start()) if template else match.start()
    opening = source.index("{", match.end())
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_actual_combine_row_pipeline_and_ub_limits(tmp_path):
    tests = Path(__file__).parent
    source = (tests.parent / "csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp").read_text(encoding="utf-8")
    # These declarations and function bodies come from production. The fixture
    # only substitutes the hardware engines and the remote address resolver.
    declaration_begin = source.index("struct WaveCombineBufferConfig {")
    declaration_end = source.index("// 非 layered", declaration_begin)
    declarations = source[declaration_begin:declaration_end]
    meta_address = re.search(r"constexpr uint32_t META_INFO_TENSOR_ADDR\s*=.*?;", source).group()
    functions = [
        _function(source, "InitWaveCombineBuffers", template=True),
        _function(source, "InitTokenCombineSliceBuffers"),
        _function(source, "DrainCombineRowBuffers"),
        _function(source, "SendWaveCombineToken", template=True),
        _function(source, "CombineWaveTokenRange", template=True),
    ]
    fixture = (tests / "combine_row_pipeline_test.cpp").read_text(encoding="utf-8")
    generated = tmp_path / "combine_row_pipeline_production.cpp"
    generated.write_text(
        fixture.replace("// INSERT_PRODUCTION_DECLARATIONS", meta_address + "\n" + declarations).replace(
            "// INSERT_PRODUCTION_FUNCTIONS", "\n\n".join(functions)
        ),
        encoding="utf-8",
    )
    compiler = os.environ.get("CXX")
    command = shlex.split(compiler) if compiler else []
    if not command:
        available = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not available:
            pytest.skip("A host C++17 compiler is required for Combine slot lifecycle tests")
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
    assert "Combine 2/4/6/8 slots: deferred DMA, events 0..7, peer segments and tail drains passed" in completed.stdout
    assert "UB: H3584/6144/7680 fit, H8192 eight-slot rejection, original Decode6/Prefill2 passed" in completed.stdout
