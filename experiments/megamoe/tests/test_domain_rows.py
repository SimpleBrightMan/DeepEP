# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Compile the production row planner and compare it with an independent routing oracle."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_domain_rows_native_oracle(tmp_path):
    explicit = os.environ.get("CXX")
    if explicit:
        compiler = shlex.split(explicit)
    else:
        found = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
        if found is None:
            pytest.skip("Domain row planner test requires a host C++17 compiler")
        compiler = [found]
    binary = tmp_path / ("domain_rows.exe" if os.name == "nt" else "domain_rows")
    command = [
        *compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "experiments/megamoe/csrc/kernels/mega_moe"),
        str(Path(__file__).with_name("domain_rows_test.cpp")),
        "-o",
        str(binary),
    ]
    compile_result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "independent-oracle cases passed" in result.stdout
