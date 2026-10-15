# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Compile the production CPU planner and compare it with an independent route oracle."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_dispatch_plan_native_oracle(tmp_path):
    source = Path(__file__).with_name("dispatch_plan_test.cpp")
    include = source.parents[1] / "csrc"
    explicit = os.environ.get("CXX")
    if explicit:
        compiler = shlex.split(explicit)
    else:
        found = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
        if found is None:
            pytest.skip("Dispatch planner test requires a host C++17 compiler")
        compiler = [found]
    binary = tmp_path / ("dispatch_plan.exe" if os.name == "nt" else "dispatch_plan")
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(include),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "independent-oracle cases passed" in run.stdout
