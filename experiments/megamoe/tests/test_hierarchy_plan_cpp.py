# Copyright (c) 2026, SimpleBright_Man
"""Compile and run the independent routing/reverse-reduction oracle without an NPU."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_hierarchy_plan_cpp(tmp_path):
    configured = os.environ.get("CXX")
    compiler = shlex.split(configured) if configured else []
    if not compiler:
        executable = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if executable is None:
            pytest.skip("Hierarchy planner oracle requires a C++17 host compiler or CXX")
        compiler = [executable]
    directory = Path(__file__).resolve().parent
    binary = tmp_path / ("hierarchy-plan.exe" if os.name == "nt" else "hierarchy-plan")
    compile_result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(directory.parent / "csrc"),
            str(directory / "hierarchy_plan_test.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "31 route/forward/reverse oracle cases passed" in result.stdout
