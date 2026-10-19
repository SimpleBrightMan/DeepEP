# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Compile the actual device-independent planner against a token-level oracle."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def compile_and_run(source, tmp_path):
    compiler = os.environ.get("CXX")
    command = shlex.split(compiler) if compiler else []
    if not command:
        available = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not available:
            pytest.skip("A host C++17 compiler is required for the communication split oracle")
        command = [available]
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / (Path(source).stem + (".exe" if os.name == "nt" else ""))
    subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "experiments/megamoe/csrc/kernels/mega_moe"),
            str(Path(__file__).with_name(source)),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    return subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=120)


def test_comm_split_route_ready_inverse_oracle(tmp_path):
    completed = compile_and_run("comm_split_test.cpp", tmp_path)
    assert "7680 independent route/ready/inverse cases, all 27 splits passed" in completed.stdout
    assert "paired-xor: 7680 varied cases" in completed.stdout
    assert "BS2048/4096/8192 EP16 slices, exact order/owners/ready/inverse passed" in completed.stdout


def test_actual_timer_writer_audit_isolation_and_nested_spans(tmp_path):
    completed = compile_and_run("comm_timer_device_test.cpp", tmp_path)
    assert "exact wire, independent spans, zero rows, audit isolation passed" in completed.stdout
