"""Compile and execute production policy/Wave helpers without CANN or an NPU."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


def compile_and_run(source, tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Set CXX to a C++17 compiler to run device-independent scheduling checks")
    executable = tmp_path / ("device_schedule.exe" if os.name == "nt" else "device_schedule")
    command = shlex.split(compiler) if os.environ.get("CXX") else [compiler]
    command += [
        "-std=c++17", "-O0", "-UNDEBUG",
        "-I", str(ROOT / "experiments/megamoe/csrc/kernels/mega_moe"),
        "-I", str(tmp_path), str(source), "-o", str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=240)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)


def test_device_policy_and_wave_arithmetic(tmp_path):
    compile_and_run(Path(__file__).with_name("device_schedule_host.cpp"), tmp_path)


def test_production_pipeline_stage_order(tmp_path):
    source = (ROOT / "experiments/megamoe/csrc/kernels/mega_moe/mega_moe_arch35.hpp").read_text(encoding="utf-8")
    methods = []
    for name, signature in (
        ("WaitForLocalInput", "void WaitForLocalInput()"),
        ("ProcessInputRouteHandshake", "void ProcessInputRouteHandshake()"),
        ("ProcessOutputStage", "void ProcessOutputStage()"),
        ("ProcessWave", "template <class Derived> void ProcessWave(Derived& derived)"),
    ):
        # Execute current C++ bodies, not a second Python implementation of the new flow.
        begin = source.index("{", source.index(f"::{name}("))
        depth, end = 1, begin + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        methods.append(signature + source[begin:end])
    (tmp_path / "process_wave_methods.inc").write_text("\n".join(methods), encoding="utf-8")
    compile_and_run(Path(__file__).with_name("pipeline_order_host.cpp"), tmp_path)
