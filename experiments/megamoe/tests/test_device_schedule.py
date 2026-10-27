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


def test_passive_timer_only_needs_clock_and_private_stores(tmp_path):
    # 桩不提供 barrier/flag API；若计时器意外增加同步，编译即失败。
    compile_and_run(Path(__file__).with_name("passive_timer_cpu.cpp"), tmp_path)


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


def cpp_body(source, marker):
    begin = source.index("{", source.index(marker))
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_production_gmm_wave_order_matches_frozen_scheduler(tmp_path):
    kernel = ROOT / "experiments/megamoe/csrc/kernels/mega_moe"
    source = (kernel / "mega_moe_wave_a8w4.hpp").read_text(encoding="utf-8")
    signatures = {
        "RunNextGmm1Slice":
            "void RunNextGmm1Slice(ExpertWaveRange& wave, uint32_t& usedMGroups, ExpertContext& gmm1ExpertContext, "
            "GMMAddrInfo& gmm1AddrInfo, uint32_t& startBlockIdx, int32_t& readySequence, uint64_t waveId)",
        "RunGmm1Wave":
            "ExpertWaveRange RunGmm1Wave(ExpertTokenPosition& gmm1Cursor, ExpertContext& gmm1ExpertContext, "
            "GMMAddrInfo& gmm1AddrInfo, uint32_t& startBlockIdx, int32_t& readySequence, uint64_t waveId)",
        "RunGmm2CombineForWaveRange":
            "void RunGmm2CombineForWaveRange(const ExpertWaveRange& wave, ExpertContext& gmm2ExpertContext, "
            "GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx)",
        "ProcessGmmWaves": "void ProcessGmmWaves(int32_t& gmm1TileReadySequence)",
    }
    methods = [signature + cpp_body(source, f"::{name}(") for name, signature in signatures.items()]
    (tmp_path / "gmm_wave_methods.inc").write_text("\n".join(methods), encoding="utf-8")

    types = (kernel / "common/mega_moe_types.hpp").read_text(encoding="utf-8")
    utils = (kernel / "common/mega_moe_utils.hpp").read_text(encoding="utf-8")
    context = "struct ExpertContext" + cpp_body(types, "struct ExpertContext") + ";\n"
    context += "ExpertContext CreateExpertContext(const MoeStageCommonConfig& context)" + cpp_body(
        utils, "CreateExpertContext("
    )
    context += "\nvoid UpdateExpertContext(ExpertContext& expertContext, uint32_t expertIdx, uint64_t expertTokenCount)"
    context += cpp_body(utils, "UpdateExpertContext(")
    (tmp_path / "expert_context_methods.inc").write_text(context, encoding="utf-8")
    compile_and_run(Path(__file__).with_name("gmm_wave_execution_host.cpp"), tmp_path)


def test_gm_uses_production_wave_role_entry(tmp_path):
    source = (ROOT / "experiments/megamoe/csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp").read_text(encoding="utf-8")
    (tmp_path / "wave_entry.inc").write_text(
        "void ProcessMoeExpertStages(int32_t& gmm1TileReadySequence)" + cpp_body(source, "::ProcessMoeExpertStages("),
        encoding="utf-8",
    )
    compile_and_run(Path(__file__).with_name("wave_entry_host.cpp"), tmp_path)
