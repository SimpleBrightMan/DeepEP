# Copyright (c) 2026, SimpleBright_Man
"""Compile the actual shared tiling header to guard host/device ABI offsets."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_actual_tiling_header_reuses_tail_padding(tmp_path):
    command = shlex.split(os.environ.get("CXX", ""))
    if not command:
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            pytest.skip("A C++17 host compiler is required for the tiling ABI test")
        command = [compiler]
    # This SDK header is unused by the plain layout declarations. Keep the real
    # MegaMoeTilingData definition, stubbing only the unavailable SDK include.
    stub = tmp_path / "kernel_tiling" / "kernel_tiling.h"
    stub.parent.mkdir()
    stub.write_text("#pragma once\n", encoding="utf-8")
    source = tmp_path / "tiling_layout.cpp"
    source.write_text(
        """#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include "kernels/mega_moe/mega_moe_tiling.hpp"
static_assert(std::is_standard_layout<MegaMoeTilingData>::value);
static_assert(sizeof(MegaMoeTilingData) == 224);
static_assert(alignof(MegaMoeTilingData) == 8);
static_assert(offsetof(MegaMoeTilingData, dispatchRanksPerNode) == 216);
static_assert(offsetof(MegaMoeTilingData, prefillCommOptimized) == 220);
static_assert(offsetof(MegaMoeTilingData, prefillCommOptimized) + sizeof(uint32_t)
              == sizeof(MegaMoeTilingData));
int main() {
    MegaMoeTilingData tiling{};
    if (tiling.prefillCommOptimized != 0U) return 1;
    tiling.dispatchRanksPerNode = 16U;
    tiling.prefillCommOptimized = 1U;
    if (tiling.dispatchRanksPerNode != 16U) return 2;
    std::cout << sizeof(tiling) << ' ' << offsetof(MegaMoeTilingData, dispatchRanksPerNode)
              << ' ' << offsetof(MegaMoeTilingData, prefillCommOptimized) << '\\n';
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("tiling_layout.exe" if os.name == "nt" else "tiling_layout")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tmp_path),
            "-I",
            str(csrc),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    completed = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
    assert completed.stdout.strip() == "224 216 220"
