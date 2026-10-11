"""Execute the actual cursor-transfer bodies with host-only field stubs.

This checks software state ownership, not hardware event completion. Device
accuracy, empty/skew routing and repeated launches remain required separately.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "experiments/megamoe/csrc/kernels/mega_moe/catlass_gmm"


def move_body(filename):
    source = (KERNEL / filename).read_text(encoding="utf-8")
    start = source.index("    void MovePipelineTo(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_actual_cursor_moves_preserve_slots_and_single_owner(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable; run on build host")
    cube_fields = ("l1AListId", "l1BListId", "l0AListId", "l0BListId", "l0CListId",
                   "l1MxScaleAListId", "l1MxScaleBListId")
    vector_fields = ("l1BListId", "ubInputListId", "ubOutputListId")
    definitions = []
    exercises = []
    for name, filename, fields in (
        ("MegaMoeBlockMmadA8W4", "block_mmad_a8w4.hpp", cube_fields),
        ("MegaMoeBlockPrologueA8W4", "block_prologue_a8w4.hpp", vector_fields),
    ):
        definitions.append("struct " + name + " {\n" +
            "static constexpr bool ENABLE_UNIT_FLAG=true, ENABLE_L1_RESIDENT=false, HAS_BIAS=false;\n" +
            "static constexpr unsigned L0C_STAGES=1;\n" +
            "bool active_=false;\n" +
            "\n".join("unsigned " + f + "=0;" for f in fields) + "\n" + move_body(filename) + "\n};")
        checks = "\n".join("assert(dst." + f + " == expected." + f + ");" for f in fields)
        initial = "\n".join("src." + f + " = (trial * " + str(2 * i + 3) + ") % " +
                             ("3" if f == "ubInputListId" else "2") + ";" for i, f in enumerate(fields))
        exercises.append("{ " + name + " src, dst;\n" + initial + "\n" +
            "src.active_=true; auto expected=src; src.MovePipelineTo(dst);\n" +
            "assert(!src.active_ && dst.active_);\n" + checks + "\n" +
            "dst.MovePipelineTo(src); assert(src.active_ && !dst.active_);\n" +
            checks.replace("dst.", "src.") + "\n}")
    source = tmp_path / "move.cpp"
    source.write_text("#include <cassert>\n" + "\n".join(definitions) +
                      "\nint main(){for(unsigned trial=0;trial<10000;++trial){\n" +
                      "\n".join(exercises) + "\n}}", encoding="utf-8")
    executable = tmp_path / "move-test"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
