"""Check production allocation and the frozen supported MTE flag contract."""
from pathlib import Path

from test_device_schedule import compile_and_run


def test_workspace_reset_and_supported_layouts(tmp_path):
    kernel_header = tmp_path / "kernel_tiling/kernel_tiling.h"
    kernel_header.parent.mkdir(parents=True)
    kernel_header.write_text("#pragma once\n", encoding="utf-8")
    math_header = tmp_path / "op_common/op_host/util/math_util.h"
    math_header.parent.mkdir(parents=True)
    math_header.write_text("""#pragma once
namespace Ops::Base {
template <class A, class B> constexpr auto CeilDiv(A a, B b) { return (a+b-1)/b; }
template <class A, class B> constexpr auto CeilAlign(A a, B b) { return CeilDiv(a,b)*b; }
}
""", encoding="utf-8")
    compile_and_run(Path(__file__).with_name("workspace_layout_host.cpp"), tmp_path)
