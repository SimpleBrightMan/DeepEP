"""Compile actual GMM1 role/lifetime helpers without CANN; DMA remains mocked."""
from pathlib import Path

import pytest

from test_device_schedule import ROOT, compile_and_run
from test_dispatch import function_source


@pytest.mark.parametrize("core", [0, 1])
def test_gmm1_role_and_slice_lifetime(tmp_path, core):
    source = (ROOT / "experiments/megamoe/csrc/kernels/mega_moe/stage/mega_moe_gmm1_activation.hpp").read_text(
        encoding="utf-8"
    )
    methods = "\n".join(function_source(source, name) for name in (
        "Gmm1AicMmadA8W4", "Gmm1Aiv0PrologueA8W4", "RunGmm1ComputeTiles",
    ))
    (tmp_path / "gmm1_role_methods.inc").write_text(methods, encoding="utf-8")
    fixture = Path(__file__).with_name("gmm1_roles_host.cpp").read_text(encoding="utf-8")
    unit = tmp_path / "gmm1_roles.cpp"
    unit.write_text(f"#define TEST_CORE {core}\n" + fixture, encoding="utf-8")
    compile_and_run(unit, tmp_path)
