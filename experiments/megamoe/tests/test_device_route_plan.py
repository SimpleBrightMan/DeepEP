"""Execute the production descriptor compiler with a CPU memory adapter."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest


def test_device_descriptors_match_independent_host_oracle(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        pytest.skip("requires a C++ compiler")
    root = Path(__file__).resolve().parents[1]
    exe = tmp_path / ("device_routes.exe" if os.name == "nt" else "device_routes")
    subprocess.run(compiler + ["-std=c++17", "-O2", "-I", str(root / "csrc"),
        str(root / "tests/device_route_plan_test.cpp"), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], check=True, text=True, capture_output=True)
    assert "cases passed=675" in result.stdout
