"""Actual parallel summary helpers versus the existing serial summary."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

def test_parallel_summary_matches_serial(tmp_path):
    compiler=shlex.split(os.environ.get("CXX","c++"))
    if not shutil.which(compiler[0]):pytest.skip("requires a C++ compiler")
    root=Path(__file__).resolve().parents[1]
    exe=tmp_path/("route_summary.exe" if os.name=="nt" else "route_summary")
    subprocess.run(compiler+["-std=c++17","-O2","-I",str(root/"csrc/kernels/mega_moe/urma"),str(root/"tests/route_summary_parallel_test.cpp"),"-o",str(exe)],check=True)
    output=subprocess.run([str(exe)],check=True,text=True,capture_output=True).stdout
    assert "parallel AIV0 route summary cases passed=4644" in output
