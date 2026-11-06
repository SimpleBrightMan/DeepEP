"""Actual producer/consumer tile owner traversal, with reversed GMM1 ownership."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

def test_owned_cursor(tmp_path):
    cxx=shlex.split(os.environ.get("CXX","c++"))
    if not shutil.which(cxx[0]): pytest.skip("host C++ compiler required")
    tests=Path(__file__).resolve().parent
    exe=tmp_path/("cursor.exe" if os.name=="nt" else "cursor")
    subprocess.run(cxx+["-std=c++17","-O2","-UNDEBUG","-I",str(tests.parent/"csrc/kernels/mega_moe/urma"),str(tests/"owned_compute_cursor_test.cpp"),"-o",str(exe)],check=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert "owned cursor cases=14000" in result.stdout
