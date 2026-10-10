"""Compile actual device protocol helpers against a CPU event-recording stub."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_completion_credit_accounting(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("C++17 compiler or ziglang required for device helper CPU test")
    tests = Path(__file__).parent
    executable = tmp_path / ("async.exe" if sys.platform == "win32" else "async")
    subprocess.run([*command, "-std=c++17", "-I", str(tests / "async_stub"),
                    str(tests / "gmm1_async_completion_test.cpp"), "-o", str(executable)], check=True,
                   env=dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str(tmp_path / "zig-global"),
                            ZIG_LOCAL_CACHE_DIR=str(tmp_path / "zig-local")))
    subprocess.run([str(executable)], check=True, timeout=20)
