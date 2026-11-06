"""Run the production batch driver with issue/completion/credit recording."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

def test_direct_reorder_batch(tmp_path):
    compiler=shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]): pytest.skip("host C++ compiler required")
    tests=Path(__file__).resolve().parent
    binary=tmp_path/("batch.exe" if os.name=="nt" else "batch")
    subprocess.run(compiler+["-std=c++17","-O2","-UNDEBUG","-I",str(tests.parent/"csrc/kernels/mega_moe/urma"),str(tests/"completion_batch_test.cpp"),"-o",str(binary)],check=True)
    result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
    assert "2305 completion batch cases passed" in result.stdout
