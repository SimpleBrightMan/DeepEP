"""Production parallel route planner vs independent legacy descriptors."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

def test_parallel_descriptor_order_reorder_and_shared_schedule(tmp_path):
    compiler=shlex.split(os.environ.get("CXX","c++"))
    if not shutil.which(compiler[0]):pytest.skip("requires a C++ compiler")
    root=Path(__file__).resolve().parents[1]
    exe=tmp_path/("parallel_routes.exe" if os.name=="nt" else "parallel_routes")
    subprocess.run(compiler+["-std=c++17","-O2","-I",str(root/"csrc/kernels/mega_moe/urma"),str(root/"tests/parallel_route_plan_test.cpp"),"-o",str(exe)],check=True)
    result=subprocess.run([str(exe)],check=True,text=True,capture_output=True)
    assert "cases passed=1800" in result.stdout
