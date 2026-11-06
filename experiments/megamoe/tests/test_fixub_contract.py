"""Capture actual FIX parameters and execute actual single-UB channel methods."""
import os
from pathlib import Path
import shlex, shutil, subprocess
import pytest

@pytest.mark.parametrize("name,stub,expected", [("fix_params_test","fixub_stubs","passed"),("single_ub_channel_test","consumer_stubs","cases=64 passed")])
def test_fixub_contract(tmp_path,name,stub,expected):
    cxx=shlex.split(os.environ.get("CXX","c++"))
    if not shutil.which(cxx[0]): pytest.skip("host C++ compiler required")
    tests=Path(__file__).resolve().parent
    exe=tmp_path/(name+(".exe" if os.name=="nt" else ""))
    subprocess.run(cxx+["-std=c++17","-O2","-UNDEBUG","-I",str(tests/stub),"-I",str(tests.parent/"csrc/kernels/mega_moe/urma"),str(tests/(name+".cpp")),"-o",str(exe)],check=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert expected in r.stdout
