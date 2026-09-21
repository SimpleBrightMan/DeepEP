"""Build preflight and artifact checks runnable without SDKs or devices."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "build_support.py").is_file():
    pytest.skip("Build-source tests run before packaging", allow_module_level=True)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build(monkeypatch):
    module = load(ROOT / "build_support.py")
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    return module


def test_platform_and_architecture_rejected(build, monkeypatch):
    with pytest.raises(RuntimeError, match="DEEPEP_NPU_ARCH"):
        build.build_configuration({})
    monkeypatch.setattr(build.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950"})


@pytest.mark.parametrize("value", ["ON", "on", "yes", "1", ""])
def test_beta_never_silently_enabled(build, value):
    with pytest.raises(RuntimeError, match="MegaMoE"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950", "DEEPEP_BUILD_MEGAMOE": value})


def test_missing_sdk_rejected(build):
    with pytest.raises(RuntimeError, match="ASCEND_HOME_PATH"):
        build.build_configuration({"DEEPEP_NPU_ARCH": "Ascend950"})


def test_sdk_manifest_required(build, tmp_path):
    cann, sdk = tmp_path / "cann", tmp_path / "shmem"
    (cann / "include/acl").mkdir(parents=True)
    (cann / "include/acl/acl.h").touch()
    (sdk / "include/host").mkdir(parents=True)
    (sdk / "include/host/shmem_host_def.h").touch()
    env = {
        "DEEPEP_NPU_ARCH": "Ascend950",
        "ASCEND_HOME_PATH": str(cann),
        "SHMEM_ROOT": str(sdk),
    }
    with pytest.raises(RuntimeError, match="deepep-sdk.json"):
        build.build_configuration(env)
    (sdk / "deepep-sdk.json").write_text('{"revision":"wrong"}')
    with pytest.raises(RuntimeError, match="revision"):
        build.build_configuration(env)


def test_metadata_no_sdk_or_framework(tmp_path):
    # Query in a copy so egg metadata does not alter the checkout.
    import shutil

    for name in ("setup.py", "build_support.py", "VERSION"):
        shutil.copy2(ROOT / name, tmp_path / name)
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ascend-deepep"


@pytest.mark.parametrize(
    "members,error",
    [
        ({}, "native extension"),
        ({"deep_ep/_C.fake.so": b"test"}, "build identity"),
        (
            {
                "deep_ep/_C.fake.so": b"test",
                "deep_ep/_build_info.json": b"{}",
                "tests/x.py": b"",
            },
            "Unexpected",
        ),
    ],
)
def test_wheel_rejects_missing_or_unexpected_contents(tmp_path, members, error):
    validator = load(ROOT / "scripts/validate_package.py")
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(RuntimeError, match=error):
        validator.inspect_wheel(wheel)


def test_lock_declares_no_kernel_qualification():
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    assert lock["device_kernels"] == []
    assert len(lock["shmem"]["revision"]) == 40
    assert len(lock["runtime_reference"]["revision"]) == 40
    assert len(lock["shmem"]["header_sha256_lf"]) == 6
