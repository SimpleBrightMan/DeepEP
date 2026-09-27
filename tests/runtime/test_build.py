"""Build preflight and artifact checks runnable without SDKs or devices."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
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


def test_lock_declares_operators_without_claiming_device_qualification():
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    assert lock["device_kernels"] == ["full-notify-metadata", "expanded-dispatch-urma-mte", "expanded-combine-urma-mte"]
    assert len(lock["combine_reference"]["revision"]) == 40
    assert len(lock["dispatch_reference"]["revision"]) == 40
    assert len(lock["notify_reference"]["revision"]) == 40
    assert len(lock["shmem"]["revision"]) == 40
    assert len(lock["runtime_reference"]["revision"]) == 40
    assert len(lock["shmem"]["header_sha256_lf"]) == 6


@pytest.mark.parametrize(
    "capability", ["elastic-notify-metadata-v1", "elastic-dispatch-expanded-v1", "elastic-combine-expanded-v1"]
)
def test_elastic_wheel_requires_device_library(tmp_path, capability):
    validator = load(ROOT / "scripts/validate_package.py")
    operations = ["ElasticBuffer.dispatch", "ElasticBuffer.combine"]
    for device_library in (False, True):
        wheel = tmp_path / f"elastic-{device_library}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("deep_ep/_C.fake.so", b"test")
            archive.writestr("deep_ep/deepseek-deepep.LICENSE", b"test")
            archive.writestr(
                "deep_ep/_build_info.json",
                json.dumps(
                    {
                        "communication_operations": operations,
                        "capabilities": [capability],
                    }
                ),
            )
            if device_library:
                archive.writestr("deep_ep/libdeepep_kernels.so", b"test")
        if device_library:
            assert validator.inspect_wheel(wheel)["communication_operations"] == operations
        else:
            with pytest.raises(RuntimeError, match="Device library"):
                validator.inspect_wheel(wheel)


def test_cache_survives_source_edits_but_not_toolchain_changes(build):
    configuration = {
        "source_sha256": "before",
        "source_version": "1",
        "build_type": "Release",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "arch": "Ascend950",
        "compiler_sha256": "host-1",
        "device_compiler_sha256": "device-1",
        "python_executable": "/python/bin/python",
        "shmem_device_headers": {"header.h": "before"},
        "build_environment": {"CXXFLAGS": ""},
    }
    key = build.build_cache_key(configuration)
    assert build.build_cache_key({**configuration, "source_sha256": "after", "source_version": "2"}) == key
    for name, value in (
        ("build_type", "RelWithDebInfo"),
        ("ASCEND_HOME_PATH", "/sdk/new-cann"),
        ("SHMEM_ROOT", "/sdk/new-shmem"),
        ("compiler_sha256", "host-2"),
        ("device_compiler_sha256", "device-2"),
        ("python_executable", "/new-python/bin/python"),
        ("shmem_device_headers", {"header.h": "after"}),
        ("build_environment", {"CXXFLAGS": "-g"}),
    ):
        assert build.build_cache_key({**configuration, name: value}) != key


def test_packaging_uses_cmake_capabilities_and_removes_old_library(build, monkeypatch, tmp_path):
    from setuptools import Distribution, Extension

    # This CPU-only test mocks the build tools, including their discovery paths.
    pybind11_cmake_dir = str(tmp_path / "pybind11 cmake")
    pybind11 = ModuleType("pybind11")
    pybind11.get_cmake_dir = lambda: pybind11_cmake_dir
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)

    configuration = {
        "source_sha256": "current",
        "source_version": "1",
        "cmake_path": "/tools/cmake",
        "compiler_path": "/tools/c++",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "build_type": "RelWithDebInfo",
        "python_executable": "/private/python",
        "build_environment": {"CXXFLAGS": "-I/private/include"},
    }
    monkeypatch.setattr(build, "build_configuration", lambda: configuration)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "6")
    ext = Extension("deep_ep._C", sources=[])
    command = build.CMakeBuild(Distribution({"ext_modules": [ext]}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / "temp")
    command.build_lib = str(tmp_path / "lib")
    destination = Path(command.get_ext_fullpath(ext.name)).parent
    destination.mkdir(parents=True)
    (destination / "libdeepep_notify.so").write_bytes(b"old")
    obsolete_modules = [
        f"_{op}{suffix}" for op in ("notify", "dispatch", "combine") for suffix in (".py", "_layout.py")
    ]
    for name in obsolete_modules:
        (destination / name).write_text("old module\n")
    calls = []
    capabilities = {"capabilities": ["test-built-target"], "communication_operations": []}

    def run(argv):
        calls.append(list(map(str, argv)))
        if "-B" in argv:
            directory = Path(argv[argv.index("-B") + 1])
            (directory / "compiler-info.json").write_text('{"id":"test","version":"1"}')
            (directory / "capabilities.json").write_text(json.dumps(capabilities))
        if "--install" in argv:
            (destination / "libdeepep_kernels.so").write_bytes(b"new")

    monkeypatch.setattr(build, "checked_run", run)
    command.build_extension(ext)
    assert "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in calls[0]
    assert "-DDEEPEP_DEVICE_JOBS=6" in calls[0]
    assert f"-Dpybind11_DIR={pybind11_cmake_dir}" in calls[0]
    assert calls[1][-2:] == ["--parallel", "6"]
    assert not (destination / "libdeepep_notify.so").exists()
    assert all(not (destination / name).exists() for name in obsolete_modules)
    info = json.loads((destination / "_build_info.json").read_text())
    assert info["capabilities"] == capabilities["capabilities"]
    assert info["source_sha256"] == "current"
    assert not {"cmake_path", "compiler_path", "python_executable", "build_environment"}.intersection(info)
