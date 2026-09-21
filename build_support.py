"""One native configure/build/install path; no network or device side effects."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


def source_fingerprint():
    files = [
        ROOT / name
        for name in (
            "VERSION",
            "setup.py",
            "pyproject.toml",
            "build_support.py",
            "CMakeLists.txt",
            "dependencies.lock.json",
            "MANIFEST.in",
            "requirements-dev.txt",
            "requirements-build.txt",
            "third-party/deepseek-deepep.LICENSE",
        )
    ]
    for name in ("deep_ep", "csrc", "cmake", "scripts", "tests", "benchmarks"):
        files.extend(
            path
            for path in (ROOT / name).rglob("*")
            if path.suffix in {".py", ".cpp", ".hpp", ".h", ".cmake", ".json"} and path.name != "_build_info.json"
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def checked_run(argv, **kwargs):
    subprocess.run([str(arg) for arg in argv], check=True, timeout=1800, **kwargs)


def build_configuration(environment=None):
    env = os.environ if environment is None else environment
    if platform.system() != "Linux":
        raise RuntimeError("Native wheels require Linux with an Ascend SDK")
    if env.get("DEEPEP_NPU_ARCH") != "Ascend950":
        raise RuntimeError("DEEPEP_NPU_ARCH must be Ascend950; other targets are unqualified")
    if env.get("DEEPEP_BUILD_MEGAMOE", "OFF") != "OFF":
        raise RuntimeError("MegaMoE is not implemented; DEEPEP_BUILD_MEGAMOE must be OFF")
    result = {"arch": "Ascend950", "python": sys.version, "machine": platform.machine()}
    for name, header in (
        ("ASCEND_HOME_PATH", "include/acl/acl.h"),
        ("SHMEM_ROOT", "include/host/shmem_host_def.h"),
    ):
        value = env.get(name)
        if not value or not (Path(value) / header).is_file():
            raise RuntimeError(f"{name} must name an installed SDK containing {header}")
        result[name] = str(Path(value).resolve())
    manifest = Path(result["SHMEM_ROOT"]) / "deepep-sdk.json"
    if not manifest.is_file():
        raise RuntimeError("SHMEM SDK needs deepep-sdk.json; see scripts/record_shmem_sdk.py")
    sdk = json.loads(manifest.read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
    if sdk.get("revision") != lock["shmem"]["revision"]:
        raise RuntimeError("SHMEM SDK source revision differs from dependencies.lock.json")
    for relative, expected in lock["shmem"]["header_sha256_lf"].items():
        data = (manifest.parent / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"SHMEM public API differs from the pinned SDK: {relative}")
    for relative, expected in sdk["sha256"].items():
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
            raise RuntimeError("Invalid SHMEM SDK manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"SHMEM SDK checksum mismatch: {relative}")
    if "lib/libshmem.so" not in sdk["sha256"] or not any(key.startswith("include/") for key in sdk["sha256"]):
        raise RuntimeError("SHMEM SDK manifest must cover headers and libshmem.so")
    result["shmem"] = sdk
    result["pybind11"] = __import__("pybind11").__version__
    if result["pybind11"] != lock["python_build"]["pybind11"]:
        raise RuntimeError("pybind11 version differs from dependencies.lock.json")
    version_file = Path(result["ASCEND_HOME_PATH"]) / "compiler/version.info"
    versions = dict(line.split("=", 1) for line in version_file.read_text().splitlines() if "=" in line)
    if versions.get("Version") != "9.2.0":
        raise RuntimeError("This runtime build requires the CANN 9.2.0 qualification baseline")
    result["cann"] = {
        "version": versions["Version"],
        "timestamp": versions.get("timestamp", "unknown"),
    }
    result["source_version"] = (ROOT / "VERSION").read_text().strip()
    result["source_sha256"] = source_fingerprint()
    compiler = shutil.which(env.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("CXX must select an installed C++ compiler executable")
    result["compiler_path"] = str(Path(compiler).resolve())
    return result


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import pybind11

        configuration = build_configuration()
        fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
        directory = Path(self.build_temp).resolve() / fingerprint
        destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        directory.mkdir(parents=True, exist_ok=True)
        checked_run(
            [
                "cmake",
                "-S",
                ROOT,
                "-B",
                directory,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                f"-DCMAKE_CXX_COMPILER={configuration['compiler_path']}",
                f"-DASCEND_HOME_PATH={configuration['ASCEND_HOME_PATH']}",
                f"-DSHMEM_ROOT={configuration['SHMEM_ROOT']}",
                f"-DCMAKE_INSTALL_PREFIX={destination}",
            ]
        )
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
        if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
            raise RuntimeError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")
        checked_run(["cmake", "--build", directory, "--parallel", jobs])
        checked_run(["cmake", "--install", directory])
        # Deployment metadata contains identities/hashes, never build-machine paths.
        info = {
            key: value
            for key, value in configuration.items()
            if key not in {"ASCEND_HOME_PATH", "SHMEM_ROOT", "compiler_path"}
        }
        info["compiler"] = json.loads((directory / "compiler-info.json").read_text())
        info["capabilities"] = ["runtime-mte", "event"]
        info["communication_operations"] = []
        (destination / "_build_info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
