"""One native configure/build/install path; no network or device side effects."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import warnings

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
            if (path.suffix in {".py", ".cpp", ".hpp", ".h", ".cmake", ".json"} or path.name == "CMakeLists.txt")
            and path.name != "_build_info.json"
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def checked_run(argv, **kwargs):
    subprocess.run([str(arg) for arg in argv], check=True, timeout=1800, **kwargs)


def build_cache_key(configuration):
    """Keep object files across source edits; isolate incompatible toolchains."""
    identity = {key: value for key, value in configuration.items() if key not in {"source_sha256", "source_version"}}
    identity["source_root"] = str(ROOT.resolve())
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def find_host_compiler(env):
    """Select an executable without resolving compiler-wrapper symlinks."""
    explicit = env.get("CXX")
    candidates = [explicit] if explicit else ["c++", "g++", "clang++", "bisheng++", "bisheng"]
    if not explicit and env.get("ASCEND_HOME_PATH"):
        directory = Path(env["ASCEND_HOME_PATH"]) / "tools/bisheng_compiler/bin"
        candidates.extend(str(directory / name) for name in ("bisheng++", "bisheng"))
    failures = []
    for name in candidates:
        compiler = shutil.which(name, path=env.get("PATH"))
        if compiler is None:
            failures.append(f"{name}: not found")
            continue
        # c++ -> ccache is valid; selecting bare ccache is not a compiler.
        if Path(compiler).name in {"ccache", "sccache"}:
            failures.append(f"{name}: select a C++ compiler, not a standalone cache launcher")
            continue
        compiler = str(Path(compiler).absolute())
        try:
            version = subprocess.check_output([compiler, "--version"], stderr=subprocess.STDOUT, timeout=30)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{name}: --version failed ({type(exc).__name__})")
            continue
        return compiler, version
    prefix = "Explicit CXX is unusable" if explicit else "No runnable Host C++ compiler found"
    raise RuntimeError(f"{prefix}. Set CXX to a C++17 compiler executable. Tried: " + "; ".join(failures))


def build_configuration(environment=None):
    env = os.environ if environment is None else environment
    if platform.system() != "Linux":
        raise RuntimeError("Native wheels require Linux with an Ascend SDK")
    if env.get("DEEPEP_NPU_ARCH") != "Ascend950":
        raise RuntimeError("DEEPEP_NPU_ARCH must be Ascend950; other targets are unqualified")
    if env.get("DEEPEP_BUILD_MEGAMOE", "OFF") != "OFF":
        raise RuntimeError("MegaMoE is not implemented; DEEPEP_BUILD_MEGAMOE must be OFF")
    result = {"arch": "Ascend950", "python": sys.version, "machine": platform.machine()}
    result["build_type"] = env.get("DEEPEP_BUILD_TYPE", "Release")
    if result["build_type"] not in {"Release", "RelWithDebInfo", "Debug"}:
        raise RuntimeError("DEEPEP_BUILD_TYPE must be Release, RelWithDebInfo or Debug")
    for name, header in (
        ("ASCEND_HOME_PATH", "include/acl/acl.h"),
        ("SHMEM_ROOT", "include/host/shmem_host_def.h"),
    ):
        value = env.get(name)
        if not value or not (Path(value) / header).is_file():
            raise RuntimeError(f"{name} must name an installed SDK containing {header}")
        result[name] = str(Path(value).resolve())
    manifest = Path(result["SHMEM_ROOT"]) / "deepep-sdk.json"
    lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
    sdk = {}
    try:
        if not manifest.is_file():
            raise RuntimeError("SHMEM SDK has no deepep-sdk.json")
        sdk = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(sdk, dict):
            raise ValueError("Invalid SHMEM SDK manifest")
        if sdk.get("revision") != lock["shmem"]["revision"]:
            raise RuntimeError("SHMEM SDK source revision differs from dependencies.lock.json")
        for relative, expected in lock["shmem"]["header_sha256_lf"].items():
            data = (manifest.parent / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"SHMEM public API differs from the pinned SDK: {relative}")
        hashes = sdk.get("sha256")
        if not isinstance(hashes, dict):
            raise ValueError("Invalid SHMEM SDK manifest checksums")
        for relative, expected in hashes.items():
            path = (manifest.parent / relative).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
                raise RuntimeError("Invalid SHMEM SDK manifest path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"SHMEM SDK checksum mismatch: {relative}")
        if "lib/libshmem.so" not in hashes or not any(key.startswith("include/") for key in hashes):
            raise RuntimeError("SHMEM SDK manifest must cover headers and libshmem.so")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        warnings.warn(f"{exc}; continuing with SHMEM_ROOT={manifest.parent}", RuntimeWarning, stacklevel=2)
    library = manifest.parent / "lib/libshmem.so"
    if not library.is_file():
        raise RuntimeError(f"SHMEM SDK is missing {library}")
    files = sorted(
        path for path in (manifest.parent / "include").rglob("*") if path.suffix in {".h", ".hpp"} and path.is_file()
    )
    files.append(library)
    # Record actual inputs so replacing an unpinned SDK invalidates the build cache.
    result["shmem"] = {
        "revision": sdk.get("revision", "unknown") if isinstance(sdk, dict) else "unknown",
        "sha256": {
            path.relative_to(manifest.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
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
    compiler, compiler_version = find_host_compiler(env)
    result["compiler_path"] = compiler
    result["compiler_sha256"] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
    result["compiler_version_sha256"] = hashlib.sha256(compiler_version).hexdigest()
    print(f"Host C++ compiler: {compiler}", flush=True)
    cmake = shutil.which("cmake", path=env.get("PATH"))
    if cmake is None:
        raise RuntimeError("CMake must be installed and available on PATH")
    result["cmake_path"] = str(Path(cmake).resolve())
    result["cmake_version"] = subprocess.check_output([cmake, "--version"], text=True, timeout=30).splitlines()[0]
    # Keep the virtualenv path even when its executable symlinks to system Python.
    result["python_executable"] = str(Path(sys.executable).absolute())
    result["build_environment"] = {
        name: env.get(name, "")
        for name in (
            "CXXFLAGS",
            "LDFLAGS",
            "CPPFLAGS",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "CMAKE_GENERATOR",
            "CMAKE_GENERATOR_PLATFORM",
            "CMAKE_GENERATOR_TOOLSET",
            "CMAKE_PREFIX_PATH",
        )
    }
    device_compiler = Path(result["ASCEND_HOME_PATH"]) / "tools/bisheng_compiler/bin/bisheng"
    if not device_compiler.is_file():
        raise RuntimeError("Device kernels require CANN Bisheng with -xasc support")
    result["device_compiler_sha256"] = hashlib.sha256(device_compiler.read_bytes()).hexdigest()
    sdk_root = Path(result["SHMEM_ROOT"])
    for relative in ("include/shmem.h", "src/device/gm2gm/shmemi_device_cc.h"):
        if not (sdk_root / relative).is_file():
            raise RuntimeError(f"Kernels require the full installed SHMEM Device SDK: missing {relative}")
    result["shmem_device_headers"] = {
        p.relative_to(sdk_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((sdk_root / "src").rglob("*"))
        if p.is_file() and p.suffix in {".h", ".hpp"}
    }
    return result


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import pybind11

        configuration = build_configuration()
        fingerprint = build_cache_key(configuration)
        directory = Path(self.build_temp).resolve() / fingerprint
        destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
        if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
            raise RuntimeError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")
        cmake = configuration["cmake_path"]
        directory.mkdir(parents=True, exist_ok=True)
        checked_run(
            [
                cmake,
                "-S",
                ROOT,
                "-B",
                directory,
                f"-DCMAKE_BUILD_TYPE={configuration['build_type']}",
                f"-DDEEPEP_DEVICE_JOBS={jobs}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                f"-DCMAKE_CXX_COMPILER={configuration['compiler_path']}",
                f"-DASCEND_HOME_PATH={configuration['ASCEND_HOME_PATH']}",
                f"-DSHMEM_ROOT={configuration['SHMEM_ROOT']}",
                f"-DCMAKE_INSTALL_PREFIX={destination}",
            ]
        )
        checked_run([cmake, "--build", directory, "--parallel", jobs])
        checked_run([cmake, "--install", directory])
        # A reused setuptools build_lib may still contain the former library.
        (destination / "libdeepep_notify.so").unlink(missing_ok=True)
        # Elastic implementation/layout modules now live under buffers/.
        for operation in ("notify", "dispatch", "combine"):
            for suffix in (".py", "_layout.py"):
                (destination / f"_{operation}{suffix}").unlink(missing_ok=True)
        # Deployment metadata contains identities/hashes, never build-machine paths.
        info = {
            key: value
            for key, value in configuration.items()
            if key
            not in {
                "ASCEND_HOME_PATH",
                "SHMEM_ROOT",
                "compiler_path",
                "cmake_path",
                "python_executable",
                "build_environment",
            }
        }
        info["compiler"] = json.loads((directory / "compiler-info.json").read_text())
        info.update(json.loads((directory / "capabilities.json").read_text(encoding="utf-8")))
        (destination / "_build_info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
