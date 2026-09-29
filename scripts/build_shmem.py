"""Explicitly build the pinned SHMEM submodule; pip/wheel never invokes this script."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

from record_shmem_sdk import record

ROOT = Path(__file__).resolve().parents[1]


def cann_version(expected):
    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann or not (Path(cann) / "compiler/version.info").is_file():
        raise RuntimeError("Source the CANN set_env.sh first (ASCEND_HOME_PATH is missing or invalid)")
    version = dict(
        line.split("=", 1) for line in (Path(cann) / "compiler/version.info").read_text().splitlines() if "=" in line
    )
    if version.get("Version") != expected:
        raise RuntimeError(f"This build requires CANN {expected}")
    return version


def prepare(root=ROOT, *, rebuild=False, if_needed=False):
    if platform.system() != "Linux":
        raise RuntimeError("SHMEM compilation requires Linux and the pinned CANN SDK")
    root = Path(root).resolve()
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))
    source = root / "third-party/shmem"
    if not (source / ".git").exists() or not (source / "scripts/build.sh").is_file():
        raise RuntimeError("Initialize dependencies first: git submodule update --init --recursive")
    if source.resolve() != source or source.parent.resolve() != source.parent:
        raise RuntimeError("SHMEM source must not be a symlink to another checkout")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
        raise RuntimeError("third-party/shmem must be its own initialized Git submodule")
    revision = git("rev-parse", "HEAD")
    if revision != lock["shmem"]["revision"]:
        raise RuntimeError("SHMEM revision differs from dependencies.lock.json; do not build a moving branch")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("SHMEM has local changes; preserve/review them before building the pinned SDK")

    version = cann_version(lock["cann_version"])
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")

    # 上游脚本会清空 build/install，并生成 ci 打包产物；禁止跳转到其他目录。
    for name in ("build", "install", "ci", "3rdparty"):
        path = source / name
        if path.resolve() != path:
            raise RuntimeError(f"Refusing SHMEM build through a symlink: {path}")
    manifest = source / "install/shmem/deepep-sdk.json"
    if if_needed and not rebuild and manifest.is_file():
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        expected_build = {
            "cann_version": version["Version"],
            "cann_timestamp": version.get("timestamp", "unknown"),
            "target": lock["target"],
            "source_patches": [],
        }
        if identity.get("revision") != revision or identity.get("build") != expected_build:
            raise RuntimeError("Existing SHMEM build identity changed; stop SDK users and rebuild explicitly")
        # 复用也校验全部头文件/库哈希，不把存在 manifest 当成构建成功。
        import sys

        sys.path.insert(0, str(ROOT))
        from build_support import validate_shmem_sdk

        validate_shmem_sdk(manifest.parent, lock["shmem"])
        print(f"[SHMEM] Reusing validated SDK: {manifest.parent}", flush=True)
        return manifest
    if not rebuild and any((source / name).exists() for name in ("build", "install")):
        raise FileExistsError(
            "SHMEM build/install already exists; reuse its SDK, or stop users of that SDK "
            "and pass --rebuild to regenerate it (then rebuild dependent wheels)"
        )
    print(f"[SHMEM] source={revision} CANN={version['Version']} target={lock['target']}", flush=True)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("Git is required to prepare SHMEM dependencies")
    # 上游脚本会调用 Git 下载 json 等依赖；只隔离 Git 的库路径，保留编译器所需的 CANN 环境。
    with tempfile.TemporaryDirectory(prefix="deepep-shmem-git-") as git_dir:
        wrapper = Path(git_dir) / "git"
        wrapper.write_text(
            "#!/bin/sh\nunset LD_LIBRARY_PATH LD_PRELOAD\n"
            f'exec {shlex.quote(str(Path(git_executable).resolve()))} "$@"\n'
        )
        wrapper.chmod(0o700)
        build_env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=jobs)
        build_env["PATH"] = git_dir + os.pathsep + build_env.get("PATH", os.defpath)
        # -e 防止上游中间命令失败后继续打包，随后误登记旧库为成功产物。
        subprocess.run(
            ["bash", "-e", "scripts/build.sh", "-soc_type", lock["target"]],
            cwd=source,
            env=build_env,
            check=True,
        )
    sdk = source / "install/shmem"
    manifest = record(
        sdk,
        revision,
        build_info={
            "cann_version": version["Version"],
            "cann_timestamp": version.get("timestamp", "unknown"),
            "target": lock["target"],
            "source_patches": [],
        },
    )
    print(f"[SHMEM] SDK identity: {manifest}", flush=True)
    print("[SHMEM] Add third-party/shmem/install/shmem/lib to LD_LIBRARY_PATH before loading the wheel.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate this submodule's build/install artifacts")
    parser.add_argument("--if-needed", action="store_true", help="Reuse a verified SDK from this CANN build")
    args = parser.parse_args()
    prepare(rebuild=args.rebuild, if_needed=args.if_needed)
