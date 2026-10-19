#!/usr/bin/env bash
# Copyright (c) 2026, SimpleBrightMan
#
# Fast development build for the All-to-All benchmark. This bypasses
# setuptools/pip and its MANIFEST.in scan, which can stall on checkouts with
# large third-party trees. It installs _C.so and libdeepep_kernels.so into the
# source deep_ep/ directory, so `run_16cards.sh` uses the source tree directly.
#
# Usage:
#   CANN_ENV=/path/to/set_env.sh SHMEM_ROOT=/path/to/shmem PYTHON=/path/to/python \
#     bash benchmarks/alltoall_mte/build_dev.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python3}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
SHMEM_ROOT=${SHMEM_ROOT:-}
JOBS=${JOBS:-64}
BUILD_DIR=${BUILD_DIR:-$ROOT/build/alltoall-dev}

[[ -n "$SHMEM_ROOT" && -d "$SHMEM_ROOT/lib" ]] || {
    echo "SHMEM_ROOT must point at the installed SHMEM SDK" >&2
    exit 2
}
[[ -f "$CANN_ENV" ]] || { echo "CANN_ENV not found: $CANN_ENV" >&2; exit 2; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "PYTHON not found: $PYTHON" >&2; exit 2; }

source "$CANN_ENV"
[[ ! -f "$SHMEM_ROOT/set_env.sh" ]] || source "$SHMEM_ROOT/set_env.sh"
export SHMEM_ROOT
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export DEEPEP_NPU_ARCH=Ascend950

CMAKE_BIN=$(command -v cmake || true)
if [[ -z "$CMAKE_BIN" ]]; then
    CMAKE_BIN=$("$PYTHON" - <<'PY'
import pathlib
try:
    import cmake
except ImportError:
    raise SystemExit(0)
print(pathlib.Path(cmake.__file__).parent / "data" / "bin" / "cmake")
PY
)
fi
[[ -n "$CMAKE_BIN" && -x "$CMAKE_BIN" ]] || {
    echo "cmake not found on PATH or in the Python environment" >&2
    exit 2
}

PYBIND_DIR=$("$PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")

"$CMAKE_BIN" -S "$ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEEPEP_DEVICE_JOBS="$JOBS" \
    -DPython_EXECUTABLE="$PYTHON" \
    -Dpybind11_DIR="$PYBIND_DIR" \
    -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
    -DSHMEM_ROOT="$SHMEM_ROOT" \
    -DDEEPEP_BUILD_MEGAMOE=OFF \
    -DDEEPEP_BUILD_EP_GMM_FUSED=OFF \
    -DCMAKE_INSTALL_PREFIX="$ROOT/deep_ep"
"$CMAKE_BIN" --build "$BUILD_DIR" --parallel "$JOBS"
"$CMAKE_BIN" --install "$BUILD_DIR"

# _native.py only requires the SHMEM library hash; keep the rest minimal.
if [[ ! -f "$ROOT/deep_ep/_build_info.json" ]]; then
    shmem_hash=$(sha256sum "$SHMEM_ROOT/lib/libshmem.so" | awk '{print $1}')
    cat >"$ROOT/deep_ep/_build_info.json" <<EOF
{
  "arch": "Ascend950",
  "build_type": "Release",
  "shmem": {
    "sha256": {
      "lib/libshmem.so": "$shmem_hash"
    }
  }
}
EOF
fi

echo "[build_dev] installed:"
ls -l "$ROOT/deep_ep/_C"*.so "$ROOT/deep_ep/libdeepep_kernels.so"

cd /tmp
PYTHONPATH="$ROOT" "$PYTHON" - <<'PY'
from deep_ep._native import require_native
module = require_native("build_dev")
assert hasattr(module, "alltoall") and hasattr(module, "alltoall_drain")
print("[build_dev] native import ok:", module.shmem_library_path())
PY
