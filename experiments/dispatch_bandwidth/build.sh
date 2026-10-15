#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON:-python3}
build_dir=${BUILD_DIR:-"$root/build-dispatch-bandwidth"}
: "${ASCEND_HOME_PATH:?Source the matching CANN set_env.sh first}"
export SHMEM_ROOT=${SHMEM_ROOT:-"$root/third-party/shmem/install/shmem"}
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE=OFF
export DEEPEP_MEGAMOE_TIMER=OFF
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Reuse the repository's CANN/SHMEM/host-toolchain checks; do not install a wheel.
"$python_bin" - "$root" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.argv[1])))
from build_support import build_configuration

build_configuration()
PY
pybind_dir=$("$python_bin" -m pybind11 --cmakedir)
python_path=$("$python_bin" -c 'import sys; print(sys.executable)')
cmake -S "$root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$python_path" -Dpybind11_DIR="$pybind_dir" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSHMEM_ROOT="$SHMEM_ROOT" \
    -DDEEPEP_BUILD_DISPATCH_BENCH=ON -DDEEPEP_BUILD_MEGAMOE=OFF \
    -DDEEPEP_MEGAMOE_TIMER=OFF -DDEEPEP_BUILD_EP_GMM_FUSED=OFF
cmake --build "$build_dir" --target dispatch_bandwidth_bench --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
printf 'Built: %s/experiments/dispatch_bandwidth/dispatch_bandwidth_bench\n' "$build_dir"
printf 'Before running, export LD_LIBRARY_PATH=%q:%q:"${LD_LIBRARY_PATH:-}"\n' \
    "$SHMEM_ROOT/lib" "$ASCEND_HOME_PATH/lib64"
