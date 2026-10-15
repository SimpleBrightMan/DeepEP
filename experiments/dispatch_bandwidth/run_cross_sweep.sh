#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON:-python3}
build_dir=${BUILD_DIR:-"$root/build-dispatch-bandwidth"}
cd -- "$root"

# Help and plans stay CPU-only, including when the machine has no CANN or plotting package.
for argument in "$@"; do
    if [[ "$argument" == --plan || "$argument" == --help || "$argument" == -h ]]; then
        exec "$python_bin" -m experiments.dispatch_bandwidth.sweep "$@"
    fi
done

if [[ -n "${ASCEND_ENV:-}" ]]; then
    source "$ASCEND_ENV"
fi
: "${ASCEND_HOME_PATH:?Source the matching CANN set_env.sh, or set ASCEND_ENV to its path}"
export SHMEM_ROOT=${SHMEM_ROOT:-"$root/third-party/shmem/install/shmem"}
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLBACKEND=Agg

# Do not install packages or launch another node automatically.
"$python_bin" -c 'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot'
PYTHON="$python_bin" BUILD_DIR="$build_dir" bash "$root/experiments/dispatch_bandwidth/build.sh"
exec "$python_bin" -m experiments.dispatch_bandwidth.sweep \
    --binary "$build_dir/experiments/dispatch_bandwidth/dispatch_bandwidth_bench" "$@"
