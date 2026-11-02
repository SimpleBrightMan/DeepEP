#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:?source the CANN environment first}"
export SHMEM_ROOT="${SHMEM_ROOT:-$ROOT/../../third-party/shmem/install/shmem}"
test -f "$SHMEM_ROOT/include/shmem.h"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSHMEM_ROOT="$SHMEM_ROOT"
cmake --build "$ROOT/build" --parallel 2
