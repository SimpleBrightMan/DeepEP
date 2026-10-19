#!/usr/bin/env bash
# Copyright (c) 2026, SimpleBrightMan
#
# Single-host 16-card MTE All-to-All (EP16, 4-card physical groups).
#
# Run this on the 16-card target after confirming the cards are idle with
# `npu-smi info`. The script refuses to start when it sees an existing NPU
# process unless ALLOW_BUSY=1 is set.
#
# Example:
#   CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
#   SHMEM_ROOT=/path/to/shmem/install/shmem \
#   PYTHON=/path/to/venv/bin/python \
#   bash benchmarks/alltoall_mte/run_16cards.sh --tokens 1024,2048,4096
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOKENS=${TOKENS:-2048}
TOKEN_BYTES=${TOKEN_BYTES:-3584}
GROUP_SIZE=${GROUP_SIZE:-4}
WORLD_SIZE=${WORLD_SIZE:-16}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
CORES=${CORES:-64}
TIMEOUT=${TIMEOUT:-1800}
MASTER=${MASTER:-$(hostname -I | awk '{print $1}')}
PORT=${SHMEM_PORT:-19091}
OUTPUT=${OUTPUT:-$ROOT/benchmarks/alltoall_mte/results/run_$(date +%Y%m%d-%H%M%S)}
PYTHON=${PYTHON:-python3}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
SHMEM_ROOT=${SHMEM_ROOT:-}

usage() {
    echo "usage: [SHMEM_ROOT=...] [PYTHON=...] bash run_16cards.sh [--tokens 1024,2048,4096]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens) TOKENS=$2; shift 2 ;;
        --token-bytes) TOKEN_BYTES=$2; shift 2 ;;
        --group-size) GROUP_SIZE=$2; shift 2 ;;
        --warmup) WARMUP=$2; shift 2 ;;
        --iters) ITERS=$2; shift 2 ;;
        --cores) CORES=$2; shift 2 ;;
        --master) MASTER=$2; shift 2 ;;
        --output) OUTPUT=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ "$WORLD_SIZE" == "16" ]] || { echo "this launcher targets exactly 16 ranks" >&2; exit 2; }
[[ -n "$SHMEM_ROOT" ]] || { echo "SHMEM_ROOT must point at the installed SHMEM SDK" >&2; exit 2; }
[[ -f "$CANN_ENV" ]] || { echo "CANN_ENV not found: $CANN_ENV" >&2; exit 2; }
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "PYTHON not found on PATH: $PYTHON" >&2
    exit 2
fi

source "$CANN_ENV"
[[ ! -f "$SHMEM_ROOT/set_env.sh" ]] || source "$SHMEM_ROOT/set_env.sh"
export SHMEM_ROOT
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DEEPEP_SHMEM_ENDPOINT="tcp://$MASTER:$PORT"
export DEEPEP_SHMEM_DENSE_QPS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

busy=$(
    npu-smi info 2>/dev/null | sed -n '/Process id/,$p' |
        grep -E '^\| [0-9]+ +\| [0-9]+' || true
)
if [[ -n "$busy" && "${ALLOW_BUSY:-0}" != "1" ]]; then
    echo "[alltoall-mte] refusing to start: NPU processes are present" >&2
    echo "$busy" >&2
    echo "[alltoall-mte] re-check with 'npu-smi info'; set ALLOW_BUSY=1 only if these are yours" >&2
    exit 3
fi

echo "[alltoall-mte] master=$MASTER world=$WORLD_SIZE tokens=$TOKENS token_bytes=$TOKEN_BYTES"
echo "[alltoall-mte] output=$OUTPUT"
mkdir -p "$OUTPUT"
npu-smi info >"$OUTPUT/npu-before.log" 2>&1 || true

pids=()
for rank in $(seq 0 15); do
    RANK="$rank" WORLD_SIZE="$WORLD_SIZE" LOCAL_RANK="$rank" \
        timeout --signal=TERM --kill-after=30s "$TIMEOUT" \
        "$PYTHON" -u -m benchmarks.alltoall_mte.bench_alltoall \
        --rank "$rank" --world-size "$WORLD_SIZE" --local-rank "$rank" --device "$rank" \
        --token-bytes "$TOKEN_BYTES" --tokens "$TOKENS" --group-size "$GROUP_SIZE" \
        --cores "$CORES" --warmup "$WARMUP" --iters "$ITERS" --output-dir "$OUTPUT" \
        >"$OUTPUT/rank_$(printf '%02d' "$rank").log" 2>&1 &
    pids+=("$!")
    if [[ "$rank" == "0" ]]; then
        sleep 5
    fi
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
npu-smi info >"$OUTPUT/npu-after.log" 2>&1 || true
if [[ "$status" != "0" ]]; then
    echo "[alltoall-mte] at least one rank failed; tail of logs:" >&2
    for log in "$OUTPUT"/rank_*.log; do
        echo "===== $log" >&2
        tail -n 20 "$log" >&2 || true
    done
    exit 1
fi

"$PYTHON" -m benchmarks.alltoall_mte.summarize --input "$OUTPUT" \
    --title "single-host EP16 group4 ${TOKEN_BYTES}B tokens=${TOKENS}"
