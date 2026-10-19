#!/usr/bin/env bash
# Copyright (c) 2026, SimpleBrightMan
#
# Two-node EP16 (8 NPUs per node, 4-card physical groups) MTE All-to-All run.
#
# Node 0 runs ranks 0..7 and starts ranks 8..15 over SSH on the peer node.
# Every rank writes one JSON report; the peer reports are copied back and the
# aggregate writes summary.{json,csv,md} plus a Markdown table on stdout.
#
# Example:
#   PEER=177.2.22.1 CANN_ENV=/home/pkg/b080/cann-9.2.0/set_env.sh \
#   SHMEM_ROOT=/path/to/shmem/install/shmem \
#   PYTHON=/path/to/venv/bin/python \
#   bash benchmarks/alltoall_mte/run_2nodes.sh --tokens 1024,2048,4096
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOKENS=${TOKENS:-2048}
TOKEN_BYTES=${TOKEN_BYTES:-3584}
GROUP_SIZE=${GROUP_SIZE:-4}
WORLD_SIZE=${WORLD_SIZE:-16}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
CORES=${CORES:-64}
TIMEOUT=${TIMEOUT:-1800}
MASTER=${MASTER:-$(hostname -I | awk '{print $1}')}
PEER=${PEER:-}
PORT=${SHMEM_PORT:-19091}
OUTPUT=${OUTPUT:-$ROOT/benchmarks/alltoall_mte/results/run_$(date +%Y%m%d-%H%M%S)}
PYTHON=${PYTHON:-python3}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
SHMEM_ROOT=${SHMEM_ROOT:-}

usage() {
    echo "usage: PEER=<peer-host> [SHMEM_ROOT=...] [PYTHON=...] bash run_2nodes.sh [--tokens 1024,2048,4096]" >&2
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
        --peer) PEER=$2; shift 2 ;;
        --master) MASTER=$2; shift 2 ;;
        --output) OUTPUT=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -n "$PEER" ]] || usage
[[ -n "$SHMEM_ROOT" && -d "$SHMEM_ROOT/lib" ]] || {
    echo "SHMEM_ROOT must point at the installed SHMEM SDK" >&2
    exit 2
}
[[ -f "$CANN_ENV" ]] || { echo "CANN_ENV not found: $CANN_ENV" >&2; exit 2; }
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "PYTHON not found on PATH: $PYTHON" >&2
    exit 2
fi
[[ "$WORLD_SIZE" == "16" && "$NPROC_PER_NODE" == "8" ]] || {
    echo "this launcher targets EP16 over 2x8 NPUs" >&2
    exit 2
}

source "$CANN_ENV"
[[ ! -f "$SHMEM_ROOT/set_env.sh" ]] || source "$SHMEM_ROOT/set_env.sh"
export SHMEM_ROOT
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DEEPEP_SHMEM_ENDPOINT="tcp://$MASTER:$PORT"
export DEEPEP_SHMEM_DENSE_QPS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT"
npu-smi info >"$OUTPUT/npu-before-node0.log" 2>&1 || true
REMOTE_ENV="source '$CANN_ENV' && export SHMEM_ROOT='$SHMEM_ROOT' && export LD_LIBRARY_PATH='$SHMEM_ROOT/lib':\"\${LD_LIBRARY_PATH:-}\" && export PYTHONPATH='$ROOT' DEEPEP_SHMEM_ENDPOINT='tcp://$MASTER:$PORT' DEEPEP_SHMEM_DENSE_QPS=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"

check_busy() {
    local where=$1 text=$2
    local busy
    busy=$(printf '%s\n' "$text" | sed -n '/Process id/,$p' | grep -E '^\| [0-9]+ +\| [0-9]+' || true)
    if [[ -n "$busy" && "${ALLOW_BUSY:-0}" != "1" ]]; then
        echo "[alltoall-mte] refusing to start on $where: NPU processes are present" >&2
        echo "$busy" >&2
        echo "[alltoall-mte] re-check with 'npu-smi info'; set ALLOW_BUSY=1 only if these are yours" >&2
        exit 3
    fi
}

check_busy node0 "$(npu-smi info 2>&1 || true)"
peer_status=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" "npu-smi info 2>&1" 2>&1 || true)
check_busy "$PEER" "$peer_status"
printf '%s\n' "$peer_status" >"$OUTPUT/npu-before-peer.log"

launch_rank() {
    local rank=$1 local_rank=$2
    RANK="$rank" WORLD_SIZE="$WORLD_SIZE" LOCAL_RANK="$local_rank" \
        timeout --signal=TERM --kill-after=30s "$TIMEOUT" \
        "$PYTHON" -u -m benchmarks.alltoall_mte.bench_alltoall \
        --rank "$rank" --world-size "$WORLD_SIZE" --local-rank "$local_rank" --device "$local_rank" \
        --token-bytes "$TOKEN_BYTES" --tokens "$TOKENS" --group-size "$GROUP_SIZE" \
        --cores "$CORES" --warmup "$WARMUP" --iters "$ITERS" --output-dir "$OUTPUT" \
        >"$OUTPUT/rank_$(printf '%02d' "$rank").log" 2>&1
}

echo "[alltoall-mte] master=$MASTER peer=$PEER world=$WORLD_SIZE tokens=$TOKENS token_bytes=$TOKEN_BYTES"
echo "[alltoall-mte] output=$OUTPUT"

pids=()
# Rank 0 creates the SHMEM store; give it a short head start, then let every
# client retry against the same endpoint.
launch_rank 0 0 &
pids+=("$!")
sleep 3

for local_rank in 1 2 3 4 5 6 7; do
    launch_rank "$local_rank" "$local_rank" &
    pids+=("$!")
done

for local_rank in 0 1 2 3 4 5 6 7; do
    rank=$((8 + local_rank))
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
        "cd '$ROOT' && mkdir -p '$OUTPUT' && $REMOTE_ENV && \
         RANK='$rank' WORLD_SIZE='$WORLD_SIZE' LOCAL_RANK='$local_rank' \
         timeout --signal=TERM --kill-after=30s '$TIMEOUT' '$PYTHON' -u -m benchmarks.alltoall_mte.bench_alltoall \
         --rank '$rank' --world-size '$WORLD_SIZE' --local-rank '$local_rank' --device '$local_rank' \
         --token-bytes '$TOKEN_BYTES' --tokens '$TOKENS' --group-size '$GROUP_SIZE' \
         --cores '$CORES' --warmup '$WARMUP' --iters '$ITERS' --output-dir '$OUTPUT'" \
        >"$OUTPUT/rank_$(printf '%02d' "$rank").log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" != "0" ]]; then
    echo "[alltoall-mte] at least one rank failed; tail of logs:" >&2
    for log in "$OUTPUT"/rank_*.log; do
        echo "===== $log" >&2
        tail -n 20 "$log" >&2 || true
    done
    exit 1
fi

npu-smi info >"$OUTPUT/npu-after-node0.log" 2>&1 || true
ssh -o BatchMode=yes "$PEER" "npu-smi info 2>&1" >"$OUTPUT/npu-after-peer.log" 2>&1 || true

# Peer node does not share this filesystem.
scp -q -o BatchMode=yes "$PEER:$OUTPUT/rank_*.json" "$OUTPUT/"
"$PYTHON" -m benchmarks.alltoall_mte.summarize --input "$OUTPUT" \
    --title "EP16 group4 ${TOKEN_BYTES}B tokens=${TOKENS}"
