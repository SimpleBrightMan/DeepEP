#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHON="${PYTHON:-$(command -v python3)}"
export TASK_QUEUE_ENABLE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ${1:-} == --launch ]]; then
    # msopprof application replay re-enters here. Every pass owns fresh workers,
    # a fresh SHMEM endpoint/session and output directory. Keep LD_PRELOAD and
    # the other injection variables supplied by msopprof unchanged.
    run_dir=$(mktemp -d "${URMA_HW_OUTPUT:?}/runs/pass_XXXXXXXX")
    endpoint_port=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
    export DEEPEP_SHMEM_ENDPOINT="tcp://127.0.0.1:$endpoint_port"
    printf '[APP REPLAY] %s endpoint=%s\n' "$run_dir" "$DEEPEP_SHMEM_ENDPOINT"
    cd "$run_dir"
    exec "$PYTHON" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="$URMA_HW_EP" --max-restarts=0 \
        --log-dir "$run_dir/torchrun" --redirects 3 --tee 3 \
        "$script_dir/urma_hw_profile.py" --output-dir "$run_dir" \
        --warmup "${URMA_HW_WARMUP:-10}" --exp-repeat "${URMA_HW_EXP_REPEAT:-50}" \
        --bs "${URMA_HW_BS:-1024}" --cv-rows "${URMA_HW_CV_ROWS:-256}" \
        --routing "${URMA_HW_ROUTING:-random-balanced}" --timeout 600
fi

metric=${1:-PipeTimeline}
case "$metric" in
    PipeTimeline|InstrTimeline) ;;
    *) printf 'Usage: bash %s [PipeTimeline|InstrTimeline]\n' "$0" >&2; exit 2 ;;
esac
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
export SHMEM_ROOT="${SHMEM_ROOT:?Set SHMEM_ROOT to the SDK used by the installed wheel}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
URMA_HW_EP=$("$PYTHON" -c 'import os; a=os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(","); assert len(a) in (4,8,16) and len(set(a))==len(a) and all(x.isdecimal() for x in a), "Select 4,8,16 distinct allocated devices"; print(len(a))')
export URMA_HW_EP
msopprof="$ASCEND_HOME_PATH/bin/msopprof"
[[ -x "$msopprof" ]] || { printf 'Missing executable: %s\n' "$msopprof" >&2; exit 2; }
"$msopprof" --version
export URMA_HW_OUTPUT="${URMA_HW_OUTPUT:-$PWD/benchmark-runs/urma_hw_${metric}_$(date -u +%Y%m%d_%H%M%SZ)_$$}"
mkdir -p -- "$URMA_HW_OUTPUT"
export URMA_HW_OUTPUT=$(cd -- "$URMA_HW_OUTPUT" && pwd)
mkdir -- "$URMA_HW_OUTPUT/runs"

options=(--replay-mode=application --aic-metrics="$metric" --warm-up=0
    --kernel-name='*mega_moe_urma_kernel*' --launch-count=1 --kill=false
    --mstx=true --mstx-include=urma_capture --output="$URMA_HW_OUTPUT/profile")
if [[ $metric == InstrTimeline ]]; then
    options+=(--instr-timeline-pipe="${URMA_HW_PIPES:-cube|fixp|mte1|mte2|mte3|vector}")
fi
printf '[CAPTURE COMMAND] '
printf '%q ' "$msopprof" "${options[@]}" /bin/bash "$script_dir/run_urma_hw_profile.sh" --launch
printf '\n[OUTPUT] %s\n' "$URMA_HW_OUTPUT"
"$msopprof" "${options[@]}" /bin/bash "$script_dir/run_urma_hw_profile.sh" --launch \
    2>&1 | tee "$URMA_HW_OUTPUT/controller.log"
printf '\n[EXPORTS] Inspect the tool log and hardware timeline files:\n'
find "$URMA_HW_OUTPUT/profile" -type f \( -name 'visualize_data.bin' -o -name 'trace.json' \) -print
