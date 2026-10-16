#!/bin/bash
# Scoped local EP8 validation. SOURCE/REFERENCE/VENV/SHMEM_ROOT/CATLaSS env are supplied by the caller.
set -eEuo pipefail
trap 'rc=$?; echo "[FAILED] line=$LINENO status=$rc" >&2; exit "$rc"' ERR
: "${TASK_ROOT:?}" "${REFERENCE_VENV:?}" "${CANDIDATE_VENV:?}" "${MASTER_ADDR:?}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH PYTHONHOME ASCEND_LAUNCH_BLOCKING
if [[ "${SKIP_CPU:-0}" != 1 ]]; then
    cd "$TASK_ROOT/source"
    "$CANDIDATE_VENV/bin/python" -m pytest experiments/megamoe/tests -q -p no:cacheprovider \
        --basetemp "$TASK_ROOT/cpu-tests" > "$TASK_ROOT/cpu-tests.log" 2>&1
fi
check_idle() {
    local count attempt status
    # The launcher may exit before the driver removes the last context.
    # Wait briefly; never reclaim devices or terminate unrelated processes.
    for attempt in {1..10}; do
        status=$(npu-smi info)
        count=$(grep -c 'No running processes found in NPU' <<< "$status" || true)
        [[ "$count" != 8 ]] || return 0
        sleep 2
    done
    echo "[STOP] expected 8 idle NPUs, observed $count" >&2
    return 1
}
cd /tmp
if [[ "${SKIP_A8W8:-0}" != 1 ]]; then
for name in baseline candidate; do
    check_idle
    echo "[START] A8W8 $name $(date -Iseconds)"
    venv="$REFERENCE_VENV"
    [[ "$name" != candidate ]] || venv="$CANDIDATE_VENV"
    timeout --signal=TERM --kill-after=15s 300s "$venv/bin/python" -I -m torch.distributed.run \
        --nnodes 1 --nproc-per-node 8 --master-addr "$MASTER_ADDR" --master-port 29679 --max-restarts 0 \
        "$TASK_ROOT/check_a8w8.py" --output-dir "$TASK_ROOT/a8w8/$name" > "$TASK_ROOT/a8w8-$name.log" 2>&1
done
fi
for stage in baseline_a candidate_a candidate_b baseline_b; do
    check_idle
    venv="$REFERENCE_VENV"
    [[ "$stage" != candidate_* ]] || venv="$CANDIDATE_VENV"
    echo "[START] $stage $(date -Iseconds)"
    timeout --signal=TERM --kill-after=30s 900s "$venv/bin/python" -I -m torch.distributed.run \
        --nnodes 1 --nproc-per-node 8 --master-addr "$MASTER_ADDR" --master-port 29679 --max-restarts 0 \
        "$TASK_ROOT/benchmark_entry.py" --config deepseek_v4_pro --bs-values 32,96,256,1024,4096 \
        --mode perf --input fp8 --routing balanced --output-dir "$TASK_ROOT/runs/$stage" \
        > "$TASK_ROOT/$stage.log" 2>&1
    echo "[DONE] $stage $(date -Iseconds)"
done
