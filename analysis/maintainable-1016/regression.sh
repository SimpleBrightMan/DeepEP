#!/bin/bash
set -eEuo pipefail
trap 'rc=$?; echo "[FAILED] line=$LINENO status=$rc" >&2; exit "$rc"' ERR
: "${TASK_ROOT:?}" "${REFERENCE_VENV:?}" "${CANDIDATE_VENV:?}" "${MASTER_ADDR:?}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH PYTHONHOME ASCEND_LAUNCH_BLOCKING
cd /tmp
for kind in random shared; do
    config=deepseek_v4_pro
    batches=33,257,4104
    routing=random
    if [[ "$kind" == shared ]]; then
        config=deepseek_v4_pro_shared1
        batches=32,257
        routing=balanced
    fi
    for name in baseline candidate; do
        count=0
        for attempt in {1..10}; do
            status=$(npu-smi info)
            count=$(grep -c 'No running processes found in NPU' <<< "$status" || true)
            [[ "$count" != 8 ]] || break
            sleep 2
        done
        [[ "$count" == 8 ]] || { echo "[STOP] expected 8 idle NPUs, got $count" >&2; exit 1; }
        venv="$REFERENCE_VENV"
        [[ "$name" != candidate ]] || venv="$CANDIDATE_VENV"
        stage="${name}_${kind}"
        echo "[START] $stage $(date -Iseconds)"
        timeout --signal=TERM --kill-after=30s 900s "$venv/bin/python" -I -m torch.distributed.run \
            --nnodes 1 --nproc-per-node 8 --master-addr "$MASTER_ADDR" --master-port 29679 --max-restarts 0 \
            "$TASK_ROOT/benchmark_entry.py" --config "$config" --bs-values "$batches" \
            --mode perf --input fp8 --routing "$routing" --output-dir "$TASK_ROOT/runs/$stage" \
            > "$TASK_ROOT/$stage.log" 2>&1
        echo "[DONE] $stage $(date -Iseconds)"
    done
done
