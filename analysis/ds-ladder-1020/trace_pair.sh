#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-ladder-1020
out=$campaign/trace-pair
mkdir "$out"
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 900 9
export EXPERIMENT_BS=1024 EXPERIMENT_ROUTING=balanced
unset MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
cd /tmp
for method in l6 ds; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" != 8; then echo 'NOT_IDLE: no new worker launched' >&2; exit 3; fi
    label=ds-ladder-1020/trace-pair/$method
    test ! -e "$root/$label.log"
    if test "$method" = l6; then
        export MEGAMOE_TRACE=1 MEGAMOE_PACKAGE="$campaign/l6/package"
        runner=("$root/ours.py" "$label")
        reference=ds-ladder-1020/screen/bs1024/l0
    else
        unset MEGAMOE_TRACE
        export DG_JIT_CACHE_DIR="$root/cache_dashboard_ds_optimized_1020"
        runner=("$root/sota_ds_trace_1020.py" optimized "$label")
        reference=dashboard1020_bs1024_baseline_a
    fi
    echo "START $label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
        --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "${runner[@]}" > "$root/$label.log" 2>&1
    python "$root/check_label.py" "$label" "$reference"
    grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
done
echo TRACE_PAIR_COMPLETE
