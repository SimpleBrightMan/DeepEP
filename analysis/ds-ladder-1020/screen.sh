#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-ladder-1020
out=$campaign/screen
mkdir "$out"
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
# 只使用两个完整通过的不可变SO；没有逐shape切换候选或运行时修改源文件。
for node in l6 l11; do
    for attempt in {1..240}; do
        if test -f "$campaign/$node/campaign.rc"; then break; fi
        sleep 5
    done
    test "$(cat "$campaign/$node/campaign.rc")" = 0
done
exec 9>"$root/benchmark.lock"
flock -w 900 9
cd /tmp
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
wait_idle() {
    local report count
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" != 8; then
        printf '%s\n' "$report" > "$out/not-idle.log"
        echo 'NOT_IDLE: stopped without launching another worker' >&2
        return 3
    fi
}
run_ours() {
    local method=$1 label=$2 reference=${3:-}
    wait_idle
    if test "$method" = l0; then export MEGAMOE_PACKAGE="$root/cv-adaptive-1020-v1/cv-package"
    else export MEGAMOE_PACKAGE="$campaign/$method/package"; fi
    test ! -e "$root/$label.log"
    unset MEGAMOE_TRACE
    if test "$method:$EXPERIMENT_BS:$EXPERIMENT_ROUTING" = l11:1024:balanced; then export MEGAMOE_TRACE=1; fi
    echo "START $method $label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
        --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
    if test -n "$reference"; then python "$root/check_label.py" "$label" "$reference"; fi
    grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
}
export EXPERIMENT_ROUTING=balanced
for bs in 32 96 256 1024 2048 4096; do
    export EXPERIMENT_BS=$bs
    case_label=ds-ladder-1020/screen/bs$bs
    mkdir "$root/$case_label"
    run_ours l0 "$case_label/l0"
    run_ours l6 "$case_label/l6" "$case_label/l0"
    run_ours l11 "$case_label/l11" "$case_label/l0"
    wait_idle
    unset MEGAMOE_TRACE
    export DG_JIT_CACHE_DIR="$root/cache_dashboard_ds_optimized_1020"
    test ! -e "$root/$case_label/ds.log"
    echo "START DS $case_label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
        --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/dashboard_ds_runner_1020.py" optimized "$case_label/ds" > "$root/$case_label/ds.log" 2>&1
    python "$root/check_label.py" "$case_label/ds" "dashboard1020_bs${bs}_baseline_a"
    grep '^RESULT' "$root/$case_label/ds.log"
done
export EXPERIMENT_BS=96
for routing in random skewed; do
    export EXPERIMENT_ROUTING=$routing
    case_label=ds-ladder-1020/screen/$routing
    mkdir "$root/$case_label"
    run_ours l0 "$case_label/l0"
    run_ours l6 "$case_label/l6" "$case_label/l0"
    run_ours l11 "$case_label/l11" "$case_label/l0"
done
export EXPERIMENT_BS=512 EXPERIMENT_ROUTING=balanced
wait_idle
unset MEGAMOE_TRACE
export DG_JIT_CACHE_DIR="$root/cache_dashboard_ds_optimized_1020"
label=ds-ladder-1020/screen/ds-trace-bs512
test ! -e "$root/$label.log"
timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/sota_ds_trace_1020.py" optimized "$label" > "$root/$label.log" 2>&1
python "$root/check_label.py" "$label" dashboard1020_bs512_baseline_a
grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
echo SCREEN_COMPLETE
