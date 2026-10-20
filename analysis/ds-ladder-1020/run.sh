#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-ladder-1020
node=${1:?node required}
parent=${2:?parent node or l0 required}
out=$campaign/$node
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
test "$(cat "$out/build.rc")" = 0
exec 9>"$root/benchmark.lock"
flock -w 900 9
cd /tmp
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
export EXPERIMENT_BS=512 EXPERIMENT_ROUTING=balanced
wait_idle() {
  local report count
  for attempt in {1..30}; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" = 8; then return 0; fi
    sleep 2
  done
  printf '%s\n' "$report" > "$out/not-idle.log"
  echo 'NOT_IDLE: no new worker launched' >&2
  return 3
}
run_ours() {
  local mode=$1 tag=$2 label=ds-ladder-1020/$node/$2
  wait_idle
  if test "$mode" = l0; then export MEGAMOE_PACKAGE="$root/cv-adaptive-1020-v1/cv-package"
  elif test "$mode" = parent; then export MEGAMOE_PACKAGE="$campaign/$parent/package"
  else export MEGAMOE_PACKAGE="$out/package"; fi
  test ! -e "$root/$label.log"
  if test "$mode:$tag" = candidate:candidate-a; then export MEGAMOE_TRACE=1; else unset MEGAMOE_TRACE; fi
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" dashboard1020_bs512_ours_a
  grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
}
run_ds() {
  local label=ds-ladder-1020/$node/ds-$1
  wait_idle
  unset MEGAMOE_TRACE
  export DG_JIT_CACHE_DIR="$root/cache_dashboard_ds_optimized_1020"
  test ! -e "$root/$label.log"
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/dashboard_ds_runner_1020.py" optimized "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" dashboard1020_bs512_baseline_a
  grep '^RESULT' "$root/$label.log"
}
run_ours l0 l0-a
if test "$parent" != l0; then run_ours parent parent-a; fi
run_ours candidate candidate-a
run_ds a
run_ds b
run_ours candidate candidate-b
if test "$parent" != l0; then run_ours parent parent-b; fi
run_ours l0 l0-b
echo CAMPAIGN_COMPLETE
