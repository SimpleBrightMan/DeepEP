#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
out=$root/aiv1-async-1020-r2
build=$root/aiv1-async-1020-v2
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/diagnose.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 600 9
test "$(cat "$out/campaign.rc")" = 0
# Only exporter metadata/tests changed after the build. C++/ABI source snapshot is immutable.
python "$out/parser_overlay.py"
python -I -m pytest -q "$out/parser-check/experiments/megamoe/tests/test_timer.py" \
  "$out/parser-check/experiments/megamoe/tests/test_breakdown.py" > "$out/parser-tests.log" 2>&1
wait_idle() {
  local report count
  for attempt in {1..15}; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" = 8; then return 0; fi
    sleep 2
  done
  echo 'NOT_IDLE: no new worker launched' >&2; return 3
}
cd /tmp
export EXPERIMENT_BS=512 EXPERIMENT_ROUTING=balanced
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
for mode in control async; do
  if test "$mode" = control; then export MEGAMOE_PACKAGE="$root/aiv1-fair-1020/fair-package"
  else export MEGAMOE_PACKAGE="$build/async-package"; fi
  wait_idle
  export MEGAMOE_TRACE=1
  label=aiv1-async-1020-r2/${mode}-trace
  test ! -e "$root/$label.log"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" dashboard1020_bs512_ours_a
  grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
  unset MEGAMOE_TRACE
  wait_idle
  export MEGAMOE_PMU=1
  label=aiv1-async-1020-r2/pmu_${mode}_PipeUtilization
  test ! -e "$root/$label.log"
  timeout -k 90s 600s msprof --output="$root/$label/profiler" --task-time=on --task-block=on \
    --aic-metrics=PipeUtilization --type=text python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  grep -q DIAGNOSTIC_COMPLETE "$root/$label.log"
  unset MEGAMOE_PMU
done
echo DIAGNOSE_COMPLETE
