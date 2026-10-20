#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
out=$root/aiv1-async-1020-r2
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/skew_recheck.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 300 9
test "$(cat "$out/edges.rc")" = 0
cd /tmp
export EXPERIMENT_BS=96 EXPERIMENT_ROUTING=skewed
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
# Extend the immediately preceding skewed control/async pair to A/B/B/A.
for mode in async control; do
  for attempt in {1..15}; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" = 8; then break; fi
    sleep 2
  done
  if test "$count" != 8; then echo 'NOT_IDLE: no worker launched' >&2; exit 3; fi
  if test "$mode" = control; then export MEGAMOE_PACKAGE="$root/aiv1-fair-1020/fair-package"
  else export MEGAMOE_PACKAGE="$root/aiv1-async-1020-v2/async-package"; fi
  label=aiv1-async-1020-r2/skewed-${mode}-b
  test ! -e "$root/$label.log"
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" aiv1-async-1020-r2/skewed-control
  grep '^RESULT' "$root/$label.log"
done
echo SKEW_RECHECK_COMPLETE
