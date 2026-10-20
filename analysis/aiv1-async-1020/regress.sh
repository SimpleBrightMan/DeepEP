#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
out=$root/aiv1-async-1020-r2
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/regress.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 900 9
test "$(cat "$out/diagnose.rc")" = 0
cd /tmp
export EXPERIMENT_ROUTING=balanced
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
for bs in 96 1024 4096; do
  export EXPERIMENT_BS=$bs
  for mode in control async; do
    for attempt in {1..15}; do
      report=$(timeout 20s npu-smi info)
      count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
      if test "$count" = 8; then break; fi
      sleep 2
    done
    if test "$count" != 8; then echo 'NOT_IDLE: no worker launched' >&2; exit 3; fi
    if test "$mode" = control; then export MEGAMOE_PACKAGE="$root/aiv1-fair-1020/fair-package"
    else export MEGAMOE_PACKAGE="$root/aiv1-async-1020-v2/async-package"; fi
    label=aiv1-async-1020-r2/bs${bs}-${mode}
    test ! -e "$root/$label.log"
    echo "START $label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
      --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
    python "$root/check_label.py" "$label" aiv1-async-1020-r2/bs${bs}-control
    grep '^RESULT' "$root/$label.log"
  done
done
echo REGRESS_COMPLETE
