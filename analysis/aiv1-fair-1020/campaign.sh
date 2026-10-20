#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
out=$root/aiv1-fair-1020
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 600 9
test "$(cat "$root/aiv1-resident-1020-v3/campaign.rc")" = 0
python "$out/prepare.py"
python -I -m pytest -q "$out/fair-source/experiments/megamoe/tests/test_aiv1_resident_ub.py" \
  "$out/fair-source/experiments/megamoe/tests/test_aiv1_fair_scheduler.py"
cd /tmp
export EXPERIMENT_BS=512 EXPERIMENT_ROUTING=balanced
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
for entry in resident:control-a fair:candidate-a fair:candidate-b resident:control-b; do
  mode=${entry%:*}; label=aiv1-fair-1020/${entry#*:}
  for attempt in {1..15}; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" = 8; then break; fi
    sleep 2
  done
  if test "$count" != 8; then echo 'NOT_IDLE: no worker launched' >&2; exit 3; fi
  test ! -e "$root/$label.log"
  if test "$mode" = resident; then export MEGAMOE_PACKAGE="$root/aiv1-resident-1020-v3/resident-package"
  else export MEGAMOE_PACKAGE="$out/fair-package"; fi
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29996 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" dashboard1020_bs512_ours_a
  grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
done
echo CAMPAIGN_COMPLETE
