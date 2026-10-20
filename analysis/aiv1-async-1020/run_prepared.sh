#!/bin/bash
# Resume only the test phase after the recorded wrapper configure/link repairs.
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
build=$root/aiv1-async-1020-v2
out=$root/aiv1-async-1020-r2
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 300 9
test -f "$build/identity.json"
cp "$build/identity.json" "$out/identity.json"
python -I -m pytest -q "$build/async-source/experiments/megamoe/tests/test_aiv1_resident_ub.py" \
  "$build/async-source/experiments/megamoe/tests/test_aiv1_fair_scheduler.py" \
  "$build/async-source/experiments/megamoe/tests/test_dispatch.py" \
  "$build/async-source/experiments/megamoe/tests/test_aiv1_async_events.py"
python -I -c "import sys; sys.path.insert(0, '$build/async-package'); import torch, torch_npu; from deep_ep_experimental.megamoe import _C; from deep_ep_experimental.megamoe.timer import protocol as p; assert _C.timer_numel == p.TOTAL_BUFFER_SIZE; print('ABI_PROBE', _C.__file__, _C.timer_numel)" > "$out/abi-probe.log" 2>&1
cd /tmp
export EXPERIMENT_BS=512 EXPERIMENT_ROUTING=balanced
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
for entry in fair:control-a async:candidate-a async:candidate-b fair:control-b; do
  mode=${entry%:*}; label=aiv1-async-1020-r2/${entry#*:}
  for attempt in {1..15}; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" = 8; then break; fi
    sleep 2
  done
  if test "$count" != 8; then echo 'NOT_IDLE: no worker launched' >&2; exit 3; fi
  test ! -e "$root/$label.log"
  if test "$mode" = fair; then export MEGAMOE_PACKAGE="$root/aiv1-fair-1020/fair-package"
  else export MEGAMOE_PACKAGE="$build/async-package"; fi
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" dashboard1020_bs512_ours_a
  grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
done
echo CAMPAIGN_COMPLETE
