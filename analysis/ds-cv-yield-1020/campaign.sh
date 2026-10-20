#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
out=$root/ds-cv-yield-1020-v1
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign-run.rc"; fi' EXIT
source "$root/env.sh"
test "$(cat "$out/build.rc")" = 0
exec 9>"$root/benchmark.lock"
flock -w 900 9
python -I -c "import sys,torch,torch_npu; sys.path.insert(0,'$out/cv-package'); from deep_ep_experimental.megamoe import _C; from deep_ep_experimental.megamoe.timer import protocol as p; assert _C.timer_numel==p.TOTAL_BUFFER_SIZE; print(_C.__file__,_C.timer_numel)" > "$out/abi-probe-retry1.log" 2>&1
cd /tmp
export EXPERIMENT_BS=512 EXPERIMENT_ROUTING=balanced
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
for entry in control:control-a cv:candidate-a cv:candidate-b control:control-b; do
  mode=${entry%:*}; label=ds-cv-yield-1020-v1/${entry#*:}
  report=$(timeout 20s npu-smi info)
  count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
  if test "$count" != 8; then echo 'NOT_IDLE: no worker launched' >&2; exit 3; fi
  test ! -e "$root/$label.log"
  if test "$mode" = control; then export MEGAMOE_PACKAGE="$root/aiv1-async-1020-v2/async-package"
  else export MEGAMOE_PACKAGE="$out/cv-package"; fi
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" dashboard1020_bs512_ours_a
  grep '^RESULT' "$root/$label.log"
done
echo CAMPAIGN_COMPLETE
