#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
name=cv-adaptive-1020-v1
out=$root/$name
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
test "$(cat "$out/build.rc")" = 0
exec 9>"$root/benchmark.lock"
flock -w 900 9
cd /tmp
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
run_case() {
  local bs=$1 mode=$2 tag=$3 route=$4 reference=$5 report count label
  for attempt in {1..30}; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" = 8; then break; fi
    sleep 2
  done
  if test "$count" != 8; then printf '%s\n' "$report" > "$out/not-idle-$tag.log"; echo 'NOT_IDLE: no worker launched' >&2; exit 3; fi
  export EXPERIMENT_BS=$bs EXPERIMENT_ROUTING=$route
  if test "$mode" = control; then export MEGAMOE_PACKAGE="$root/aiv1-async-1020-v2/async-package"
  else export MEGAMOE_PACKAGE="$out/cv-package"; fi
  label=$name/$tag
  test ! -e "$root/$label.log"
  echo "START $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29997 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" "$reference"
  grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
}
for item in control:control-a adaptive:adaptive-a adaptive:adaptive-b control:control-b; do
  run_case 512 "${item%:*}" "${item#*:}" balanced dashboard1020_bs512_ours_a
done
for bs in 32 96 256 1024 2048 4096; do
  for mode in control adaptive; do
    run_case "$bs" "$mode" "bs${bs}-${mode}" balanced "$name/bs${bs}-control"
  done
done
for route in random skewed; do
  run_case 96 adaptive "$route-regression" "$route" "ds-cv-stage-1020-v1/$route-bs96-control"
done
export MEGAMOE_TRACE=1
run_case 512 adaptive adaptive-trace balanced dashboard1020_bs512_ours_a
unset MEGAMOE_TRACE
echo CAMPAIGN_COMPLETE
