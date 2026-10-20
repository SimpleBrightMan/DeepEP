#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-bisect-1021
node=${1:-c4}
out=$campaign/screen
mkdir "$out"
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
for attempt in {1..900}; do
  if test -f "$campaign/$node/campaign.rc"; then break; fi
  sleep 5
done
test "$(cat "$campaign/$node/campaign.rc")" = 0
exec 9>"$root/benchmark.lock"
flock -w 900 9
cd /tmp
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
idle() {
  local report count
  report=$(timeout 20s npu-smi info)
  count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
  if test "$count" != 8; then printf '%s\n' "$report" > "$out/not-idle.log"; return 3; fi
}
ours() {
  local method=$1 label=$2 reference=$3
  idle
  if test "$method" = l6; then export MEGAMOE_PACKAGE="$root/ds-ladder-1020/l6/package"
  else export MEGAMOE_PACKAGE="$campaign/$node/package"; fi
  test ! -e "$root/$label.log"
  echo "START $method $label $(date -Is)"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/ours.py" "$label" > "$root/$label.log" 2>&1
  python "$root/check_label.py" "$label" "$reference"
  grep '^RESULT' "$root/$label.log"
}
export EXPERIMENT_ROUTING=balanced
for bs in 32 96 256 512 1024 2048 4096; do
  export EXPERIMENT_BS=$bs
  label=ds-bisect-1021/screen/bs$bs
  mkdir "$root/$label"
  if test "$bs" = 512; then reference=dashboard1020_bs512_ours_a
  else reference=ds-ladder-1020/screen/bs$bs/l0; fi
  ours l6 "$label/l6" "$reference"
  ours candidate "$label/candidate" "$label/l6"
  idle
  export DG_JIT_CACHE_DIR="$root/cache_dashboard_ds_optimized_1020"
  timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
    --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "$root/dashboard_ds_runner_1020.py" optimized "$label/ds" > "$root/$label/ds.log" 2>&1
  python "$root/check_label.py" "$label/ds" dashboard1020_bs${bs}_baseline_a
  grep '^RESULT' "$root/$label/ds.log"
done
export EXPERIMENT_BS=96
for routing in random skewed; do
  export EXPERIMENT_ROUTING=$routing
  label=ds-bisect-1021/screen/$routing
  mkdir "$root/$label"
  ours l6 "$label/l6" ds-ladder-1020/screen/$routing/l0
  ours candidate "$label/candidate" "$label/l6"
done
echo SAME_BINARY_SCREEN_COMPLETE
