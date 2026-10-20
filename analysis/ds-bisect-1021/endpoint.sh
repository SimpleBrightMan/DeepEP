#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-bisect-1021
out=$campaign/endpoint
mkdir "$out"
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 900 9
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
cd /tmp
idle() {
  local report count
  report=$(timeout 20s npu-smi info)
  count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
  if test "$count" != 8; then printf '%s\n' "$report" > "$out/not-idle.log"; return 3; fi
}
for bs in 512 1024; do
  export EXPERIMENT_BS=$bs EXPERIMENT_ROUTING=balanced
  for item in l6:a l12:a reference:a endpoint:a endpoint:b reference:b l12:b l6:b; do
    idle
    method=${item%:*}; round=${item#*:}; label=ds-bisect-1021/endpoint/bs$bs-$method-$round
    test ! -e "$root/$label.log"
    echo "START $label $(date -Is)"
    if test "$method" = l6 || test "$method" = l12; then
      export MEGAMOE_PACKAGE="$root/ds-ladder-1020/$method/package"
      runner=("$root/ours.py" "$label")
      if test "$bs" = 512; then baseline=dashboard1020_bs512_ours_a
      else baseline=ds-ladder-1020/screen/bs1024/l0; fi
    elif test "$method" = reference; then
      export DG_JIT_CACHE_DIR="$root/cache_dashboard_ds_optimized_1020"
      runner=("$root/dashboard_ds_runner_1020.py" optimized "$label")
      baseline=dashboard1020_bs${bs}_baseline_a
    else
      export DG_JIT_CACHE_DIR="$campaign/d0/jit-cache"
      runner=("$campaign/ds_runner.py" --root "$root" --node d0 --label "$label" --bs "$bs")
      if test "$round" = a; then runner+=(--trace); fi
      baseline=dashboard1020_bs${bs}_baseline_a
    fi
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
      --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "${runner[@]}" > "$root/$label.log" 2>&1
    python "$root/check_label.py" "$label" "$baseline"
    grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
  done
done
echo ENDPOINT_CAMPAIGN_COMPLETE
