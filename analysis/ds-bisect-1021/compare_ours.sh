#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-bisect-1021
node=${1:?node required}
parent=${2:?parent required}
out=$campaign/$node
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 900 9
test "$(cat "$out/build.rc")" = 0
if test -d "$campaign/$parent/package"; then parent_package="$campaign/$parent/package"
else parent_package="$root/ds-ladder-1020/$parent/package"; fi
test -d "$parent_package/deep_ep_experimental/megamoe"
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
cd /tmp
for bs in 512 1024; do
  export EXPERIMENT_BS=$bs EXPERIMENT_ROUTING=balanced
  for item in parent:a candidate:a ds:a ds:b candidate:b parent:b; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" != 8; then printf '%s\n' "$report" > "$out/not-idle.log"; exit 3; fi
    method=${item%:*}; round=${item#*:}; label=ds-bisect-1021/$node/bs$bs-$method-$round
    test ! -e "$root/$label.log"
    unset MEGAMOE_TRACE
    if test "$method" = ds; then
      export DG_JIT_CACHE_DIR="$campaign/d0/jit-cache"
      runner=("$campaign/ds_runner.py" --root "$root" --node d0 --label "$label" --bs "$bs")
      baseline=dashboard1020_bs${bs}_baseline_a
    else
      if test "$method" = parent; then export MEGAMOE_PACKAGE="$parent_package"
      else export MEGAMOE_PACKAGE="$out/package"; fi
      if test "$method:$round" = candidate:a; then export MEGAMOE_TRACE=1; fi
      runner=("$root/ours.py" "$label")
      if test "$bs" = 512; then baseline=dashboard1020_bs512_ours_a
      else baseline=ds-ladder-1020/screen/bs1024/l0; fi
    fi
    echo "START $label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
      --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "${runner[@]}" > "$root/$label.log" 2>&1
    python "$root/check_label.py" "$label" "$baseline"
    grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
  done
done
echo COMPUTE_CAMPAIGN_COMPLETE
