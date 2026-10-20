#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-bisect-1021
node=${1:?node required}
parent=${2:-d0}
out=$campaign/$node
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 900 9
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
cd /tmp
for bs in 512 1024; do
  for item in parent:a candidate:a candidate:b parent:b; do
    report=$(timeout 20s npu-smi info)
    count=$(printf '%s\n' "$report" | grep -c 'No running processes found in NPU' || true)
    if test "$count" != 8; then printf '%s\n' "$report" > "$out/not-idle.log"; exit 3; fi
    method=${item%:*}; round=${item#*:}; label=ds-bisect-1021/$node/bs$bs-$method-$round
    if test "$method" = parent; then selected=$parent; else selected=$node; fi
    export DG_JIT_CACHE_DIR="$campaign/$selected/jit-cache"
    runner=("$campaign/$selected/source/analysis/ds-bisect-1021/ds_runner.py" --root "$root" --node "$selected" --label "$label" --bs "$bs")
    if test "$method:$round" = candidate:a; then runner+=(--trace); fi
    test ! -e "$root/$label.log"
    echo "START $label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
      --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "${runner[@]}" > "$root/$label.log" 2>&1
    python "$root/check_label.py" "$label" dashboard1020_bs${bs}_baseline_a
    grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
  done
done
echo DS_REVERSE_ABLATION_COMPLETE
