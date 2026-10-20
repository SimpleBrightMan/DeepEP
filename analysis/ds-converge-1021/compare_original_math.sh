#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-converge-1021
out=$campaign/r31
source "$root/env.sh"
exec 9>"$root/benchmark.lock"
flock -w 7200 9
unset MEGAMOE_TRACE MEGAMOE_PMU DS_TRACE TILEPACK_EXTERNAL_PMU
export DG_JIT_CACHE_DIR="$root/ds-bisect-1021/d0/jit-cache"
cd /tmp
for bs in 512 1024; do
  export EXPERIMENT_BS=$bs EXPERIMENT_ROUTING=balanced
  for item in parent:a candidate:a ds:a ds:b candidate:b parent:b; do
    idle=$(timeout 20s npu-smi info)
    if test "$(printf '%s\n' "$idle" | grep -c 'No running processes found in NPU' || true)" != 8; then
      printf '%s\n' "$idle" > "$out/not-idle.log"; exit 3
    fi
    method=${item%:*};round=${item#*:};label=ds-converge-1021/r31/bs$bs-$method-$round
    test ! -e "$root/$label.log"
    unset MEGAMOE_TRACE
    if test "$method" = ds; then
      export MEGAMOE_PACKAGE="$root/ds-bisect-1021/c5-r1/package"
      runner=("$campaign/ds_common.py" "$label" --bs "$bs")
      baseline=ds-converge-1021/r0/bs$bs-ds-common-a
    else
      if test "$method" = parent; then export MEGAMOE_PACKAGE="$root/ds-bisect-1021/c5-r1/package";baseline=ds-converge-1021/r0/bs$bs-ours-a
      else export MEGAMOE_PACKAGE="$out/package";baseline=ds-converge-1021/r31/bs$bs-candidate-a; fi
      if test "$method:$round" = candidate:a; then export MEGAMOE_TRACE=1; fi
      runner=("$campaign/ours_common.py" "$label")
    fi
    echo "START $label $(date -Is)"
    timeout -k 90s 600s python -I -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
      --master-addr 127.0.0.1 --master-port 29998 --max-restarts 0 "${runner[@]}" > "$root/$label.log" 2>&1
    if test "$method:$round" = candidate:a; then
      python "$root/check_label.py" "$label" "ds-converge-1021/r0/bs$bs-ours-a"
    else python "$root/check_label.py" "$label" "$baseline"; fi
    grep -E '^RESULT|^TRACE_COMPLETE' "$root/$label.log"
  done
  python "$campaign/check_common.py" "$out/bs$bs-candidate-a" "$out/bs$bs-ds-a"
done
echo MATH_BRIDGE_NODE_COMPLETE
