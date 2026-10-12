#!/bin/bash
set -eo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
rank="$1"; ep="$2"; stage="$3"; run="$4"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zyx
source /home/pkg/b070/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH="/home/zyx/ep32-main-24b4221/shmem/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT=tcp://177.28.27.1:26853
unset PYTHONPATH PYTHONHOME ASCEND_LAUNCH_BLOCKING
cd /tmp
if [ "$stage" = golden ]; then
    exec timeout --signal=TERM --kill-after=30s 1800s python -I "$root/entry.py" golden \
        --config "$root/config.json" --ep-size "$ep" --ranks "$((rank*8))-$((rank*8+7))" \
        --golden-dir "$root/golden-ep$ep" --input fp8 --routing balanced --check-tokens 16 --workers 2
fi
snapshot=$(npu-smi info)
if [ "$(printf '%s\n' "$snapshot" | grep -c 'No running processes found in NPU' || true)" != 8 ]; then
    echo 'NPU occupied or state unknown; no job was terminated'; exit 2
fi
options=()
if [ "$stage" = accuracy ]; then
    options+=(--check-tokens 16 --golden-dir "$root/golden-ep$ep")
elif [ "$run" = perf1 ]; then
    options+=(--timer --timer-ranks 0)
fi
exec timeout --signal=TERM --kill-after=30s 1800s python -I -m torch.distributed.run \
    --nnodes "$((ep/8))" --nproc-per-node 8 --node-rank "$rank" \
    --master-addr 177.28.27.1 --master-port 29774 --max-restarts 0 \
    "$root/entry.py" benchmark --config "$root/config.json" --mode "$stage" \
    --input fp8 --routing balanced --output-dir "$root/ep$ep-$run" "${options[@]}"
