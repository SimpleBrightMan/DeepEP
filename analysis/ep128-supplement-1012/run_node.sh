#!/bin/bash
set -eo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
rank="$1"; method="$2"; run="$3"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/zyx
source /home/pkg/b070/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH="$root/shmem:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT=tcp://177.28.27.1:26863
export DG_JIT_CACHE_DIR="$root/ds-cache"
unset PYTHONPATH PYTHONHOME ASCEND_LAUNCH_BLOCKING
if [ "$(npu-smi info | grep -c 'No running processes found in NPU' || true)" != 8 ]; then
    echo 'NPU occupied or state unknown'; exit 2
fi
cd /tmp
if [ "$method" = probe ]; then
    exec timeout --signal=TERM --kill-after=10s 300s python -I -c 'import torch,torch_npu; print(torch.__version__,torch_npu.__version__,flush=True); assert torch.__version__=="2.12.0+cpu" and torch_npu.__version__=="2.12.0.post2"; [(torch.npu.set_device(i),torch.ones(1,device=f"npu:{i}"),torch.npu.synchronize(),print("DEVICE_OK",i,flush=True)) for i in range(8)]'
fi
options=()
if [ "$method" = ours ]; then
    options=("$root/entry.py" benchmark --config "$root/config.json" --mode perf --input fp8 --routing balanced --output-dir "$root/$run")
    if [ "$run" = ours1 ]; then options+=(--timer --timer-ranks 0); fi
elif [ "$method" = ds ]; then
    options=("$root/ds_entry.py" --output-dir "$root/$run" --bs-values 512 2048)
else
    echo 'Unknown method'; exit 2
fi
exec timeout --signal=TERM --kill-after=30s 1800s python -I -m torch.distributed.run \
    --nnodes 16 --nproc-per-node 8 --node-rank "$rank" --master-addr 177.28.27.1 \
    --master-port 29784 --max-restarts 0 "${options[@]}"
