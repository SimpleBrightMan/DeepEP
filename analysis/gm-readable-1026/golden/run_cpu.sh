#!/bin/bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
node=$1 master=$2 ep=$3 label=$4 port=$5
[[ $ep == 32 && $node =~ ^[0-3]$ && $label =~ ^[a-z0-9-]+$ && $port =~ ^[0-9]+$ ]] || exit 2
test ! -e "$root/$label" || exit 3
source /home/pkg/b080/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH="/root/miniconda3/envs/zyx/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
printf 'COHORT_READY\n'
IFS= read -r task_control
[[ $task_control == GO ]] || exit 5
IFS= read -r MEGAMOE_CONTROL_TOKEN
[[ $MEGAMOE_CONTROL_TOKEN =~ ^[0-9a-f]{64}$ ]] || exit 5
export MEGAMOE_CONTROL_TOKEN
exec timeout --signal=TERM --kill-after=10s 180s /root/miniconda3/envs/zyx/bin/python -I "$root/local_workers.py" \
    --node "$node" --master "$master" --ep "$ep" --port "$((port+1))" "$root/probe_cpu.py" --label "$label"
