#!/bin/bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
node=$1 master=$2 ep=$3 label=$4 port=$5
[[ $ep == 8 || $ep == 32 ]] && [[ $node =~ ^[0-3]$ && $label =~ ^[a-z0-9-]+$ ]] || exit 2
[[ $port =~ ^[0-9]+$ && $port -ge 1024 && $port -le 30000 ]] || exit 2
test -f "$root/cpu-ep$ep/prepared-ep$ep-r$((node*8)).json" || { echo 'CPU Golden not prepared'; exit 3; }
test ! -e "$root/$label" || { echo 'Accuracy output exists; choose new label'; exit 3; }
source /home/pkg/b080/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH="/home/codex-deepep/ep32-r31-ds-1021/shmem/lib:/root/miniconda3/envs/zyx/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH PYTHONHOME ASCEND_LAUNCH_BLOCKING HCCL_IF_IP
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export RES_OPTIONS='attempts:1 timeout:1' ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT="tcp://$master:$port"
read -r ephemeral_first ephemeral_last < /proc/sys/net/ipv4/ip_local_port_range
if (( port <= ephemeral_last && port+3 >= ephemeral_first )); then echo 'Ephemeral port conflict'; exit 3; fi
task_ports=$(ss -Htan "( sport = :$port or sport = :$((port+1)) or sport = :$((port+3)) )")
[[ -z $task_ports ]] || { echo 'Experiment ports in use'; exit 3; }
source "$root/cohort_gate.sh" /home/codex-deepep/ep32-r31-ds-1021/launch.lock
IFS= read -r MEGAMOE_CONTROL_TOKEN
[[ $MEGAMOE_CONTROL_TOKEN =~ ^[0-9a-f]{64}$ ]] || exit 5
export MEGAMOE_CONTROL_TOKEN
exec timeout --signal=TERM --kill-after=30s 1200s /root/miniconda3/envs/zyx/bin/python -I "$root/local_workers.py" \
    --node "$node" --master "$master" --ep "$ep" --port "$((port+1))" "$root/worker.py" \
    --package-root /home/codex-deepep/gm-readable-slice-prepared-a-1027 \
    --output "$root/$label" --golden-dir "$root/cpu-ep$ep/cache" --label "$label"
