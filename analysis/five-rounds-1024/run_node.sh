#!/bin/bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime=/home/codex-deepep/ep32-r31-ds-1021
node=$1
master=$2
ep=$3
method=$4
bs=$5
label=$6
trace=$7
port=$8
[[ $port =~ ^[0-9]+$ && $port -ge 1024 && $port -le 65532 ]] || exit 2
[[ $ep == 8 || $ep == 32 ]] || exit 2
[[ $node =~ ^[0-3]$ && $method =~ ^(control|combined)$ && $label =~ ^[a-z0-9-]+$ ]] || exit 2
[[ $bs =~ ^(32|96|256|512|1024|2048|4096)$ && ($trace == 0 || $trace == none || $trace == probe) ]] || exit 2
source /home/pkg/b080/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH="$runtime/shmem/lib:/root/miniconda3/envs/zyx/lib:${LD_LIBRARY_PATH:-}"
export HCCL_SOCKET_IFNAME='=enp35s0f2' HCCL_HOST_SOCKET_PORT_RANGE=auto
unset HCCL_IF_IP PYTHONPATH PYTHONHOME ASCEND_LAUNCH_BLOCKING
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export RES_OPTIONS='attempts:1 timeout:1' ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT="tcp://$master:$port"
# 临时端口可能被出站连接随机占用；在READY前拒绝冲突，不修改系统端口配置。
read -r ephemeral_first ephemeral_last < /proc/sys/net/ipv4/ip_local_port_range
if (( port <= ephemeral_last && port+3 >= ephemeral_first )); then
    echo 'Experiment ports overlap host ephemeral range'; exit 3
fi
if ! port_snapshot=$(ss -Htan "( sport = :$port or sport = :$((port+1)) or sport = :$((port+3)) )"); then
    echo 'Cannot inspect experiment ports'; exit 3
fi
if [[ -n $port_snapshot ]]; then
    echo 'Experiment ports already in use'; exit 3
fi
source "$root/cohort_gate.sh" "$runtime/launch.lock"
cd /tmp
args=("$root/benchmark.py" "$method" --bs "$bs" --label "$label" --ep-size "$ep" --package-root "$root")
if [[ $trace == probe ]]; then
    args=("$root/probe.py" --label "$label" --method "$method")
elif [[ $trace == 0 ]]; then
    args+=(--trace-ranks 0)
fi
exec timeout --signal=TERM --kill-after=30s 900s /root/miniconda3/envs/zyx/bin/python -I -m torch.distributed.run \
    --nnodes "$((ep/8))" --nproc-per-node 8 --node-rank "$node" --master-addr "$master" --master-port "$((port+1))" \
    --max-restarts 0 "${args[@]}"
