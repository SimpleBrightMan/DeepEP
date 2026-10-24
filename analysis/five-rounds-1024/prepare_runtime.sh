#!/bin/bash
# 新柜只克隆独立zyx环境并补齐冻结SHMEM；不安装/升级其他人的环境。
set -euo pipefail
stage=/home/codex-deepep/r31-ep32-continuous-1024-runtime
test -f "$stage/shmem-libs.tar.gz"
exec 9>"$stage/prepare.lock"
flock -n 9 || { echo 'Runtime preparation already active'; exit 3; }
if [[ ! -e /root/miniconda3/envs/zyx ]]; then
    clone_options=(--offline)
    if [[ ${1:-offline} == online ]]; then
        clone_options=()
        export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=10 CONDA_REMOTE_READ_TIMEOUT_SECS=30 CONDA_REMOTE_MAX_RETRIES=0 CONDA_NO_PLUGINS=true
    fi
    /root/miniconda3/bin/conda create --solver classic "${clone_options[@]}" --yes --clone /root/miniconda3/envs/hkq --prefix /root/miniconda3/envs/zyx
fi
set +u
source /home/pkg/b080/cann-9.2.0/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
/root/miniconda3/envs/zyx/bin/python -I -c 'import sys,numpy,torch,torch_npu; assert sys.version_info[:2]==(3,12); assert torch.__version__=="2.12.0+cpu" and torch_npu.__version__=="2.12.0.post2"; print("ZYX_READY",sys.version.split()[0],torch.__version__,torch_npu.__version__)'
/root/miniconda3/envs/zyx/bin/python -I "$stage/install_shmem_runtime.py"
