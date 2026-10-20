#!/bin/bash
set -eo pipefail
campaign=/home/codex-deepep/bs1024-master-1019/ds-bisect-1021
node=${1:?node required}
revision=${2:?revision required}
parent=${3:-d0}
source "$campaign/../env.sh"
python "$campaign/prepare_ds.py" "$campaign" "$node" "$revision"
exec bash "$campaign/compare_ds.sh" "$node" "$parent"
