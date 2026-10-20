#!/bin/bash
set -eo pipefail
campaign=/home/codex-deepep/bs1024-master-1019/ds-bisect-1021
node=${1:?node required}
revision=${2:?revision required}
parent=${3:?parent required}
source "$campaign/../env.sh"
set +e
python "$campaign/prepare_ours.py" "$node" "$revision"
rc=$?
set -e
if test -d "$campaign/$node"; then printf '%s\n' "$rc" > "$campaign/$node/build.rc"; fi
if test "$rc" != 0; then exit "$rc"; fi
exec bash "$campaign/compare_ours.sh" "$node" "$parent"
