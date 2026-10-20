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
python -I -m pytest -q "$campaign/$node/source/experiments/megamoe/tests/test_public_api.py" \
    > "$campaign/$node/api-cpu-tests.log" 2>&1
exec bash "$campaign/compare_ours.sh" "$node" "$parent"
