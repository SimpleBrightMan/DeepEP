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
if test -d "$campaign/$parent"; then
  for attempt in {1..900}; do
    if test -f "$campaign/$parent/campaign.rc"; then
      if test "$(cat "$campaign/$parent/campaign.rc")" != 0; then
        printf '4\n' > "$campaign/$node/campaign.rc"
        echo "PARENT_FAILED $parent: no NPU worker launched" >&2
        exit 4
      fi
      exec bash "$campaign/compare_ours.sh" "$node" "$parent"
    fi
    sleep 5
  done
  printf '4\n' > "$campaign/$node/campaign.rc"
  echo "PARENT_INCOMPLETE $parent: no NPU worker launched" >&2
  exit 4
fi
exec bash "$campaign/compare_ours.sh" "$node" "$parent"
