#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
campaign=$root/ds-converge-1021
out=$campaign/r17
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid" && test -d "$out"; then printf "%s\n" "$rc" > "$out/campaign.rc"; fi' EXIT
source "$root/env.sh"
python "$campaign/prepare_ours.py" r17 "${1:?revision}"
python -I -m pytest -q "$out/source/experiments/megamoe/tests/test_public_api.py" "$out/source/experiments/megamoe/tests/test_workspace_layout.py" "$out/source/experiments/megamoe/tests/test_source_weight_window.py" "$out/source/experiments/megamoe/tests/test_packed_weight_tiles.py" > "$out/api-cpu-tests.log" 2>&1
for attempt in {1..1440}; do
  if test -f "$campaign/r7-r2/measurement.rc"; then break; fi
  sleep 5
done
test "$(cat "$campaign/r7-r2/measurement.rc")" = 0
test "$(cat "$campaign/r0/campaign.rc")" = 0
bash "$campaign/compare-math-r17.sh"
python "$campaign/validate_node.py" "$campaign" r17
