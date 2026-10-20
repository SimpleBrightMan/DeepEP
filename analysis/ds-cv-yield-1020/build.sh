#!/bin/bash
set -eo pipefail
root=/home/codex-deepep/bs1024-master-1019
out=$root/ds-cv-yield-1020-v1
controller_pid=$BASHPID
trap 'rc=$?; if test "$BASHPID" = "$controller_pid"; then printf "%s\n" "$rc" > "$out/build.rc"; fi' EXIT
source "$root/env.sh"
python "$out/prepare.py"
echo BUILD_AND_CPU_COMPLETE
