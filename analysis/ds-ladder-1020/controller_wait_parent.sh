#!/bin/bash
set -eo pipefail
campaign=/home/codex-deepep/bs1024-master-1019/ds-ladder-1020
node=${1:?node required}
revision=${2:?revision required}
parent=${3:?parent required}
source "$campaign/../env.sh"
set +e
python "$campaign/prepare.py" "$node" "$revision"
rc=$?
set -e
if test -d "$campaign/$node"; then printf '%s\n' "$rc" > "$campaign/$node/build.rc"; fi
if test "$rc" != 0; then exit "$rc"; fi
# 可以提前构建，但无效/未完成父版不能进入累计性能链；不把编译顺序当作执行依赖。
for attempt in {1..180}; do
    if test -f "$campaign/$parent/campaign.rc"; then
        if test "$(cat "$campaign/$parent/campaign.rc")" != 0; then
            printf '4\n' > "$campaign/$node/campaign.rc"
            echo "PARENT_FAILED $parent: no NPU worker launched" >&2
            exit 4
        fi
        exec bash "$campaign/run.sh" "$node" "$parent"
    fi
    sleep 5
done
printf '4\n' > "$campaign/$node/campaign.rc"
echo "PARENT_INCOMPLETE $parent: no NPU worker launched" >&2
exit 4
