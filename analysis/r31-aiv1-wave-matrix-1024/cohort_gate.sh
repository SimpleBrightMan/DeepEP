#!/bin/bash
# 由节点脚本source：持锁等待控制器全组GO，任一节点失败则全组不启动worker。
exec 9>"$1"
flock -n 9 || { echo 'Test slot locked'; return 3; }
if ! task_snapshot=$(npu-smi info); then
    printf '%s\n' "$task_snapshot"
    return 4
fi
task_idle_count=$(printf '%s\n' "$task_snapshot" | grep -c 'No running processes found in NPU' || true)
if [[ $task_idle_count != 8 ]]; then
    printf 'Idle probe count=%s; not starting workers\n%s\n' "$task_idle_count" "$task_snapshot"
    return 4
fi
printf 'COHORT_READY\n'
IFS= read -r task_control || return 5
[[ $task_control == GO ]] || return 5
