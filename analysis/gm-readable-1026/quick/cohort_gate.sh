#!/bin/bash
# 由节点脚本source：持锁等待控制器全组GO，任一节点失败则全组不启动worker。
exec 9>"$1"
flock -n 9 || { echo 'Test slot locked'; return 3; }
# NPU暂未建上下文不等于空闲：已加载NPU运行时的外部Python也可能正在准备下一轮。
# 这里只读maps/进程名，不读取命令行里的私有参数，不停止任何外部进程。
while IFS= read -r task_pid; do
    if grep -qE '/(libtorch_npu|libascendcl)[^/]*\.so' "/proc/$task_pid/maps" 2>/dev/null; then
        printf 'External NPU Python still active: pid=%s; not starting workers\n' "$task_pid"
        return 4
    fi
done < <(ps -eo pid=,comm= | awk '$2 ~ /^(python[0-9.]*)$/{print $1}')
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
