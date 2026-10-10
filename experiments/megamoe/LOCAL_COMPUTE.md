# 零通信本地计算验证

基于`codex/megamoe-compute-replay`；不改变GMM1/SwiGLU/GMM2设备流水。
原分支需要SHMEM初始化和一次完整forward作为priming；本分支新增local-only Plan，
在host准备专家连续FP8输入、E8M0 scales及每AIC一份block-major专家计数。
不初始化SHMEM/HCCL，不调用torch.distributed，不建立跨卡数据通路。
现有扩展仍链接SHMEM库（构建依赖不等于运行时建立通信），保留原版API。

服务器使用`/root/miniconda3/envs/zyx`，从hkq克隆，CANN入口：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zyx
source /home/pkg/b070/cann-9.2.0/set_env.sh
bash experiments/megamoe/scripts/build.sh
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
# 单设备先跑，负载仍是EP8的每卡48专家、BS4096、每专家512token
python experiments/megamoe/scripts/local_compute.py --devices 0
# 8张卡独立运行相同负载；不是8卡集体通信测量
python experiments/megamoe/scripts/local_compute.py
```

默认3次重复，结果在`benchmark-runs/local_compute_<时间>/`。
检查内部GMM2结果finite、nonzero及重复一致性，不读取无效的最终MoE输出。
这不是完整数值Golden校验，不宣称精度通过。
当前脚本是功能smoke；默认不打timer，不输出可用于性能对比的kernel时延。
`host_wall_seconds`只用于排查卡住，包含host开销且可能受其他任务影响。
即使没有通信，仍会争用计算和HBM带宽，不能把共跑结果作为性能优化依据。
新环境/构建只作用于zyx及独立源码目录，不改hkq或其他用户任务。

本地地址兼容区由普通NPU tensor持有，不注册对称堆。local-only Plan在准备完成前拒绝forward，
避免错误地进入通信路径；数据/计数/容量/设备必须匹配。输入计数冻结，测试与真实路由输入不同。

## 2026-10-10 实机验证

- Ascend950DT_9582，CANN B070，Torch 2.12.0+cpu，torch_npu 2.12.0.post2。
- 原始基线`c4790e6`；原生扩展及设备库已成功构建，计算实现验证提交`7da0228`。
- BS4096，H7168，I3072，TopK6，无共享专家；EP8形状每卡48专家，每专家512 token。
- 单卡0和八卡0–7各执行3轮；全部输出finite/nonzero，重复结果逐元素相同。
- GMM2输出先填NaN，确保未覆盖输出不能误判通过；不等价于完整数值Golden。
- 每卡Plan workspace为7,393,402,880字节；这不含权重、输入、本地地址兼容区和框架缓存。
- 服务器八卡日志及汇总：`/home/zyx/smoke-local-compute-ep8/`。
- 默认不启用timer；本次只验证功能，没有发布可比较的kernel性能结论。
