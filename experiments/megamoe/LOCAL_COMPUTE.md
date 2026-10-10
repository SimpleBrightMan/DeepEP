# 零通信本地计算验证

本分支 `codex/megamoe-compute-opt` 基于 `codex/megamoe-local-compute`，
后者基于 `codex/megamoe-compute-replay`。保持计算调度和数学运算，优化 GMM2 完成边界。
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
脚本默认是功能smoke；默认不打timer，不输出kernel时延。性能采样使用下文的 `--profile`。
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

## GMM2 优化及复现实验（2026-10-10）

### 运行方法与测量口径

完成上面的环境加载和构建后，在仓库根目录执行：

```bash
# 八卡分别跑 EP8 形状的本地计算，不建立通信。正式性能比较需独占设备。
python experiments/megamoe/scripts/local_compute.py --bs 4096 --profile
# 单卡：性能阶段 timer 关闭，结束后另跑一次诊断流水，不混入性能样本。
python experiments/megamoe/scripts/local_compute.py --devices 0 --bs 4096 --profile --trace
# 与已保存的同设备、同 BS、同 seed 基线校验完整 GMM2 输出
python experiments/megamoe/scripts/local_compute.py --bs 4096 --profile \
  --reference-dir /home/zyx/opt-baseline-ep8
```

默认 warmup 20；三轮各采 50 次 MegaMoE kernel Duration，统计每轮后 25 次。
每轮先排入独立的小计算以减少 host 提交不及时，其耗时不纳入 MegaMoE 样本。
这不是完整 MoE E2E：没有真实 Dispatch、Combine 和跨卡同步。
`device_N.json` 保存所有 kernel 样本、统计值、输出 SHA256 和已安装设备库的 SHA256。
`source_commit` 是 runner 源码版本，不保证等于已安装 wheel 版本；切换 wheel 对比时应看
`kernel_binary_sha256`。`summary.json` 汇总八卡；`npu_before/after.txt` 保存设备快照。
设备快照不能证明整个测量窗口独占，因此 `performance_valid` 保守保留为 false。

`--trace` 额外生成 `device_N_timer.json`，不保存 wait 条，等待仍体现为原时间轴空白。
GMM2 标为 `AIC_GMM2_ISSUE`：表示发射区间而非 FIX 完成区间；不能将其累计值冒充计算总时长。
新增 timer 槽位需要重新构建 wheel，旧 raw timer buffer 不应套用新协议。

### 实测结果

H7168、I3072、384 专家、TopK6、无共享专家，BS4096，EP8 形状每卡48专家、每专家512 token。
同一台 Ascend950DT_9582 / CANN B070，timer 关闭，数据来自 profiler kernel Duration。

| 实验 | kernel 时间（μs） | 处理 |
| --- | ---: | --- |
| 原始计算基线，单卡三轮均值约 | 4196 | 对照 |
| 移除 TokenCombine 逐 tile FIX 排空，单卡 | 4113 | 保留 |
| 在上项基础上扩大 A 的 K 搬运窗口到512 | 4135 | 负收益，撤回 |
| 在 FIX 版本基础上改成四槽 L1 B | 4129 | 负收益，撤回 |
| 基线回测，八卡×三轮均值 | 4196.252 | 对照 |
| 最终版，八卡×三轮均值 | 4114.466 | 降低81.786μs，约1.949% |

最终版还去除了 slice-ready 发布前重复的 FIX 等待；未单独证明这一处有可分辨收益。
两项缓冲试验保留在 git 历史供查阅，但不在分支最终代码中。它们只证明这两种具体实现无收益，
不证明短 K 供数、缓冲或同步已经最优。

服务器结果：`/home/zyx/opt-baseline-ep8/`、`/home/zyx/opt-final-ep8/`；
负收益试验：`/home/zyx/opt-a512-1/`、`/home/zyx/opt-b4-1/`。
最终设备代码提交 `543525e`，wheel 为 `dist/build-jp70l016/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`。
最终设备库 SHA256：`ed071caeaf6023529fa7113ff8a7f22fcc1179e0332200bcd83078b75488750b`。
基线使用未改计算实现的 local-only wheel；八卡回测时 runner 已更新，因此不能仅用其 source_commit 识别内核。

### 为什么可以删除等待

代码入口：`csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp` 的
`Gmm2AicMmadA8W4`、`NotifyTokenCombineSliceReady`，以及 `mega_moe_arch35.hpp` 的调用点。

- TokenCombine 在 slice 完成后才发布就绪计数，原先每 tile 的 Scalar FIX 排空不是消费者需要的粒度，
  却阻止了后继发射与 FIX 输出重叠。现在只删除这个分支的逐 tile 排空。
- 逐 tile 消费路径 `NotifyCombineTileReady` 仍保留 FIX 等待；L0C unit flag 保护不变。
- `RunGmm2A8W4` 返回前，局部 `CatlassGmm2` 析构经 pipeline 的 `Cube::End` 完成 FIX_S，
  然后调用方才执行 `NotifyTokenCombineSliceReady` 发布计数。该 helper 不能移到析构之前使用。
- 未删除 activation 就绪依赖、资源回收排空或输出发布顺序。缓冲布局、MMAD tile 和 K 分块最终均保持原样。

### 验证边界与后续方向

BS4096 八卡全部与基线完整 GMM2 输出 hash 一致；单卡 BS96、BS4104（每专家513 token，含尾块）也一致。
BS4104 单独 timer pass 输出一致、协议解析成功，JSON 约3.27 MB，带 group_id/core_type/sub_id。
这些是回归等价验证，不是独立数值 Golden；真实通信模式尚未做端到端回归。
主机单测78项通过。该轮没有测量 timer-on 相对 timer-off 的开销。

基线测试前曾观察到他人的 NPU 进程，测试后快照为空；结果重复稳定，但不能保证整个窗口独占。
应在预约独占设备后复测再作为正式性能结论；未停止或重置他人任务。
DS 未在本次同机同输入条件下重新测试，因此不能把历史可见区间差285μs直接当成可回收E2E收益。

下一步优先研究跨 tile/slice 保持计算 pipeline 状态，减少反复初始化、资源排空和 flag 握手；
先统计其次数与时间，再做保持依赖关系的单变量实验。DS 参考实现的缓冲更持久，
而单纯在当前生命周期中增加槽位还会增加初始化和切换成本。
之后再检查短 K 的 L1→L0 等待、MMAD 连续发射及编译器生成代码。
目前证据支持“过早完成等待贡献约2%”，不支持“剩余差距全是 FIX 或全是矩阵乘指令本身”。
