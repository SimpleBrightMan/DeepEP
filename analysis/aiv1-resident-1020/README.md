# AIV1 三任务常驻 UB 实验

父版本：`93e5805`（设备代码 `2adf28c`），不叠加已被否决的 GMM1 descriptor 实验。独立分支，不修改 master。

## 假设和改动

原实现 Dispatch prefix 与激活、Combine 复用 UB，任务切换需要从 GM 恢复 prefix。先只隔开三套工作区，保留严格优先级和全部原生完成等待，区分容量/恢复成本与调度策略的收益。

- 激活私有工作块从最大 256 行改为 64 行，原 GMM tile、完整 tile ready 计数和数值算法不变。大 tile 内部分段，全部完成后才通知 GMM2。
- Dispatch prefix、index batch、六槽 token ring 和 metadata 独立常驻；不再写入、恢复 GM prefix 备份。
- Combine 两槽（BS>256）或六槽（BS≤256）及 metadata 独立常驻，Tensor 视图仅绑定一次。
- **常驻不等于跨任务异步**：原 MTE/V 完成和槽位释放等待仍保留，事件 0..5 仍在任务返回前释放。
- 启动前校验容量，实验版暂不支持共享专家；不静默回退或覆盖 UB。

EP8 / BS512 / H7168 / I3072 / E384 / TopK6 / 无共享专家：

| UB 字节范围（左闭右开） | 内容 |
| --- | --- |
| 0–65536 | SwiGLU/MX 工作区，内部按原有依赖复用 |
| 65536–113856 | Dispatch prefix、index、六槽、metadata |
| 113856–142528 | Combine 两个 BF16 行槽 |
| 142528–191680 | Combine metadata |

共 191680 字节，约 187.19 KiB。源代码 `common/mega_moe_aiv1_ub_layout.hpp` 是 host/device 容量计算的唯一来源。

## 验证口径

- 完整通信 EP8、均衡路由、固定 seed；timer-off，warmup10/sample20，沿用末10轮逐轮 rank-min 均值，同时保留 rank-max。
- 同窗口 ABBA；8 rank 有限值及与冻结基线逐位输出对照。不是独立 Golden。
- CPU 测试执行生产 UB 规划函数，检查对齐、边界和隔离；不能代替设备同步验证。
- 流水另采完整被动大点版：32 个 group，AIC/AIV0/AIV1；不因打点增加完成边界。

原生测试正在进行，结果以归档 `summary.json` 及本文件后续更新为准。未验证其他 EP、共享专家，不合入 master。
