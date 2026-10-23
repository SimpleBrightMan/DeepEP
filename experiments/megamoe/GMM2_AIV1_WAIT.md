# GMM2 完成等待下放至 AIV1

基线：`codex/megamoe-packed-prefetch-original-math-1021` 的 `d986ac9dbe576af5e3d966617099cc9a0f302baf`。
本次仅迁移 routed A8W4 Wave 路径的完成等待和 ready 发布；保留 packed prefetch、原数学计算、
GMM2 同向分核、GMM1/GMM2 专家 slice/prologue、资源 handoff 和 AIV1 的公平轮转。
现有 CV 单 M 块路径继续使用原小任务完成协议。

## 协议

- AIV1 初始化一份 EMPTY credit。AIC 在整个 Wave 的 GMM2 输出下发后，在 FIX 队列中先等 EMPTY，
  再向配对 AIV1 发 FULL；不在 Wave 末做 Scalar FIX 排空。
- AIC 随后写每核独占 cache line 的 GM issued 序号。它仅表示 Wave 已提交，可能早于真实 FIX 完成。
  AIV1 读不到预期序号时立即返回调度器；读到后通过 `CrossCoreWaitFlag<4U, PIPE_S>(FULL)` 等真实完成。
- AIV1 按原 Wave 规划和同向 rolling owner 重放实际生产核，每个有效 slice 递增一次原 Combine ready 计数，
  然后归还 EMPTY。AIV0 不参与转发，AIC 不重复发布这些 ready。空 Wave/尾部空专家仍严格消费一次事件。
- 最后回收一份 EMPTY 并完成唯一的 FIX→Scalar 收尾，再销毁物理计算资源。零 Wave 同样回收初始化 credit。
- FULL/EMPTY 使用 0/1；权重通道 4～11、GMM1 通道 12～15 保持不变。新增 issued 存储进入原连续 reset 区，
  host 仍从实际 `WorkspaceInfo::flagResetElementCount` 推导重置大小，不修改 reset 顺序。

`AIV_COMBINE_WAIT` 的新增完成等待设置实际 Wave 并清除旧 slice 坐标；计时不额外增加同步。
此事件族仍可能包含其他 Combine 等待，不能仅凭名字推断唯一原因。

## 验证边界

- 生产 helper 的延迟 FIX/Scalar 队列测试覆盖单 credit、issued 早到、AIV1 唯一发布、AIV0 不参与和多次调用清理。
- 生产 AIV1 消费函数在全部 32 核上重放空专家、尾片、多 Wave 和随机计数；未 issued 时不推进游标。
- 原 GMM Wave 调度与冻结基线一致；AIV1 公平调度、连续 flag reset 和工作区边界测试覆盖新增协议。
- timer OFF 的 Ascend950/CANN 9.2.0 原生 wheel 在独立 venv 编译并导入成功。
- 完整本机 CPU 回归中的 9 项已有失败在未修改 `d986ac9` 同样复现：GMM1 async stub 的 PIPE_S/PIPE_V 缺失、
  两项 GMM1 role stub 与现行 API 不匹配、六项 SDK 校验测试先触发 macOS 平台拒绝。相关新回归通过。

本次未运行目标分支的 NPU 精度或性能测试。此前其他源码的 8P 性能/timer 结果不作为本分支的验收结果。
