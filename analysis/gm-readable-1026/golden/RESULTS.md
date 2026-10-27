# 独立Golden通过与main提升依据

2026-10-27。可读性GM＋slice通知候选的EP8、EP32独立Golden均已完整通过，满足本轮提升main的精度门槛。

## 验证范围

H7168、I3072、E384、TopK6、FP8输入/FP4权重、无共享专家、capacity4096、seed20260922；真实SHMEM/MTE通信。全部token和全部H输出，不抽样，不用指纹相等代替Golden。

| 路由 | BS/卡 | EP8 | EP32 |
|---|---|---|---|
| balanced | 32、96、256、512、1024、2048、4096 | 7/7通过 | 7/7通过 |
| random（含负权重） | 96 | 通过 | 通过 |
| skewed（空专家/偏斜） | 96 | 通过 | 通过 |

共18个分布式配置、360个rank-case，检查330240个token、2367160320个输出元素。所有rank专家计数一致、非零初值7下两次累加一致、重复输出一致，全部worker正常退出。

| 汇总指标 | EP8 | EP32 |
|---|---:|---:|
| 最大相对RMSE（各rank/case取最大） | 0.056817% | 0.089516% |
| 最大混合误差 | 0.0001220703125 | 0.0001220703125 |
| 超过仓库混合误差阈值的元素比例 | 0 | 0 |

严格沿用仓库`golden.compare`：`diff_thd=0.01,pct_thd=0.05,max_diff=0.1`，未放宽。NumPy参考建模BF16和MXFP8舍入，不是Cube累加或exp的逐位模拟。

## 身份与证据

- 算法冻结提交`8118a7647f43db5e1fe86116a1d473514bcb6134`；后续只增加测试和文档，csrc、Python API与构建配置保持一致。
- 包SHA256 `32dbbb93e008f6fa327298760fb87cfe4fcca1fc00de6475d0c7d0b7591097ac`；设备库SHA256 `5d42b7e1d66a6de8e15be8a51ca260ee5149646e6a1ea1faaec2cdb713acdb24`。
- EP8：05柜CPU1，`outputs/gm-readable-1026/golden-1027/ep8-r05-1-a`，`summary.json`状态`full_golden_pass`。
- EP32：05柜CPU3/4/6/8，`outputs/gm-readable-1026/golden-1027/ep32-r05-3468-f`，`summary.json`状态`full_golden_pass`。
- 两目录均包含完整rank身份、独立参考合同、逐rank精度报告、完成记录、日志和证据哈希。相对路径均以工作区`C:\Project\Infra\ops-transformer-hs`为根，证据已在本地，不依赖服务器在线。
- Golden参考模块19项CPU测试通过；离线审计8项CPU测试及实际TCP协调器测试通过；32rank纯CPU探针24次gather＋24次barrier，全部校验及退出通过。

## 控制面失败与修复，未修改算子

初次EP32在BS2048的Gloo误差报告汇总处超时。将报告先保存为各rank文件后，另一次仍在下一用例配置同步处超时，未证明仅与大报告有关。独立TCPStore纯CPU探针也超时。

最终精度测试只替换了测试主机协调：使用经过单次随机密钥认证的标准库TCP Manager、直接启动/管理本地8个worker，不创建Gloo/HCCL ProcessGroup，也不经过torchrun agent的Store/退出屏障。先完成真实32rank CPU探针，再运行EP32；以各子进程真实退出码和完整报告审计，不使用`os._exit(0)`掩盖清理问题。

EP8有效窗口仍保留原Gloo测试控制，EP32有效窗口使用新测试控制；独立NumPy判定、输入合同与设备库完全相同。本轮不是性能比较，不借此改变或反填此前timer-off性能数据。新协调器仅位于`analysis/`，没有改生产API或设备通信。

失败窗口`ep32-r05-1468-a`、`ep32-r05-3468-e`、CPU探针失败及多次占用门禁取消均完整保留，不算PASS。柜1部署尝试因占用取消，没有执行NPU用例。本次未停止其他任务、重置设备或修改共享环境。

## main与剩余边界

旧main为`7de059fc5b5534c660926c51b84b7e3687b89915`，备份分支`codex/main-backup-before-gm-readable-slice-20261027`。以正常fast-forward提升候选，禁止force覆盖其他人的新提交。

我的建议是作为新的开发/优化基线：可读性整理已完成，EP8/32性能筛选未见明显新回退，本轮独立精度也通过。但这不是全域SOTA证明：EP64/128、BS>4096、其他模型维度、A8W8、共享专家均未在本轮验收；默认旧测试启动器的跨机稳定性也不由本精度脚本修复。既有解释器退出时SymmBuffer析构warning与格式warning保留日志，未在本轮顺手改无关代码。
