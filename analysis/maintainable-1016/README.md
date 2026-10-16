# MegaMoE 工程整理与 EP8 验证记录

日期：2026-10-16，北京时间。代码整理已完成；设备验证矩阵不完整，共享专家用例因资源占用未运行。

## 版本与改动边界

- 分支：`codex/megamoe-maintainable-1016`。
- 父版本：`18ada75b1`；原生对照使用此前 `0f90f5333` 构建的 candidate wheel，父版本后续仅补文档/测试。
- 代码提交：`c7a774c27`。设计说明：[ARCHITECTURE.md](../../experiments/megamoe/ARCHITECTURE.md)。
- 受测源码 overlay SHA256：`3051f8cf86a35f70bc7ad03802f82f3f113b7e505cf5ccf967c8365610fc6ac9`。
- 最终代码提交相对受测 overlay，仅更正三行 C++ 注释；两份相关头文件的所有非注释行已逐行核对一致。
- wheel、SDK、编译器、SHMEM/CATLASS 身份和每轮原始结果见 [report.json](evidence/report.json)。

主要整理：删除不可达 URMA/layered、量化 Combine、prefetch 执行分支；A8W8 MX 计算移入独立文件；
W4 持久流水改为具体类型并校验配置匹配；Host Reset 预算直接使用 WorkspaceInfo 的真实分配结果；
清理计算配置中的无效模板参数及 K-plan 参数。公开 Python API、224B tiling ABI、有效 GM 布局和调度语义保持。

清理范围仅为 MegaMoE 实验内核。DeepEP 其他通信算子的 URMA runtime 能力和共用上下文 ABI 不属于本次删除范围。
没有照搬 DeepGEMM-Ascend 的调度、环形 GM 或数值核；借鉴的是角色分工、职责分离和布局单一来源。
没有在设备热循环中新增 timeout、barrier 或计时排空；这类诊断开销需要独立实验，不与工程整理混合。

## 验证结果与限制

| 项目 | 结果 / 范围 |
|---|---|
| 原生构建 | Ascend950 三个实际入口实例构建、安装、导入通过，包括 W4 配对/非配对及 A8W8 |
| Linux CPU 全套 | 313 passed，包含打包、启动、真实生产函数抽取、调度及资源协议桩 |
| 本地 Windows | 264 passed / 10 skipped；不含依赖 Bash 的启动/打包测试，完整结果以上述 Linux 套件为准 |
| GM 布局 | 288 配置：冻结基线与候选的总大小、Reset 元素数、GMM2 起始偏移完全一致；另检查区域边界与对齐 |
| W4A8 balanced | EP8，BS32/96/256/1024/4096，ABBA 四组；逐 rank 的完整 BF16 输出 SHA256 一致 |
| W4A8 random | EP8，BS33/257/4104，旧/新版本逐 rank 完整 BF16 输出 SHA256 一致，覆盖配对阈值两侧 |
| A8W8 | EP8，H1024/I256/E8/TopK2；BS1→17→96→257→4096→17，两种输入 BF16/FP8；96 对逐 rank 输出哈希及 count 一致 |
| W4 共享专家 | **未运行**；计划 S1、BS32/257。CPU1 启动前空闲检查发现其他训练占用八卡，CPU3 也被占用；未终止其他任务 |
| 独立 NPU Golden | 未运行；上述旧/新输出等价不能替代独立数学精度验证 |
| 其他规模/诊断 | 未重测 EP32/128、共享专家设备路径、compute replay、timer-on/off 开销；A8W8 只做输出回归，不宣称性能收益 |

机器为共享环境，每阶段启动前检查八卡空闲，不具备全程排他锁。补充随机路由采样只用于输出回归，
不作为性能结论。旧报告中的共享专家 PASS 不继承为本次 PASS。

`report.json` 明确记录 `all_tested_outputs_equal=true`、`device_matrix_complete=false`。
汇总器遇到缺失结果默认报错；本次显式记录共享专家未测原因后写入不完整报告，并以非零状态退出，不能用于自动晋升 main。

## 完整通信性能

柜2 CPU1，Ascend950DT，CANN9.2.0 B080，Python3.11.16，Torch2.7.1+cpu / torch_npu2.7.1.post10。
CATLASS `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1`，SHMEM `4dd80e2129da42bd773982c2bb3758af1ed0bba8`。
ABBA 时间窗口 15:16:48–15:21:36。H7168/I3072/E384/TopK6，无共享专家，balanced，FP8/FP4。

每组 warmup10、正式20轮，取后10轮。指标为完整通信 MegaMoE kernel Duration 的逐轮 rank-min 均值；
每个版本两个独立运行再平均，rank-max 另保留在 JSON 中。wheel 支持 timer，但正式调用不传 timer。

| BS | 基线 μs | 整理后 μs | 延迟变化，正值变慢 |
|---|---:|---:|---:|
| 32 | 633.039 | 631.839 | −0.190% |
| 96 | 660.691 | 660.839 | +0.022% |
| 256 | 722.208 | 720.531 | −0.232% |
| 1024 | 1160.471 | 1161.756 | +0.111% |
| 4096 | 4130.261 | 4130.387 | +0.003% |

结论：本窗口未观察到明显性能回退，不宣称新的加速或所有 EP/平台上的“最快”。
两个独立运行不构成充分的统计显著性证明；不能把 20 个轮次当作 20 次独立实验。

## 复现与证据

- `validate.sh`：CPU、A8W8 和 W4 ABBA；提供隔离的 TASK_ROOT、REFERENCE_VENV、CANDIDATE_VENV、MASTER_ADDR 及 CANN/SHMEM 环境。
- `regression.sh`：W4 random 与 shared 输出回归；新的输出目录必须为空，不能覆盖旧结果。
- `check_a8w8.py`：独立 A8W8 公共 API fixture；不依赖 MegaMoE benchmark 的 W4 假设。
- `collect_report.py`：严格检查轮数、测量口径、各 rank 报告、输出哈希与缺失项目。
- `archive.py`：保留有效配置、原始 kernel CSV、输出哈希、脚本、构建日志和受测 C++ 文件 SHA256。
- [compact-evidence.tar.gz](evidence/compact-evidence.tar.gz)：约 1.3 MiB；不含大体积输出数组和完整 profiler trace。
- [cpu-tests.log](evidence/cpu-tests.log)、[build-final.log](evidence/build-final.log)：CPU 和最终原生构建日志。

剩余晋升条件：在空闲八卡节点上补齐旧/新共享专家回归，随后重新生成完整报告；需要生产级精度保证时，
再运行独立 Golden。此次只保存独立候选分支，不覆盖远端 main，也未安排自动重试。
