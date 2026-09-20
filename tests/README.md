# 功能与集成验证

按照 API 契约、独立参考、设备功能和集成行为组织测试。当前可运行 API 契约、CPU 参考及测试框架自检；V1/V2 设备用例已提供，但后端仍未实现，不能据此宣称通信或训练通过。

## CPU 验证

从仓库根目录运行（建议使用隔离的 Python 环境）：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not framework" --junitxml=cpu-correctness.xml
```

无需安装 Torch、torch_npu、NumPy、CANN 或 SHMEM。此命令运行契约、参考与框架自检，并收集设备用例，以明确原因报告 skip。只检查接口时仍可指定 `tests/api`。

- `test_reference.py`：人工计算锚点、全局路由参考、两种布局及独立数值通道；注入错路由、漏行、重复行、错索引/计数/权重、遗漏贡献、额外乘 gate 和 NaN，确认检查器拒绝错误结果。
- `test_harness.py`：设备执行开关、空资格选择、启动参数、异常清理与已有进程组保护；模拟对象只检查环境生命周期，不代表通信后端。
- `test_benchmark.py`：配置/能力门禁、独立路由字节数、完成边界计时、逐轮 rank MAX、失败传播和 JSON/CSV 协议；轨迹对象只验证引擎，不产生设备性能证据。
- `utils/reference.py`：只从全 rank 输入和路由生成预期，不读取待测实现的 handle 或内部表，不调用设备代码。
- `utils/precision.py`：BF16/E4M3FN 上游 dtype 模拟、L0-L2 双标杆指标和小值域判定。
- `golden/deepep_reference_v2.json`：由固定公式重建的版本化 Golden 摘要，同时锁定设备资格矩阵；生成与审核方法见[精度标准](../docs/precision.md)。

在已配置 Torch/torch_npu 的环境额外运行：

```bash
python -m pytest --require-framework -s --junitxml=cpu-correctness.xml
```

该模式要求框架兼容检查实际执行；依赖缺失会失败。检查仅导入框架、验证接口仍明确报错且未初始化 NPU，不分配设备 Tensor 或运行通信。普通 CPU 运行会明确跳过该单项；其他错误不会被转换为 skip。

GitCode 的 `compile` 触发流水线运行 CPU 契约、参考及框架自检，并上传 JUnit；设备 skip 数保留在结果里。框架导入检查、设备通信资格与流水线扫描结果分别记录。

## 场景与判断口径

| 范围 | 本批内容 |
| --- | --- |
| 路由 | 普通、同 rank 多 expert、重复 expert 槽位、`-1`/完全屏蔽 token、空 expert、倾斜、零 token rank、全部无效路由 |
| CPU 语义层 | 两 rank、四 expert、K=2 的手算场景；用于检查路由、布局、重复槽位、mask 和独立值通道，不作为设备规模证明 |
| CPU 参考矩阵 | EP 1/2/4/8、H 16/127/128/256/7168、K 1/2/4/6/8、每 rank 最大 4096 token；隔离正交维度并生成 Golden |
| NPU 资格矩阵 | EP 1/2/4/8、H 256/512/1024/7168、K 1/2/4/6/8、T 边界 1/15/16/17/256/257/4096、alignment 1/128；V2 combine case 强制 `H % 256 == 0` |
| dtype | BF16 与上游 V2 明确支持的 FP8 E4M3FN dispatch 双路径；索引宽度服从原生 `topk_idx_t`，值通道 FP32 |
| 数值域 | 均匀 `[-5,5]`、近似正态（均值 `[-100,100]`、标准差 `[1,25]`）、dtype 小值域、千分之一乘性离群点；全部由无状态公式确定 |
| 普通布局 | 每个源 token 向每个目标 rank 只发一行，索引转为本地 expert ID；不比较无效槽位中未定义的 dispatch 权重 |
| 展开布局 | 每个有效槽位一行，expert-major；保留重复槽位的重数，允许同一 expert 内调度顺序变化 |
| combine 单项 | dispatch 仅准备真实 handle 和行位置；模拟专家结果由全局输入生成，答案直接从源路由求和 |
| 往返 | 模拟专家处理实际收到的 x；答案独立从源输入和路由计算，不能直接要求 `combine(dispatch(x)) == x` |
| 值通道 | 有权重/无权重分别运行；combine 输入采用不同于 dispatch 权重的值，检查独立回传、不额外乘 x、无效源槽位输出为零 |
| handle | V1 cached 更新 x；V2 两种布局 cached 更新 x，省略 `do_expand` 时继承原布局 |

检查器使用输入中可识别的 token/槽位标记匹配真实输出，按独立参考检查数据、路由与计数，不读取 `recv_src_metadata` 或其他内部映射。设备适配只读取公开计数以确定有效行；不要求后端复制参考模型的内部行结构。

预设语义样本继续采用精确比较。资格矩阵另以 CPU FP64 为 Golden、上游同 dtype 路径为第二标杆，按 L2 的 MARE/MERE/RMSE 和小值域错误数比判定。完整规则见[精度标准、Golden 与比较方法](../docs/precision.md)。

## 设备功能资格（后端实现后执行）

仅收集设备用例，无需安装框架：

```bash
python -m pytest tests/legacy tests/elastic --collect-only -q
```

在实现后端、安装匹配的 Torch/torch_npu/CANN/SHMEM 且相应数量 NPU 已获分配的环境，从仓库根目录分别运行各 EP 规模：

```bash
torchrun --standalone --nproc-per-node=1 -m pytest tests/elastic/test_workflows.py -k qualification_matrix --run-device --device-capabilities=all -x -q
torchrun --standalone --nproc-per-node=2 -m pytest tests/elastic/test_workflows.py -k qualification_matrix --run-device --device-capabilities=all -x -q
torchrun --standalone --nproc-per-node=4 -m pytest tests/elastic/test_workflows.py -k qualification_matrix --run-device --device-capabilities=all -x -q
torchrun --standalone --nproc-per-node=8 -m pytest tests/elastic/test_workflows.py -k qualification_matrix --run-device --device-capabilities=all -x -q
```

两 rank 的手算语义、cached 和 V1 用例仍使用 `--nproc-per-node=2` 单独运行。资格记录必须保存四次 EP 命令的全 rank 结果；只运行其中一个 EP 不能声明矩阵整体通过。

默认执行只报告设备用例未运行。`--run-device` 还必须通过
`--device-capabilities`（或 `DEEPEP_DEVICE_CAPABILITIES`）声明后端已经实现的能力；
可选值为 `legacy`、`elastic-ordinary`、`elastic-expanded`、`elastic-fp8`、
`elastic-combine`、`elastic-cached`、`elastic-reinitialize`，完整兼容后端可写 `all`。
单 rank V2 还需声明 `elastic-single-rank`。
不支持的参数组合在初始化 NPU/SHMEM 前明确 skip，避免把缺失能力误判成算子精度失败。
缺少启动信息、设备、依赖，或已声明支持的路径运行时报错，仍然失败。

若后端没有 `elastic-reinitialize`，一次 torchrun 只允许选中一个受支持用例；门禁会在设备初始化前拒绝多用例运行。此时由外层脚本为每个 node id 启动独立 torchrun。例如仅支持 expanded BF16 dispatch 的后端，可按单个 node id 运行：

```bash
torchrun --standalone --nproc-per-node=2 -m pytest \
  'tests/elastic/test_workflows.py::test_elastic[dispatch-bf16-False-expanded-normal]' \
  --run-device --device-capabilities=elastic-expanded -q
```

启用设备执行却没有选中任何受支持用例也会失败。当前 API 桩不能通过此命令。

设备环境按当前 `torchrun` world size 初始化 HCCL，通信超时 90 秒，退出时清理本次创建的进程组；发现调用方已有进程组则拒绝接管。每个用例显式销毁 buffer。测试保留事件等待；异步重叠及延迟消费另行验证。alignment=128 时只比较公开计数确定的有效行，并显式排除 padding。多 rank 输出按 rank 分别保存，所有 rank 通过才能记录资格，不能只摘取某一个 rank 的成功结果。

本批不覆盖低时延专用 API、padding/alignment>1、跨节点拓扑资格、bias、性能、autograd/完整训练、cached 新权重、多 handle 交错及长期输出存活。FP8 已进入参考、Golden 与 V2 设备收集入口，但原生后端未实现，不能据此宣称设备通过。

## 目录职责

- `api/`：固定原生 API 基线、导出和典型完整调用。
- `legacy/`：V1 dispatch、combine、低时延和复用路径。
- `elastic/`：V2 dispatch、combine、EPHandle 与布局衔接。
- `integration/`：核心/Beta 安装、框架接入与资源生命周期。
- `utils/`：分布式环境辅助和业务参考；Beta 专属测试放在实验目录。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
