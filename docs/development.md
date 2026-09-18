# 开发指南

本文说明源码组织、命名约定与测试分工。文件职责见[目录设计](repository-layout.md)，开发进展见[路线图](roadmap.md)。

## 从一次 API 调用定位代码

```text
deep_ep/buffers/          参数处理与返回组织
csrc/bindings/           原生注册与转换
csrc/ops/                业务、布局和 handle 语义
csrc/execution/          plan、variants、launcher
csrc/kernels/            完整设备实现及私有辅助
csrc/runtime/            共享上下文、资源和事件
```

`deep_ep/__init__.py` 汇总导出，`csrc/python_api.cpp` 汇总核心原生注册。Beta 在 `experiments/megamoe/` 中建立自己的入口与构建目标。

## 文件粒度与命名

- `legacy`、`elastic` 表达接口家族；支持状态另行记录。
- 第一方 C++/AscendC 使用 `.cpp/.hpp`。小型实现可采用内联头或邻接私有函数。
- 单算子辅助留在 Kernel 邻接位置，同族共享留在对应算法族；跨算子组件满足共同契约后进入 `device/`。
- `runtime` 表达持续资源；`execution` 表达单次准备与启动。
- `topologies` 表达多设备资源连接；`policies` 表达算法执行选项；`release` 表达发行功能与配套。
- `.gitkeep` 仅用于保存空目录；目录有真实文件后可移除。

## 新增实现

先固定相关 API 基线和完整业务用例，再定义输入/输出、handle 与完成条件。添加 ops、Kernel 和合法变体时，同步补齐参数声明、编译清单、能力索引与测试。绑定负责注册和转换，业务及算法留在对应实现模块。

源码依赖、Host/Device 编译域与安装清单应能逐项对应。核心与 Beta 通过根构建解析同一组有效依赖；正式配置只使用经过验证的支持域。

## 测试分工

| 目录 | 验证责任 |
| --- | --- |
| `tests/api/` | 公开对象、签名、返回与完整调用契约 |
| `tests/legacy/` | V1 dispatch/combine、低时延和复用路径 |
| `tests/elastic/` | V2 布局、handle、dispatch/combine |
| `tests/integration/` | 安装、依赖加载、资源生命周期与框架接入 |
| `tests/utils/` | 分布式环境辅助与业务正确性参考 |
| `experiments/megamoe/tests/` | Beta 业务正确性与完整流程 |

`examples/` 展示用户流程，`benchmarks/` 测量性能。计时和启动等窄辅助可复用，业务参考与场景配置由对应测试或实验维护。

## 文档维护

检查 Markdown 相对链接、目录路径、UTF-8 编码、SVG 有效性与 `git diff --check`。保持中英文首页的功能状态与使用说明一致。此类静态检查不能替代 API、构建或设备测试。
