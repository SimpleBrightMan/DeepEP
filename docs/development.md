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

## 提交前检查与 CI

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) 是本地与 CI 共用的检查入口。框架版本固定在 `requirements-dev.txt`，hook 固定到提交 SHA。使用 Python 3.11 按[贡献指南](../CONTRIBUTING.md)安装开发依赖并运行 `pre-commit install` 后，每次提交会检查暂存文件。

| 检查 | 范围与约定 |
| --- | --- |
| `trailing-whitespace` | 清理行尾空白，保留 Markdown 的双空格换行 |
| `end-of-file-fixer` | 确保非空文本文件以一个换行符结束 |
| `check-yaml` | 检查 YAML 语法和重复键 |
| `check-json` | 检查 JSON 语法和重复键 |
| `check-added-large-files` | 拦截超过 1 MiB 的普通 Git 文件；Git LFS 文件由工具排除 |
| `check-merge-conflict` | 检查遗留的合并冲突标记 |
| `detect-private-key` | 检查常见私钥文件标记 |
| `ruff-check` | 检查 Python 导入、语句、语法、未使用导入和未定义名称等常规问题，规则见 [`.ruff.toml`](../.ruff.toml) |
| `ruff-format` | 统一 Python 缩进、空格、换行和引号；使用空格缩进、双引号和 LF 换行，目标行宽 120 列 |
| clang-format | 第一方 C/C++/AscendC，包括 `.asc`；采用 [`.clang-format`](../.clang-format) 中的四空格、120 列等规则 |
| codespell | 文档、代码及文本配置；`AGRS` 和 `CANN` 为有效术语，JSON/CSV 数据及 SVG 图稿不做拼写改写 |

第三方源码不运行 Ruff、clang-format 和 codespell；`third-party/README.md` 仍检查拼写。基础检查没有全局排除第三方目录，也不整体跳过 SVG/CSV 文件。私钥标记检查不能替代现有的依赖与安全扫描。

大小检查使用 `--enforce-all`，冲突标记检查使用 `--assume-in-merge`，确保在 CI 的干净 checkout 中也检查已提交文件，不依赖暂存区或正在进行的 merge 状态。

```bash
# 手动检查暂存文件。
pre-commit run --show-diff-on-failure

# 首次接入或修改规则后检查全仓。
pre-commit run --all-files --show-diff-on-failure

# 提交后检查本次 PR 相对目标分支的改动。
git fetch upstream main
PRE_COMMIT_BASE="$(git merge-base upstream/main HEAD)"
pre-commit run --show-diff-on-failure --from-ref "$PRE_COMMIT_BASE" --to-ref HEAD
```

行尾修复、ruff-format 和 clang-format 可能改写文件。检查差异、重新暂存并提交后，重复检查直到退出码为 0 且没有新的改写。首次运行会从 GitCode 镜像获取 hook，并从 Python 包源安装各自的隔离环境；下载失败同样会使检查失败。更新版本时需同步核对版本注释并执行全仓检查。

[GitCode 流水线](../.gitcode/workflows/PR-pipeline_deepep.yml)沿用现有 PR 评论 `compile` 触发方式。`CodeCheck_pre_commit` 检查 PR 的合并结果，使用固定开发依赖运行 `python3.11 -m pre_commit run --all-files --show-diff-on-failure`，不受 `need_build_ut` 条件限制，因此纯文档 PR 也执行。失败直接使作业失败，流水线不会自动提交修改。

CPU 正确性、benchmark dry-run、SCA 和恶意代码扫描保持各自的作业与执行条件。pre-commit 不启动 NPU，也不代替这些验证。
