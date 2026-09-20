# 贡献指南

欢迎参与昇腾实现、测试、性能基准和文档的开发。开始前请阅读[目录设计](docs/repository-layout.md)和[开发指南](docs/development.md)。

## 代码范围与职责

- 保持与已固定上游版本对应的 V1/V2 公开接口契约。
- Python 封装放在 `deep_ep/`，原生实现放在 `csrc/`，完整的 MegaMoE 实验放在 `experiments/megamoe/`。
- 单个 Kernel 的辅助代码与实现放在一起；有明确的使用方和测试后，再提取共享设备组件。
- 依赖版本由根构建统一解析，并记录实际选用的版本及环境。

## 变更说明与验证

说明受影响的用户流程、支持的输入、构建目标和可观察行为。提供相关的正确性测试与安装检查结果；涉及性能时，补充测量结果。设备测试结果应记录硬件、软件版本、拓扑、执行命令和计时范围。明确列出未执行的检查。

修改文档时保持相对链接有效，并同步中英文首页的内容。项目自身的 C++/AscendC 文件使用 `.cpp`/`.hpp` 扩展名。文本使用 UTF-8 编码和 LF 换行，具体规则见 `.editorconfig`。

## 开发命令

使用 Python 3.11 安装固定版本的开发工具，并启用本地 Git 钩子：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files --show-diff-on-failure
```

钩子检查行尾空白、文件末尾换行、YAML/JSON、文件大小、合并冲突标记、私钥标记和拼写。Ruff 检查 Python 常规错误并格式化代码；clang-format 格式化 C/C++/AscendC。格式化工具可能改写文件，应检查并暂存这些修改，再次运行检查。钩子在隔离环境中运行，首次使用需要网络，不依赖 Torch 或 NPU。

现有流水线通过 PR 评论 `compile` 触发，对 PR 的合并结果运行相同的全仓 pre-commit 命令，纯文档 PR 也会检查。任一钩子失败都会使作业失败，CI 不会自动提交修复。暂存文件和 PR 改动范围的检查命令见[开发指南](docs/development.md)。

API 契约测试已可用，见[测试说明](tests/README.md)。原生构建与设备测试入口仍在开发中，见[构建指南](docs/build.md)。

## 许可证

项目尚未选定许可证。引入依赖源码时应保留版权和许可证信息。采用的接口声明来源记录在[第三方依赖说明](third-party/README.md)中。
