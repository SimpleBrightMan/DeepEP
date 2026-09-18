# MegaMoE Beta

将通信与专家计算放入完整实验单元，目标逻辑为 dispatch–GMM–激活–GMM–combine。当前尚无实现。

## 规划内容与职责

- `csrc/python_api.cpp`：独立实验扩展注册；`forward.hpp/.cpp`：Host 操作。
- `csrc/kernels/`：融合 Device 实现与变体；辅助头随实际模板和代码规模形成。
- `python/deep_ep_experimental/megamoe/`：独立 API、benchmark 和随包配置。
- `tests/`：Beta 业务正确性与完整流程；`CMakeLists.txt`：可选构建和安装。
- `DEEPEP_BUILD_MEGAMOE` 计划默认 OFF，ON 时同步选择实验扩展、包与配置。
- 核心与 Beta 共享根工程解析的依赖，并按需复用所有权明确的公共资源。
- Beta 接口、精度、负载和复现入口按版本记录，性能与相同语义的完整组合基线比较。

[完整文件规划](../../docs/repository-layout.md) · [架构说明](../../docs/architecture.md)
