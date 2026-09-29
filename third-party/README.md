# 第三方依赖

集中管理依赖来源和版本边界。SHMEM、CATLASS 已登记为固定提交的 Git 子模块；不跟随上游分支自动升级。

## 已接入依赖

| 目录 | 用途 | 固定提交 |
| --- | --- | --- |
| `shmem/` | 通信 SDK，PR858 shared Jetty 与端点选路修复 | `b68a7252f4279c45b28aed20cc31f8ddde08e8c3` |
| `catlass/` | MegaMoE 编译时使用的模板头文件，不单独编译动态库 | `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1` |

来源见根 [.gitmodules](../.gitmodules)，版本约束见 [dependencies.lock.json](../dependencies.lock.json)。
SHMEM 使用 fork 的固定兼容提交；升级需重编 SDK 和 wheel，见[PR850 验证说明](../docs/shmem-pr850-validation.md)。
两个子仓库的许可证均为 CANN Open Software License Agreement Version 2.0，原文保留在各自的 `LICENSE` 中。
MegaMoE 源文件中的 CANN 版权和许可声明保留，许可全文归档于 [cann.LICENSE](cann.LICENSE)，并随启用 MegaMoE 的 wheel 分发。
项目级许可证遵循[贡献指南](../CONTRIBUTING.md#许可证)中的约定，目前尚未选定。

基础包按[构建说明](../docs/build.md)准备 SDK 并构建；MegaMoE 的便利入口见[实验使用说明](../experiments/megamoe/README.md)。
SHMEM 已有构建产物时，确认停止使用该 SDK 后可向准备脚本显式传 `--rebuild`。
下列分步骤入口仍保留给 SDK 准备与 CI 使用：

```bash
git submodule update --init --recursive
# 先加载 CANN 9.2.0 环境，再显式准备 SHMEM；不需要另外 clone CATLASS。
python scripts/build_shmem.py
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
```

以上命令从主仓库根目录运行。SHMEM 准备入口检查固定提交、干净源码和 CANN 版本，调用其上游构建脚本，
成功后生成 `shmem/install/shmem/deepep-sdk.json`。上游首次构建 Ascend950 会下载 nlohmann/json v3.11.3；
离线环境需预先提供 `shmem/3rdparty/json/single_include/nlohmann/json.hpp` 及对应源码。

已有 build/install 时默认停止，避免覆盖正在被其他作业使用的 SDK。明确需要重建时，先停止使用该 SDK 的作业，
再运行 `python scripts/build_shmem.py --rebuild`；上游会重新生成该子模块的 build/install，随后必须重建依赖它的 wheel。
源码存在修改时仍会拒绝，`--rebuild` 不会强制重置或覆盖源码。

未设置环境变量时，wheel 构建默认使用 `third-party/shmem/install/shmem` 与 `third-party/catlass`。
已有 SDK/源码可继续用 `SHMEM_ROOT`、`CATLASS_ROOT` 显式覆盖。`pip wheel` 本身不下载、不编译外部 SDK。
本入口按锁定版本和上游默认配置构建 SHMEM。

Python sdist/wheel 不携带这两个子模块源码或 SHMEM 动态库；从 sdist 构建时需设置上述两个外部路径（基础 OFF 构建不需要 CATLASS）。
SDK 和 CANN 由部署环境提供，完整编译命令见 [构建说明](../docs/build.md) 与 [MegaMoE README](../experiments/megamoe/README.md)。

## API 声明来源

本批 EP 接口签名参照 DeepSeek DeepEP 的提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`，并采用 Ascend 平台命名。源文件及 SHA-256 见 [API 基线](../tests/api/api_contract.json)，对应 MIT 版权与许可全文保留于 [deepseek-deepep.LICENSE](deepseek-deepep.LICENSE)。这里仅复用接口声明，不包含上游设备执行实现；本项目自身的发行许可证仍由维护者确定。

## 测试语义来源

CPU 参考和设备测试按同一固定提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8` 的以下源码语义重新实现：

- [deep_ep/utils/refs.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/utils/refs.py)：普通布局的 rank 去重、索引本地化和源 token 归并。
- [tests/elastic/test_ep.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/elastic/test_ep.py)：首次/cached 分发、展开布局、独立权重通道及不等长输入。
- [tests/legacy/test_intranode.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/legacy/test_intranode.py)：V1 layout→dispatch→combine 和 handle 复用。

实现位于 `tests/utils/`、`tests/test_reference.py` 及 V1/V2 设备用例中。改为无框架依赖的全 rank 小规模参考，增加人工算例、故障注入和零 token 场景；设备启动采用延迟导入的 NPU/HCCL 适配。不复制上游设备 collective、内部 metadata、FP8 布局或计时实现。原始 MIT 许可保留于 [deepseek-deepep.LICENSE](deepseek-deepep.LICENSE)。这些测试属于目标验收条件，不表示上游或当前 Ascend 后端已支持全部边界场景。

## 后续规划

`catccos/` 尚未接入，按实际实现需要再选择通算融合模板；不为未使用的功能增加子模块。
依赖更新时同时更新 gitlink、依赖锁及资格验证记录，不能只修改 `.gitmodules` 的分支名称。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
