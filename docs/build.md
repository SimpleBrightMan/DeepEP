# 构建、打包与安装设计

**当前状态：构建系统尚未实现。** 本文的文件名、target、开关和命令是目标设计。仓库目前没有 `pyproject.toml`、`setup.py`、根 `CMakeLists.txt` 或可安装 wheel。

## 构建主链

```text
PEP 517 前端
  → setuptools（pyproject.toml / setup.py）
  → build_support.py：统一配置、构建、安装、收集
  → 一次根 CMake：核心与选定 Beta，共同解析依赖
  → Host 编译 + AscendC Device 编译
  → 安装暂存树
  → wheel / 源码安装 / editable 安装
```

metadata 查询只读取必要包信息。native 编译在实际构建阶段进行。不同架构、依赖、变体或 Beta 配置用独立构建目录和配置指纹隔离。

## 文件与目标

| 文件或模块 | 规划责任 |
| --- | --- |
| `pyproject.toml`、`setup.py` | 构建后端、Python 包元数据和扩展声明 |
| `build_support.py` | 一次根 CMake 调用及产物收集；各 hook 共用配置 |
| `CMakeLists.txt` | 架构、依赖、核心/Beta 目标与安装入口 |
| `cmake/Dependencies.cmake` | 依赖来源、固定版本与有效 targets |
| `cmake/HostTargets.cmake` | Python、Torch、torch_npu 与 Host ABI |
| `cmake/AscendC.cmake` | 所选 CANN 工具链的真实 Device 编译助手 |
| `cmake/Install.cmake` | 扩展、库、配置与构建信息的安装清单 |
| `VERSION`、`MANIFEST.in` | 版本来源与源码发行内容 |
| `dependencies.lock.json` | 实际依赖提交及工具链/运行环境配套 |
| `build.sh`、`develop.sh`、`install.sh` | 同一构建流程的薄入口 |

| 拟议 target | 输入 | 目标产物 |
| --- | --- | --- |
| `deepep_runtime` | `csrc/runtime/` | 核心/Beta 按需复用的 Host 运行库 |
| `deepep_kernels` | 核心 Kernel 与已形成的共享组件 | 工具链支持的 Kernel 链接/加载产物 |
| `deepep_python` | 注册、bindings、ops、execution | `deep_ep._C` |
| `megamoe_kernels` | Beta Device 实现 | 独立 Beta Kernel 产物 |
| `megamoe_python` | Beta 注册与 Host 操作 | `deep_ep_experimental.megamoe._C` |

Kernel 库的实际形式与后缀须用锁定的 AscendC 工具链验证。Host/Device 共用参数只维护一份声明。编译清单只消费真实实现及实际启用的变体。

## 环境与配置

目标运行/编译环境为 Ascend Linux 配套，需锁定驱动、CANN/AscendC、SHMEM、Python、PyTorch、`torch_npu` 和 Host 编译器。CATLASS/CatCCOS 仅在所选实现需要时引入。当前没有经过验证的版本组合。

规划的配置优先级：显式命令或环境覆盖 → release 预设 → 项目默认值。

| 配置 | 目标语义 |
| --- | --- |
| `DEEPEP_BUILD_MEGAMOE` | `OFF` 构建核心；`ON` 同时包含完整 Beta 内容 |
| `DEEPEP_NPU_ARCH` | 已验证的 NPU 编译目标 |
| `SHMEM_ROOT` | 选定 SHMEM SDK 的有效路径 |
| `configs/release/` | 发行功能、目标架构及依赖配套 |

这些选项尚未实现解析。最终配置与实际产物应记录在安装树 `_build_info.json`，包含源码版本、依赖来源、工具链、架构、变体、Python ABI 和 Beta 开关。

## 目标命令

仅在实现和配套环境就绪后使用；`CANN_ENV`、`DEEPEP_NPU_ARCH`、`SHMEM_ROOT` 由验证配套给出：

```bash
source "$CANN_ENV"
export DEEPEP_NPU_ARCH SHMEM_ROOT
python -m pip install -r requirements-dev.txt

DEEPEP_BUILD_MEGAMOE=OFF python -m build --wheel --no-isolation
# 使用独立构建目录或清洁工作树构建 Beta。
DEEPEP_BUILD_MEGAMOE=ON python -m build --wheel --no-isolation
python -m build --sdist --no-isolation

python -m pip install "$WHEEL"
python -m pip install --no-build-isolation -e .
```

源码发行必须包含完整第一方和实验源码，并提供许可允许、可取得的固定依赖方案。只有 `.gitmodules` 无法保证脱离 Git 元数据的源码包可重建。

## 安装验收

在干净环境分别检查核心与 Beta wheel 内容、独立加载、共享库查找、版本和构建信息；从 sdist 重建并核对功能清单。核心配置只安装核心内容，Beta 配置同步安装其源码对应产物、Python 包和配置。最后在目标 NPU 上运行完整示例及安装集成测试。

构建、安装及设备验证状态见[路线图](roadmap.md)。
