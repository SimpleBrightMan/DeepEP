# 构建、打包与安装

当前提供 **Host 运行时基础 wheel**，不含 dispatch/combine kernel。
详细范围、资源所有权和验收矩阵见[运行时设计](runtime-design.md)。
包名为 `ascend-deepep`，导入名仍为 `deep_ep`；不要与其他提供 `deep_ep` 的包装在同一环境。

## 前置条件

- Linux、Python ≥3.11、C++17 编译器、CMake ≥3.20。
- CANN 9.2.0 与配套驱动/框架，目标 Ascend950；其他组合尚未放开。
- 预先构建固定提交的 SHMEM SDK（见 `dependencies.lock.json`），安装根包含 `include/` 和 `lib/libshmem.so`。
- 普通 CPU 测试仅需 `requirements-dev.txt`；wheel 构建依赖固定在 `pyproject.toml`。
- 本仓不自动下载/编译外部 SDK，不将驱动/CANN/SHMEM 打入 wheel。SDK 及其传递依赖必须由部署环境提供。

```bash
source "$CANN_ENV"
export DEEPEP_NPU_ARCH=Ascend950
export SHMEM_ROOT=/path/to/pinned/shmem-sdk
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"

# 在 SDK 的专用副本中记录身份；已存在 manifest 时不覆盖。
python scripts/record_shmem_sdk.py --sdk "$SHMEM_ROOT" \
  --revision df2fa21daa43a6bc4c44672af9192a1247c31999

python -m pip install -r requirements-build.txt -r requirements-dev.txt
python -m build --sdist --wheel --no-isolation
```

版本声明由 SDK 提供方负责，脚本校验固定公共头文件、记录头文件及库的校验和；不把“头文件一致”等同于“二进制来源可信”。
若 SDK 无法匹配，必须提供固定提交的构建，不能修改 manifest 绕过检查。

`setup.py` 元数据查询不需要 SDK/框架。实际构建只经 `build_support.py` 调用根 CMake。
不同环境使用不同指纹目录；构建信息随 wheel 安装。`DEEPEP_BUILD_MEGAMOE` 仅允许 OFF。
安装产物没有构建机绝对 RPATH，使用部署环境的 SDK 路径；加载时核对实际 SHMEM 二进制身份。

## 自动化构建、隔离安装与验证

在已准备好匹配 torch/torch_npu、pytest 与构建依赖的环境中运行：

```bash
python scripts/validate_package.py --output /path/to/new-validation-directory \
  --system-site-packages
```

该命令创建新 venv，在其中自动安装固定的构建/测试依赖，运行源码测试，构建 wheel/sdist、从 sdist 再构建 wheel、检查内容与元数据，安装后在源码目录外运行测试。
无网络环境使用 `--wheelhouse /path/to/wheels`，只从预下载目录安装；缺包明确失败。
`--system-site-packages` **明确复用已有框架依赖**；不是宣称完全隔离所有 Python 依赖。
省略此选项创建不继承系统包的环境，需要另行准备框架/测试依赖，否则验证失败。
不会修改调用方 Python 环境，也不会默认占用设备。输出目录必须全新；失败日志/产物保留用于定位。

设备资源已由作业调度方分配后，可追加真实运行时验证：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
python scripts/validate_package.py --output /path/to/another-new-directory \
  --system-site-packages --runtime-ranks 2
```

设备编号和端口为示例，必须替换为本作业已预留资源。不要复用已有任务的通信端口。
支持显式选择 EP2/4/8 的资源测试；只有实际完成的规模才能记入资格。
每个 rank 的成功结果、JUnit XML 和 `validation.json` 一并保留；失败不会写成功摘要。

`--runtime-ranks` 只验证资源、事件与生命周期，不是 dispatch/combine 正确性或带宽。
dispatch → 专家计算 → combine 集成已增加独立选项 `--ep-roundtrip npu-stub|native`：
前者使用真实 NPU Tensor 和 HCCL 参考通信，后者仅调用安装包公开算法 API。
开发者也可用 `scripts/validate_ep.py` 跳过重复构建/安装，直接验收候选 wheel；
完整命令、算子接入要求和结果边界见 [NPU 集成测试](npu-roundtrip.md)。
正式发布前仍需许可证确认、受支持配套环境审批以及后续算法批次的独立资格验证。

## CI 接入

现有 `compile` 流水线自动运行 CPU 契约、构建预检、Python/原生资源故障注入与回归测试。
原生 fault-injection 的编译要求 Linux/g++；不是 CANN ABI 或设备测试。
原生构建/设备 runner 还需维护者配置 SDK、框架和独占设备资源，当前不伪造 runner 标签或镜像。
对应 runner 的必选门禁命令就是上面的 `validate_package.py`；依赖或能力缺失应失败，不应改成 skip。
归档 `source-tests.xml`、`installed-tests.xml`、`validation.json` 和每个 rank 的输出；wheel/sdist 作为独立产物留存。
