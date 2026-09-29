# 构建、打包与安装

当前 wheel 包含 Host 运行时、Notify 元数据、展开式 dispatch 和 BF16 combine kernel，接口限制见 [Elastic 文档](elastic.md)。
详细范围、资源所有权和验收矩阵见[运行时设计](runtime-design.md)。
包名为 `ascend-deepep`，导入名仍为 `deep_ep`；不要与其他提供 `deep_ep` 的包装在同一环境。

## 前置条件

- Linux、Python ≥3.11、C++17 编译器、CMake ≥3.20，以及 Make 或 Ninja；构建工具须在 PATH 中。
- CANN 9.2.0 与配套驱动/框架，目标 Ascend950；其他组合尚未放开。
- 预先构建固定提交的 SHMEM SDK（见 `dependencies.lock.json`），安装根包含 `include/`、
  `lib/libshmem.so` 以及 `src/device`、`src/device_simt` 等 Device 头文件。
- CANN 安装中须包含 `tools/bisheng_compiler/bin/bisheng`，Device 使用 `-xasc` 编译。
- 普通 CPU 测试仅需 `requirements-dev.txt`；wheel 构建依赖固定在 `pyproject.toml`。
- 本仓不自动下载/编译外部 SDK，不将驱动/CANN/SHMEM 打入 wheel。SDK 及其传递依赖必须由部署环境提供。

```bash
# 在仓库根目录执行，先激活已安装匹配 torch/torch_npu 的 Python 环境。
source /path/to/ascend-toolkit/set_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export SHMEM_ROOT=/path/to/pinned/shmem-sdk
source "$SHMEM_ROOT/set_env.sh"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"

# 仅首次使用、且 SDK 中没有 deepep-sdk.json 时执行；已有 manifest 时跳过此命令。
python scripts/record_shmem_sdk.py --sdk "$SHMEM_ROOT" \
  --revision df2fa21daa43a6bc4c44672af9192a1247c31999

python -m pip install -r requirements-build.txt
```

版本声明由 SDK 提供方负责，脚本校验固定公共头文件、记录头文件及库的校验和；不把“头文件一致”等同于“二进制来源可信”。
若 SDK 无法匹配，必须提供固定提交的构建，不能修改 manifest 绕过检查。

`setup.py` 元数据查询不需要 SDK/框架。实际构建只经 `build_support.py` 调用根 CMake。
不同工具链、SDK、Python 环境和构建模式使用不同指纹目录；源码哈希仅用于标识产物，
不再因修改 Python、测试或单个 kernel 而切换整个构建目录。`DEEPEP_BUILD_MEGAMOE` 仅允许 OFF。
安装产物没有构建机绝对 RPATH，使用部署环境的 SDK 路径；加载时核对实际 SHMEM 二进制身份。

## 日常构建与后续算子接入

Host 编译器优先使用 `CXX` 指定的可执行文件；未设置时依次查找 PATH 中的
`c++`、`g++`、`clang++`、`bisheng++`、`bisheng`，再查找 CANN 的
`tools/bisheng_compiler/bin`。自动探测会跳过无法执行 `--version` 的候选；
显式指定的 `CXX` 无效则报错。保留 `c++ -> ccache` 等符号链接的调用名称，
不要把 `CXX` 设成裸 `ccache`。编译器是否满足 C++17 和链接要求仍由 CMake/实际构建验证。
日志会输出选中的 Host 编译器路径，Device kernel 继续使用 CANN Bisheng。

单机编译并安装仍使用：

```bash
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

这条命令会编译 Host 和 Device、生成 wheel 并安装到当前 Python 环境。
`requirements-build.txt` 只安装构建依赖，不能替代上述命令；修改 kernel 或 Python 包代码后重新执行上述命令。
只改 `tests/` 或文档时，更新共享源码即可，无需重新编译安装。

多机共享目录只需在一台机器生成 wheel，再在各节点的 Python 环境安装同一个文件：

```bash
python -m pip wheel -v --no-build-isolation --no-deps . -w /path/to/shared/wheels
python -m pip install --force-reinstall --no-deps /path/to/shared/wheels/实际文件名.whl
```

`pip wheel` 只编译打包，不安装。第一条仅执行一次，第二条在每个独立 Python 环境执行；
使用本次生成的实际文件名，不要硬编码旧版本，也不要在多台机器同时编译同一共享 checkout。
每台运行节点都需要加载匹配的 CANN/SHMEM 环境并设置运行时共享 endpoint，详见
[Elastic 单机/多机运行命令](elastic.md#编译与运行)。

若需要同时交付源码包与 wheel，可在安装构建依赖后执行
`python -m build --sdist --wheel --no-isolation`，产物位于 `dist/`；该命令同样不会安装。

以下选项同时作用于 Host 和 Device；默认 Release、4 个编译任务：

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8
export DEEPEP_BUILD_TYPE=RelWithDebInfo  # Release / RelWithDebInfo / Debug
```

支持 Ninja、Unix Makefiles 等单配置生成器。不同构建模式保留各自的缓存。
并行度改变不会切换缓存；内外层并行任务可能重叠，内存不足时降低并行度。
增量复用要求保留同一 checkout 的 `build/`；从新的源码包目录构建会使用新的缓存。

工程分工：

- `build_support.py`：环境校验、工具链缓存、调用 CMake、写入 `_build_info.json`。
- 根 `CMakeLists.txt`：Host 封装、Python 扩展 `_C.so`、能力声明、链接及安装。
- `cmake/Device.cmake`：每次进入独立 Device 构建，由子工程判断是否需要重新编译。
- `cmake/device/CMakeLists.txt`：Bisheng 源文件列表及编译选项，生成 `libdeepep_kernels.so`。
- `cmake/Dependencies.cmake`：Host/Device 共用的 CANN 和 SHMEM 库定位。

新增算子时，在 Device 子工程登记 kernel 源文件，在根工程增加 Host 封装及能力声明，
并添加 Python/native 绑定。不需要新建每算子子工程或修改 Python 构建器的算子列表。
编译器依赖信息负责追踪公共头文件；无改动、仅改 Python/测试时，不重新编译原生目标。
修改一个 kernel 时仅编译受影响的对象并重新链接，修改公共头文件时重编依赖它的对象。

wheel 中 `_C.so` 和 `libdeepep_kernels.so` 放在同一目录，通过 `$ORIGIN` 查找。
安装阶段会清除 setuptools 构建目录里遗留的旧 `libdeepep_notify.so`；包校验拒绝混入旧库。
CPU 构建预检和打包回归位于 `tests/runtime/test_build.py`，检查配置、缓存键、构建命令与产物要求；
这些测试不执行真实 Bisheng 编译，也不替代 Device 增量构建和设备验收。

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
当前 V2 算子精度请在安装后运行 `tests/elastic/test_dispatch.py` 和 `tests/elastic/test_combine.py`，
完整的 dispatch → 模拟专家计算 → combine 由后者覆盖，命令见 [Elastic 文档](elastic.md#编译与运行)。
普通 pytest 和本脚本默认的构建安装检查不会自动执行这两个 NPU 用例。

`--ep-roundtrip npu-stub|native` 以及 `scripts/validate_ep.py` 属于另一套 V1 验证入口：
前者中的 npu-stub 使用真实 NPU Tensor 和 HCCL 参考通信，native 则要求尚未实现的三个 `Buffer` 操作。
当前 wheel 仅声明两个 `ElasticBuffer` 操作，因此不要用 `--ep-roundtrip native` 验收当前 V2。
这套入口的完整命令和结果边界见 [NPU 集成测试](npu-roundtrip.md)。
正式发布前仍需许可证确认、受支持配套环境审批以及后续算法批次的独立资格验证。

## CI 接入

现有 `compile` 流水线自动运行 CPU 契约、构建预检、Python/原生资源故障注入与回归测试。
原生 fault-injection 的编译要求 Linux/g++；不是 CANN ABI 或设备测试。
原生构建/设备 runner 还需维护者配置 SDK、框架和独占设备资源，当前不伪造 runner 标签或镜像。
对应 runner 的必选门禁命令就是上面的 `validate_package.py`；依赖或能力缺失应失败，不应改成 skip。
归档 `source-tests.xml`、`installed-tests.xml`、`validation.json` 和每个 rank 的输出；wheel/sdist 作为独立产物留存。
