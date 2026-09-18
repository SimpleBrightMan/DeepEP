# 仓库目录设计

本文列出规划中的源码目录与文件职责。部分文件尚未实现，功能进展见[路线图](roadmap.md)。可选组件和第三方源码在具体实现及依赖版本确定后接入。

```text
DeepEP/
├── README.md  # 中文项目首页、功能与快速开始
├── readme_en.md  # 英文项目首页、功能与快速开始
├── LICENSE  # 发布审核确认的项目许可证
├── NOTICE  # 实际采用组件的版权与分发声明
├── CONTRIBUTING.md  # 贡献、命名与开发约定
├── .gitignore  # 构建产物与本地配置忽略规则
├── .gitmodules  # 源码依赖地址；父仓记录固定提交
├── .clang-format  # 第一方 C++/AscendC 格式约定
├── .editorconfig  # 跨语言文本格式
├── pyproject.toml  # PEP 517 元数据与 setuptools 构建后端
├── setup.py  # Python 包发现与 CMake 构建桥接
├── build_support.py  # 一次根 CMake 调用、安装收集和 Beta 内容选择
├── CMakeLists.txt  # 原生工程入口与组件选择
├── VERSION  # 发行版本单一来源
├── MANIFEST.in  # 第一方源码与实验源码发行清单
├── requirements-dev.txt  # 构建与测试工具配套
├── build.sh  # wheel 构建的薄入口
├── develop.sh  # 开发安装的薄入口
├── install.sh  # 安装指定 wheel 的薄入口
├── dependencies.lock.json  # 组件、工具链与运行环境的配套锁定
├── deep_ep/
│   ├── __init__.py  # 公开 Buffer、ElasticBuffer、EPHandle、Config 与事件导出
│   ├── _native.py  # 集中加载 deep_ep._C 与配套原生库
│   ├── _version.py  # 读取发行版本与构建版本信息
│   ├── buffers/
│   │   ├── __init__.py  # Buffer 接口族导出
│   │   ├── legacy.py  # V1 调用与返回组织
│   │   └── elastic.py  # V2 调用、EPHandle 与其业务元数据
│   └── utils/
│       ├── __init__.py  # 公开工具导出
│       ├── event.py  # EventOverlap 的用户完成包装
│       └── envs.py  # 读取并校验本库环境选项
├── csrc/
│   ├── CMakeLists.txt  # Host、绑定与 Device 目标组合
│   ├── python_api.cpp  # deep_ep._C 单一模块注册入口
│   ├── kernel_args.hpp  # Host/Device 共用的 POD 参数与布局；单一声明源
│   ├── bindings/
│   │   ├── bindings.hpp  # 模块注册函数声明；原生绑定私有头
│   │   ├── legacy.cpp  # V1 原生对象和操作绑定
│   │   ├── elastic.cpp  # V2 原生对象和操作绑定
│   │   └── config.cpp  # Config 类型与默认参数的语言绑定
│   ├── runtime/
│   │   ├── context.hpp  # EP 操作共享的运行上下文声明
│   │   ├── context.cpp  # 上下文初始化及对资源集合的协调
│   │   ├── resources.hpp  # 本库持有的内存与通信资源集合声明
│   │   ├── resources.cpp  # 资源集合的取得、引用和释放适配
│   │   ├── event.hpp  # 原生完成对象声明
│   │   └── event.cpp  # 框架完成对象实现
│   ├── ops/
│   │   ├── config.hpp  # Config 的原生定义与默认值；与 Device 策略分工
│   │   ├── legacy.hpp  # V1 Host 操作声明
│   │   ├── legacy.cpp  # V1 参数与业务流程
│   │   ├── elastic.hpp  # V2 Host 操作声明
│   │   ├── elastic.cpp  # V2 参数与业务流程
│   │   ├── layout.hpp  # 路由布局的 Host 接口
│   │   └── layout.cpp  # 路由计数和布局准备编排
│   ├── execution/
│   │   ├── plan.hpp  # 执行请求、合法配置和准备结果的内部类型
│   │   ├── plan.cpp  # 按实现能力表与配置准备本次执行方案
│   │   ├── launcher.hpp  # Host 启动的私有接口
│   │   ├── launcher.cpp  # 将准备结果绑定到 Kernel 入口
│   │   ├── variants.hpp  # 已编译变体的查询接口
│   │   └── variants.cpp  # 查询构建生成的变体索引
│   ├── kernels/
│   │   ├── CMakeLists.txt  # AscendC 编译单元与变体实例化
│   │   ├── arch.hpp  # （按需）跨 Kernel 共用的芯片架构与局部资源类型；由实际复用形成
│   │   ├── legacy/
│   │   │   ├── layout.cpp  # V1 路由准备的 Device 入口及完整实现
│   │   │   ├── dispatch.cpp  # V1 高吞吐 dispatch 入口及完整实现
│   │   │   ├── combine.cpp  # V1 高吞吐 combine 入口及完整实现
│   │   │   ├── low_latency_dispatch.cpp  # V1 低时延 dispatch 入口及完整实现
│   │   │   └── low_latency_combine.cpp  # V1 低时延 combine 入口及完整实现
│   │   ├── elastic/
│   │   │   ├── dispatch.cpp  # V2 dispatch 入口及完整实现
│   │   │   └── combine.cpp  # V2 combine 入口及完整实现
│   │   └── variants.json  # 编译变体声明单一来源；生成索引与能力记录
│   └── device/  # （按需）共享设备源码组件；按实际复用抽取并确定文件名
├── cmake/
│   ├── Dependencies.cmake  # 依赖查找与按需构建 targets
│   ├── HostTargets.cmake  # Torch、torch_npu、Python 与 Host ABI
│   ├── AscendC.cmake  # 锁定 CANN 工具链的 Device 编译助手
│   ├── Install.cmake  # 本包扩展、库和构建元数据安装
│   └── build_info.json.in  # 实际配套和能力的构建信息模板
├── third-party/
│   ├── README.md  # 依赖来源、可选能力与版本说明
│   ├── shmem/  # （依赖）固定版本源码根或已安装 SHMEM SDK
│   ├── catlass/  # （可选）由所选计算/架构组件使用
│   └── catccos/  # （可选）由所选通算融合模板使用
├── configs/
│   ├── topologies/
│   │   └── example.json  # 设备分组、互联链路与共享资源的拓扑实例
│   ├── policies/
│   │   └── default.json  # 执行策略候选：分核、Tile、流水和重叠
│   └── release/
│       ├── core.json  # 核心发行预设和验证范围
│       └── with_megamoe.json  # （Beta）包含 MegaMoE Beta 的发行预设
├── docs/
│   ├── architecture.md  # 首发模块、依赖、目录和调用链
│   ├── api.md  # 公开 API 与 Ascend 参数语义
│   ├── build.md  # 构建、打包、安装与运行
│   ├── performance.md  # EP 与 Beta 测量口径和复现
│   ├── development.md  # 实现导航、命名、变体与测试
│   └── roadmap.md  # 功能资格与公共能力接入
├── figures/
│   ├── architecture.svg  # 中文首页及架构文档使用的架构图
│   └── architecture_en.svg  # 英文首页使用的对应架构图
├── tests/
│   ├── __init__.py  # 测试包入口
│   ├── utils/
│   │   ├── __init__.py  # 测试辅助包入口
│   │   ├── envs.py  # 测试使用的分布式环境准备辅助
│   │   └── reference.py  # 业务正确性参考
│   ├── api/
│   │   ├── upstream_api.json  # 选定原生接口基线、版本与覆盖记录
│   │   ├── test_exports.py  # 公开导出与类型身份
│   │   └── test_calls.py  # 典型用户调用工作流
│   ├── legacy/
│   │   ├── test_dispatch.py  # V1 dispatch
│   │   ├── test_combine.py  # V1 combine
│   │   └── test_low_latency.py  # V1 低时延调用
│   ├── elastic/
│   │   ├── test_dispatch.py  # V2 dispatch
│   │   ├── test_combine.py  # V2 combine
│   │   └── test_handle.py  # EPHandle 重用和布局衔接
│   └── integration/
│       ├── test_install.py  # 发行物的核心/Beta 加载
│       └── test_framework.py  # 北向框架调用集成
├── benchmarks/
│   ├── __init__.py  # 模块运行入口
│   ├── _utils.py  # 计时、预热、聚合和结果写入
│   ├── bench_ep.py  # V1/V2 通信性能
│   ├── tune.py  # （按需）对已有合法变体进行测量和候选搜索；随调优能力落地
│   └── results/
│       ├── README.md  # 测量字段与复现说明
│       └── template.csv  # 标准 EP 性能结果模板
├── examples/
│   ├── __init__.py  # 示例模块入口
│   ├── _distributed.py  # 分布式环境准备与退出
│   ├── legacy.py  # V1 完整调用用例
│   ├── elastic.py  # V2 完整调用用例
│   └── overlap.py  # 通信与独立计算重叠用例
└── experiments/
    └── megamoe/
        ├── README.md  # （Beta）Beta 定位、支持域和复现入口
        ├── CMakeLists.txt  # （Beta）独立目标与条件安装
        ├── csrc/
        │   ├── python_api.cpp  # （Beta）实验扩展注册入口
        │   ├── forward.hpp  # （Beta）MegaMoE Host 操作声明
        │   ├── forward.cpp  # （Beta）业务参数与启动组织
        │   └── kernels/
        │       ├── forward.cpp  # （Beta）融合 Device Kernel 入口
        │       ├── dispatch_gmm.hpp  # （按需）Beta 内的分发与 GMM 组合；按具体模板分文件
        │       ├── swiglu.hpp  # （按需）Beta 内的激活/量化实现；按实际代码体量分文件
        │       ├── gmm_combine.hpp  # （按需）Beta 内的 GMM 与归并组合；按具体模板分文件
        │       └── variants.json  # （Beta）Beta 变体声明
        ├── python/
        │   └── deep_ep_experimental/
        │       ├── __init__.py  # （Beta）独立实验命名空间
        │       └── megamoe/
        │           ├── __init__.py  # （Beta）实验功能导出
        │           ├── _api.py  # （Beta）当前版本的调用包装
        │           ├── benchmark.py  # （Beta）随包安装的性能体验入口
        │           └── configs/
        │               └── default.json  # （Beta）Beta 负载/策略示意
        └── tests/
            └── test_forward.py  # （Beta）Beta 业务正确性与端到端用例
```

## 文档与导航

文档索引位于 [docs/README.md](README.md)，各模块的 `README.md` 说明职责与关联文档。`.gitkeep` 用于保留尚无源码的目录，在加入实际文件后可移除。

## 实施规则

- 成对声明/实现是推荐位置，小型内部实现可合并，但责任边界保持一致。
- `csrc/device/` 目前仅有职责说明；具体组件名称由真实跨算子复用决定。
- 功能支持以对应实现和验证记录为准，统一维护在[路线图](roadmap.md)中。
- `.gitmodules` 及 `dependencies.lock.json` 在确定真实来源、固定提交和配套后建立。
- 生成扩展、库、构建信息及安装目录不作为源码占位文件提交。
