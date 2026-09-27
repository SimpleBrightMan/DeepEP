# 测试入口

当前 Elastic 算子只保留两个测试文件：

| 文件 | 内容 |
| --- | --- |
| `elastic/test_dispatch.py` | Notify 元数据、BF16/FP8 dispatch、weights、fresh/cached、路由和输出生命周期 |
| `elastic/test_combine.py` | dispatch → 专家输出 → BF16 combine 闭环、重复调用、事件依赖和精度 |

编译、单机和多机 Python 启动命令统一见 [Elastic 接口与限制](../docs/elastic.md)。
测试使用已安装包和直接 SHMEM 初始化，不需要 Gloo/HCCL 进程组。
Notify 的元数据验证已并入 fresh dispatch；通用未来功能矩阵不作为当前 Elastic 的运行入口。

CPU 检查：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not framework" -q
```

其中 dispatch 文件包含路由参考、地址范围、启动器和安装包导入的 CPU 检查；
combine 是显式运行的 NPU 脚本，不会被普通 pytest 当作设备验证执行。

仓库基础测试继续保留：`api/` 检查接口契约，`runtime/` 检查构建与资源生命周期，
`integration/` 检查集成链路，`legacy/` 保留尚未实现的 V1 设备资格入口。
`test_reference.py`、`test_precision.py` 等验证独立 CPU 参考和比较器；
`utils/` 是这些基础测试的公共辅助，不是新增的算子入口。

普通 pytest 不自动占用 NPU；CPU 通过不能代替真实 kernel 编译或设备精度结果。
完整构建安装流程见 [构建说明](../docs/build.md)。
