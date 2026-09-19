# 第三方依赖

集中说明依赖来源和版本边界；当前没有下载或登记任何子模块。

## API 声明来源

本批 EP 接口签名参照 DeepSeek DeepEP 的提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`。源文件及 SHA-256 见 [API 基线](../tests/api/upstream_api.json)，对应 MIT 版权与许可全文保留于 [deepseek-deepep.LICENSE](deepseek-deepep.LICENSE)。这里仅复用接口声明，不包含上游 CUDA 执行实现；本项目自身的发行许可证仍由维护者确定。

## 测试语义来源

CPU 参考和设备测试按同一固定提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8` 的以下源码语义重新实现：

- [deep_ep/utils/refs.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/utils/refs.py)：普通布局的 rank 去重、索引本地化和源 token 归并。
- [tests/elastic/test_ep.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/elastic/test_ep.py)：首次/cached 分发、展开布局、独立权重通道及不等长输入。
- [tests/legacy/test_intranode.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/legacy/test_intranode.py)：V1 layout→dispatch→combine 和 handle 复用。

实现位于 `tests/utils/`、`tests/test_reference.py` 及 V1/V2 设备用例中。改为无框架依赖的全 rank 小规模参考，增加人工算例、故障注入和零 token 场景；设备启动采用延迟导入的 NPU/HCCL 适配。不复制上游 CUDA collective、内部 metadata、FP8 布局或计时实现。原始 MIT 许可保留于 [deepseek-deepep.LICENSE](deepseek-deepep.LICENSE)。这些测试属于目标验收条件，不表示上游或当前 Ascend 后端已支持全部边界场景。

## 依赖规划

- `shmem/`：首发通信依赖，选择固定源码或已安装 SDK。
- `catlass/`：按实现需要选择计算、架构或局部模板。
- `catccos/`：按实现需要选择通算融合模板。
- 实际地址、提交、许可证、来源和验证组合在接入时确定，并写入根依赖锁及发行记录。
- 上述源码目录暂不创建；确定版本后再作为独立依赖根接入。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
