# 第三方依赖

集中说明依赖来源和版本边界；当前没有下载或登记任何子模块。

## API 声明来源

本批 EP 接口签名参照 DeepSeek DeepEP 的提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`。源文件及 SHA-256 见 [API 基线](../tests/api/upstream_api.json)，对应 MIT 版权与许可全文保留于 [deepseek-deepep.LICENSE](deepseek-deepep.LICENSE)。这里仅复用接口声明，不包含上游 CUDA 执行实现；本项目自身的发行许可证仍由维护者确定。

## 规划内容与职责

- `shmem/`：首发通信依赖，选择固定源码或已安装 SDK。
- `catlass/`：按实现需要选择计算、架构或局部模板。
- `catccos/`：按实现需要选择通算融合模板。
- 实际地址、提交、许可证、来源和验证组合在接入时确定，并写入根依赖锁及发行记录。
- 上述源码目录暂不创建；确定版本后再作为独立依赖根接入。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
