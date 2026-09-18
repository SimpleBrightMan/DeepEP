# 完整设备算子

保留完整 AscendC 算子入口、流水、模板实例化和邻接私有辅助。

## 规划内容与职责

- `legacy/`：layout、dispatch、combine 和低时延模式。
- `elastic/`：V2 dispatch、combine。
- `variants.json`：编译变体声明；`CMakeLists.txt`：真实编译单元与实例化。
- `arch.hpp` 仅在跨 Kernel 架构类型形成实际复用后创建。
- 同族辅助在算法族内组织，跨算子共享组件按真实契约进入 device/。

[完整文件规划](../../docs/repository-layout.md) · [架构说明](../../docs/architecture.md)
