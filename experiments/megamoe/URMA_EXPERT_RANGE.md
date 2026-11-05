# 可变路由专家数（routing protocol7）

两个shared分支均支持运行时选择全局路由专家数E：

- EP仍为4、8、16；EP <= E <= 384，E必须被EP整除，TopK <= min(8,E)。
- 默认384；每卡专家等量且连续分布。共享专家数不计入E。
- 例如 E=64/128/256/384；E=12也可用于EP4，不能用于EP8/16。
- benchmark使用 --experts 256；算子接口使用 num_experts=256。
- 原有UB/邮箱/摘要容量保持不变；384仍是容量上限，不是每次的实际专家数。
  分桶、前缀、计数张量、Pack搜索和共享trace标识均使用实际E。
- 空专家owner跳过输入读取，但仍参加阶段同步。commfirst的gate/up融合判断使用实际E。
- 专家桶预留字节为 E * maxBS * TopK * 4；不根据Device实际路由做Host分配。

这是保持当前固定缓冲容量的首版范围，并非硬件专家数极限。
主仓lueelu/DeepEP main@89e19399260c16c483b3614f57d2e3c90eb9fc2e的公共接口：
全局E在1..16384、E整除EP、每卡最多128专家，因此固定EP时范围为EP..min(16384,128*EP)；
还须E>=TopK。该接口范围不代表所有规格均通过实测。
本次未扩展到该上限，避免同时扩大分桶UB/元数据布局与邮件槽。

Python/native routing protocol7需要完整重编安装wheel。
按用户此前要求未编译、未运行测试，没有为新专家数提供性能/精度结论。
