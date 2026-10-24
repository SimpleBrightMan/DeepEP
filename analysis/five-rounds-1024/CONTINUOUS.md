# 连续性能测试协议

2026-10-24 起按用户要求默认采用连续模式；不修改生产 kernel。

- 一次 worker 会话依次运行 BS32、96、256、512、1024、2048、4096，共用权重和最大 BS 容量的通信/工作区；再独立启动第二次完整会话。
- 每个 BS 独立 prepare、预热10次、采样20次、统计后10次；timer-off、真实通信、无共享专家，不运行 Golden。
- 分别报告逐轮 rank-min 和 rank-max，不能把 rank-min 称作全局最慢 rank 延迟。
- 每例保存实际 BS、buffer_capacity、会话 BS 列表、源码/native 身份、原始 CSV 和输入/输出指纹。整组 worker 正常退出后才验收该会话。
- 连续模式默认 capacity=max(BS列表)。capacity 不只是预留内存，也影响 Dispatch/SendMask 路由分批和 UB 配置；因此不能先验认定性能影响可以忽略。需要核对时，用同一空闲窗口的 `--isolated-bs` 或 `--buffer-capacity` 做容量对照。
- 每次独立启动前重新检查全组设备、外部 NPU 上下文和共享测试锁；发现他人任务就取消，不终止对方程序。跨柜结果分开记录，不拼成同窗对照。

## 启动接口

`campaign.py` 默认连续模式。指定 `--method-only control --ep 32 --bs-values 32 96 256 512 1024 2048 4096`，并按实际节点填写 inventory、已验证 host keys、归档、独立输出目录和端口。`control` 对应本轮冻结的目标算法 `904131c`，不是归档里的 combined 候选。

可通过 `--jump-rack 01 --jump-node CPU1` 跳转；补充 host keys 只能使用已认证可信主机保存的公钥，未知或冲突密钥拒绝。凭据仅从本地清单读入内存，不进入日志或版本控制。

## 本次进度

03柜 CPU4/6/7/8 已准备独立 zyx 环境及完整匹配的 SHMEM 运行库，原 hkq 未改。CPU 解析/报告测试通过不等于 NPU 验证通过；以 `outputs/r31-ep32-continuous-1024` 下实际会话、审计报告为准。失败尝试保留，不覆盖历史性能数据。
