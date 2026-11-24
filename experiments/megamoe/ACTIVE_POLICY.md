# 当前有效策略与阅读入口

本页描述统一Wave＋GM的routed A8W4入口，不把历史实验或保留的兼容代码当成当前支持能力。

## 支持范围

- 主要入口：Ascend950、MTE/SHMEM、非量化Combine、SwiGLU、FP8激活/FP4权重，推理用途。
- 本兼容工作树在 A8W4 routed 路径接入 FP8 共享专家，并增加显式 clamp 传递。
  原 main 的已验证范围与本候选的待验收能力必须分开，见 [DeepInfer 兼容状态](DEEPINFER_COMPATIBILITY.md)。
- A8W8保留自己的原有入口和策略，不随本次A8W4整理改动。
- BS输入上限65536；每个shape仍须通过GM/SHMEM/UB容量检查，不保证上限用例一定能分配或发射。

## 策略表

| 决策 | 有效条件 | 作用 |
|---|---|---|
| 调度/结果通路 | 所有routed A8W4 | Wave＋GM；没有Linear或CV执行路径 |
| gate/up配对 | 全部合法BS | 逻辑N128对应物理gate128/up128 |
| 小BS输入协议 | 实际BS≤256 | Count/Epoch就绪；满足8卡节点/核数条件时使用rail路由 |
| 小BS Wave预算 | 实际BS≤256 | 保持已有连续提交预算 |
| 其他Wave预算 | 实际BS>256 | 两轮AIC预算，并保证GMM2最低任务量 |
| GMM2 lag | 实际BS≥512 | 延后一波；最后补齐pending wave |
| 供数分块 | 当前有效tile M≤128 | K512/N128；否则K256/N256 |
| 权重SF缓存提示 | 完整专家M | 不用slice M冒充完整复用范围 |

## 数量的作用域

- `bs` / `tokenNum`：本卡本次输入token数，决定输入协议与Wave/lag策略。
- `numMaxTokensPerRank`：输入buffer容量，可大于本次BS；不能用于策略分界。
- `expertTokenCount`：一个本地专家的完整接收行数，跨slice保持。
- `ExpertSlice::rowCount` / `ProblemConfig::m`：当前交给阶段执行的范围，不覆盖完整专家元数据。
- `actualShape.m()` / `mBlockActual`：当前计算tile的有效M；尾块可能选用另一套供数计划。

## 调用与完成

1. `api.py` → `forward.cpp` → `mega_moe.cpp`：输入、容量、tiling与实际派发。
2. `mega_moe_arch35.hpp::ProcessWave`：公共输入/输出阶段。
3. `mega_moe_wave_a8w4.hpp`：AIC/AIV0的`ProcessGmmWaves`；AIV1的`ProcessAiv1PriorityStages`。
4. `stage/mega_moe_gmm1_activation.hpp`与`stage/mega_moe_gmm2_combine.hpp`：地址、slice和阶段依赖。
5. `catlass_gmm/`：UB/L1/L0、权重转换、MMAD/FIX及owner交接。

纯策略集中在`common/mega_moe_schedule_policy.hpp`：Wave预算、lag使用本次实际BS。
`common/mega_moe_aiv1_ub_layout.hpp`逐项计算激活、prefix、route index、搬运环、metadata及Combine地址；
host检查容量，device按同一结果建立视图，不各抄一份偏移公式。

GMM1：Scalar issued → AIV1可进入等待 → **FIX full** → GM读入/SwiGLU/量化 → 释放epilogue credit。

GMM2：AIC在本核实际拥有输出的每个slice末，将完成事件排入FIX并发布issued → AIV1按相同owner/slice顺序等待**真实FIX完成** → 只发布该slice的ready → Combine。空生产核不发出或消费不存在的事件；不逐slice排空AIC Scalar。

AIV1进入GMM2完成等待前，先安排消费同一FIX队列中更早的GMM1（lag时包括下一wave），避免等待FULL反过来阻塞GMM1 EMPTY归还。计算与消费端共用`UseGmm2WaveLag`，最终退出仍回收最后credit。

Handoff只转移唯一资源owner和游标，不删除真实依赖。这里的slice-ready不是token-prefix提前归约。
