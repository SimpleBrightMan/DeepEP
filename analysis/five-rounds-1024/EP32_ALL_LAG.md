# EP32全BS滞后一波：单变量实验

父算法904131c，分支`codex/megamoe-r31-two-round-wave-1023`。候选只将`ProcessGmmWaves`的条件改为`worldSize == 32 || BS >= 4096`；EP8/128保持原条件，Linear路径、gate/up、Wave预算、缓存、缓冲及flag协议不变。BS4096在两版均开启，是控制shape，不能把其波动解释为lag收益。

EP32、每卡BS32/96/256/512/1024/2048/4096，E384/H7168/I3072/topk6/shared0、FP8/FP4、balanced seed20260922。每次会话连续七BS、capacity4096；同一四节点按默认→全lag→全lag→默认完整ABBA，warm10/采20取后10，timer-off。保存每rank原始CSV、有效配置/native身份及输入/输出指纹，rank-min/rank-max分列；按要求不跑Golden。

测试执行实际生产C++编排，EP32七BS覆盖空专家、单波、跨专家尾块及多波，比较冻结lag调度器的事件顺序/范围/owner；EP8原阈值测试保留。CPU测试不能代替NPU同步正确性，最终以完整通信及跨版本输出回归为准。

证据根`outputs/ep32-all-lag-1024`；原始结果不覆盖历史。只在全组空闲时启动，不终止他人程序。结果未审计前不称收益或合入推荐基线。
