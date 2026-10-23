# R31：保留配对、两轮AIC wave预算

父版：`codex/megamoe-packed-prefetch-original-math-1021`，`d986ac9dbe576af5e3d966617099cc9a0f302baf`。
实验分支：`codex/megamoe-r31-two-round-wave-1023`。

仅修改`csrc/forward.cpp::MakeTiling`：对已启用配对、BS>256的A8W4路径，
wave预算取`max(ceil(2*AIC/ceil(I/128)), ceil(AIC/ceil(H/256)))`。
原小BS、未配对和A8W8预算不变；不加入Combine配额，不改权重打包、数学、完成事件或GMM2一波滞后。

在H7168/I3072/AIC32下，6→3个M256组/波；均衡EP32/BS4096为96组，16→32波。
不是所有形状wave数都严格翻倍；专家尾块、向上取整和GMM2任务下界均需保留。
配对仍是gate/up各128列、物理N256，计算量不变。

CPU测试编译执行实际host预算分支，覆盖BS边界、其他后端、不同N/H及尾wave。
设备实验应固定原R31与候选、同节点交替复测，完整通信、timer-off；另采仅rank0完整32组被动timer及成对CSV。
本次不跑独立Golden，保存输入/输出指纹与有限值检查；不能标精度PASS。

历史paired父版4703.41→4275.18μs不是本分支成绩。R31的新性能结果需另行采集，不能继承9.10%收益。

## R31本轮结果（2026-10-23）

同组四节点、CANN B080、真实EP32/BS4096，原版→候选→候选→原版，各次完整退出。
warm10/采20取后10，主值逐轮rank-min均值，timer-off：

|版本|两次启动 μs|均值 μs|rank-max均值 μs|
|---|---|---:|---:|
|原R31 d986ac9|4796.2135 / 4786.8760|4791.5448|4799.2140|
|两轮wave 7512c6c|4255.7187 / 4258.4703|4257.0945|4263.4730|

降低534.4503μs（11.154%）；不与历史窗口拼均值，也没有本窗DS成绩。
控制/候选设备库与原R31 SHA均为`b3102af2bc82a227a98e264ea60bdc8cf8ddc52187569be97ce88fe1dab6cdd7`。
实际wave16→32，GMM1/GMM2总任务仍2304/2688、每AIC仍72/84。
rank0最后GMM1 ISSUE→EXPORT结束约345–353→168–169μs；区间可重叠，不当纯网络时间或相加账本。
候选timer-on差异+0.027%/−0.015%；负差视为波动，不宣称负开销。

256份原始off/on CSV逐轮重算通过；32rank输入/输出指纹一致且有限，不是独立Golden。
四轮退出后设备无本次worker残留；原R31解释器退出阶段的`__del__`告警保留，未更改计时内实现来消除它。
只验证本shape；其他EP/BS、随机/偏斜及独立精度仍待验证，不直接合入main/master。
详细编排和审计见`analysis/r31-two-round-wave-1023`；原始证据保存在本地`outputs/r31-two-round-wave-1023/r05-7341-a`。
