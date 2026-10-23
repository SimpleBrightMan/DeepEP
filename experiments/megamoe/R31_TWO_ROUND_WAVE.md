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
