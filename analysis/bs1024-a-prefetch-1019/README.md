# A窗口提前预取：保留候选，尚未合入master

父版本master ae0176c；本次设备源码2adf28c。仅修改`block_mmad_a8w4.hpp::operator()`，把下一A窗口搬运排到当前A/scale完成通知之后、当前MMAD提交之前。首窗仍显式加载，后续窗不重复加载或获取empty。

## 依赖与资源

- 沿用3个64KiB A槽，L1总容量不变；没有改A/scale/B布局、缓存策略、tile、wave或MMAD顺序。
- 先排当前MTE2→MTE1完成，再预取下一A，避免未来槽的等待挡住当前消费。
- 下一A取得其原empty credit后才写入；当前槽仍在最后一次MTE1读取之后归还。
- 下一循环原完成通知覆盖已预取的A。跨tile/专家/阶段的槽号和退出回收保持原规则。
- 不包含无收益的AIV0跳过项，也不包含退化的提前scale预取。

## 性能证据

EP8、H7168/I3072/E384/topk6、无共享专家、FP8/FP4，真实通信，timer-off。warmup10、采20次取后10次、逐轮rank-min均值，另存rank-max。两轮为两次独立进程启动，镜像顺序中有其他独立候选。

|独立同机条件|master两轮均值 us|A-only两轮均值 us|延迟降低|
|---|---:|---:|---:|
|柜02 CPU3，BS1024 balanced|1159.5677|1150.1013|0.816% / 9.4664us|
|柜05 CPU7，BS1024 balanced|1161.6182|1149.0610|1.081% / 12.5572us|

两个节点结果分别计算，不混成一组样本；同为Ascend950DT/CANN B080/torch2.12.0+cpu/torch_npu2.12.0.post2，驱动分别26.2.0.b020与25.6.rc1.b010。

|补测筛选（各一轮）|master us|A-only us|解释|
|---|---:|---:|---|
|柜02 CPU3 BS4096 balanced|4130.0592|4130.5820|基本持平|
|柜02 CPU3 BS1024 random|1230.5759|1224.5452|快6.0307us，仍需更多重复|
|柜05 CPU7 BS96 balanced|655.4932|655.9050|基本持平|

上述有效运行八卡输出均与本机同输入master位级一致、有限值；未运行独立Golden，不称为完整精度通过。CPU预取地址/credit模型和原生构建通过。EP32/128、共享专家、BS32/256及更多随机种子尚未验证，不宣称统一基线验收完成。

## 流水解释与限制

CPU3独立trace轮，rank0前六组均值：GMM1 ISSUE累计650.028→641.480us，GMM2 303.313→302.643us；最后GMM2 ISSUE相对本组首GMM1开始1037.399→1028.533us。与收益主要落在GMM1供数接续相容。ISSUE不是纯MAC时间；不能把其他角色累计变化相加或认为改变了SwiGLU算术。

此轮A-only被动timer开销0.606%；master−0.102%属于测量波动。没有新增设备timer或同步。

柜05配对分核PipeUtilization已完成：device0、block0/15/16/31、每角色40条记录，按本核duration换算。AIC MTE2等效时间865.413→762.942us，Scalar434.862→447.556us，MAC953.340→950.239us，MTE1 610.660→611.111us；AIV0 Vector约415us不变。MTE2下降102us不意味着端到端省102us；实际普通性能改善12.6us，不能相加流水时间。未采新的Memory/L2组，不能声称物理流量或cache miss减少。

仍未证明追平DS。本候选仅回收约1%耗时，不能把历史DS节点的绝对耗时拼到新节点上。

## 本机复测

准备适配Ascend950的CANN与PyTorch环境，在本分支重新构建，不复用其他分支旧wheel：

```bash
python3 experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 1024 --no-timer
```

默认即性能模式；去掉`--no-timer`按原脚本保存流水。实验原始结果保存在本地工作区`outputs/bs1024-master-1019/cpu3/`与`cpu5r05/`；完整逐项说明见报告分支`codex/megamoe-bs1024-master-1019`的`analysis/bs1024-master-1019/FRONTEND_EXPERIMENTS.md`。
