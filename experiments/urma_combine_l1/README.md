# Data-only Dispatch/Combine + real GMM contention demo

本分支现在是完整通信/计算并发对照，替换旧的纯HBM→L1压力循环。生产`megamoe-urma`分支不改。

## 执行内容

```text
设备全卡同步 → Exp前置 → 本轮混合kernel

AIV1前EP−1个核（每核一个peer、QP0）：
    for 每256行：PUT(X, RO)；PUT(scale, RO)
    for 每256行：PUT(Combine BF16数据, RO)
    quiet(QP0)       # 唯一一次，确认Dispatch+Combine都完成

AIC + 配对AIV0（同一个compute开关）：
    for 主线计算batch：
        GMM1整批 / 对应W4反量化
        GMM2整批 / 对应W4反量化

本卡收尾同步 → 写计时记录 → 设备全卡同步
```

通信只有data PUT，不发SO/ready flag。Dispatch全部下发后立即下发Combine，不等待计算，不在中间quiet。仍使用主线`Put()`的URMA RO接口和单peer单核独占QP0，shared jetty关闭。

GMM直接调用主线`UrmaGmm1`/`UrmaGmm2`：包含默认L2策略、GMM1分段Gather、MX scale、W4反量化、L1/L0搬运、MMAD、GMM1输出及GMM2 Scatter。**不设置L2 bypass**。保留AIC/AIV0必要的L1槽位交接和阶段收尾；没有额外通算ready检查或逐tile软件完成通知。

使用主线balanced路由规则、真实M256分组、全局tile编号及每核归属。batch边界沿用主线（EP4为wave12，EP8/EP16为专家batch），每batch按GMM1→GMM2交替，不提前执行下一批GMM1。没有本核任务的batch自然跳过。

删除Pack、SwiGLU、Reduce和这些阶段的协议。GMM1输入、GMM2输入、Dispatch接收区、Combine发送区、GMM输出区相互隔离，并在计时前准备：取消等待不会造成读取未完成结果的竞争。输入/权重使用有限常量，占用和布局依照当前H3584/I1280/E384/TopK8场景；这不构成真实MoE数值精度测试。

## 编译

先source当前CANN包，在仓库根目录运行：

```bash
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export CATLASS_ROOT="$PWD/third-party/catlass"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
bash experiments/urma_combine_l1/build.sh
```

现在需要与MegaMoE相同的CATLASS和CANN Tensor API头文件；复用已有依赖，不需要安装wheel或重编SHMEM。仅支持Ascend950，启动28 AIC/56 AIV。S920静态`tiling_api`位于`platform`之前的链接顺序修复保留。

## 对照命令

`--compute`同时控制GMM1、GMM2和反量化；旧的`--l1-load`、`--l1-repeat`和`--l1-gap-cycles`不再适用。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
# EP4，组内：先只通信，再通信+真实计算
python3 experiments/urma_combine_l1/run.py \
  --ep-size 4 --devices 0,1,2,3 --bs 8192 --compute off,on \
  --warmup 10 --iters 20 --exp-repeat 50

# EP16
python3 experiments/urma_combine_l1/run.py \
  --ep-size 16 --bs 8192 --compute off,on \
  --warmup 10 --iters 20 --exp-repeat 50
```

只跑一组用`--compute off`或`--compute on`；ABBA顺序用`--compute off,on,on,off`。EP8同样支持。按实际设备组选择`--devices`：EP4选0,1,2,3只有组内；选0,4,8,12则全部是跨组（默认物理连续4卡一组）。启动器保存npu-smi快照，所选卡须已分配且空闲。

BS8192每peer行数为`8192×8/EP`。Dispatch每CV的X/scale分别为256×3584和256×112字节，Combine每CV为256×3584×2字节。没有self通信。

| EP | 每peer行数 | 每peer CV | 每peer Dispatch（含scale） | 每peer Combine |
|---|---:|---:|---:|---:|
| 4 | 16384 | 64 | 57.75 MiB | 112 MiB |
| 8 | 8192 | 32 | 28.875 MiB | 56 MiB |
| 16 | 4096 | 16 | 14.4375 MiB | 28 MiB |

## 统计和限制

每种模式使用新进程/通信域，分配相同的所有缓冲区。保留设备全卡同步、默认50次Exp、独立warmup/测量kernel。Exp默认两块FP32 `(100000,20000)` 缓冲区，合计14.90GiB/卡；计时不包含它。没有逐tile/per-CV timer，记录在本卡工作全部结束后写出。

- `TOTAL/INTRA/CROSS`：X+scale+Y发送量 / 从首个Dispatch PUT开始到最后quiet结束的同卡包络。这是有完成保证的整体有效带宽。
- `COMBINE_TOTAL/INTRA/CROSS`：Y发送量 / 首个Combine PUT到最终quiet的包络。**包含尚在途Dispatch的排空影响，不是独立Combine服务带宽**；off/on保持同样规则，可作为对照。
- `DISPATCH_SUBMIT_*`：仅下发跨度，不生成完成带宽。没有独立Dispatch完成探针。
- `GMM/DEQUANT`：所有参与核的首尾包络；打印每轮跨rank最大/最小包络的均值，不计算伪HBM带宽。
- `GMM active coverage`：各AIC活跃区间与通信包络的交集长度之和 / (28×通信包络)。查看计算是否覆盖了足够多的通信时间。

字节只算单向发送量，不将收发相加；不同rank绝对时钟不直接相减。结果在`benchmark-runs/combine_l1/...`，含rank日志/CSV、config.json、comparison.csv和summary.json。

最终逐字节验证Dispatch X/scale和Combine payload及未写self区域；不声称GMM数值精度通过。去掉生产者依赖和激活/Reduce后，此demo用于隔离资源竞争，不等价于MegaMoE端到端性能；输入就绪时序和缓存命中可能与融合算子不同。

2026-11-02：25机器CANN 9.2.0下完整设备kernel和Host程序编译链接通过；24组EP/BS纯CPU规划检查及3项统计/结构检查通过。未运行NPU，通信校验和GMM运行情况仍需设备实测。
