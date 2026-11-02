# Combine CV256 + AIC HBM→L1 contention demo

基于 `megamoe-urma` 的 `5f371d1`。直接包含该版本的 `transport.hpp`，生产算子不改。

一个混合kernel启动28个AIC、56个AIV。AIV1前EP−1个核各负责一个peer，QP0、shared jetty关闭：

```text
所有Combine数据提前准备好
每个通信核：
    for 每个256行CV：
        PUT(data, RO, QP0)
        atomic_add(该CV的bit, SO, QP0)  // 主线的PublishCombineRange(..., count=1)
    quiet(QP0)                        // 仅最后一次

每个AIC（可关闭）：
    从互不重叠的GM范围，连续DataCopy到L1的4个64KiB槽
    全部提交后MTE2_S确认完成
```

SO flag沿用主线bitmap atomic-add，并非普通RO写flag。接收端在每轮启动前清零；前后设备全卡barrier防止下一轮清零与上一轮SO冲突。主线的数据生产、Pack、GMM、SwiGLU、Reduce均不运行，没有GMM ready等待。最终逐字节核对所有远端payload和SO位图，以及未触碰的self区域。

## 数据量

默认BS8192/H3584/I1280/E384/TopK8/EP16。通信按理想均衡的peer数据量，所有rank同时all-to-all，排除self。

| 项目 | 数量 |
|---|---:|
| 每peer行数 | 4096 |
| 每次data PUT | 256×3584×2 = 1,835,008 B = 1.75 MiB |
| 每peer PUT / SO次数 | 16 / 16 |
| 每peer发送量 | 28 MiB |
| 每卡本组 / 跨组 / 总发送量 | 84 / 336 / 420 MiB |

EP8/EP4也支持，每peer行数为 `BS×8/EP`，通信核数为EP−1；AIC始终28个。这里不加入dispatch、scale或元信息字节，不把收发相加，不发送self。

## AIC搬运模型的边界

当前主线W4权重由AIV0反量化送入L1，不能将整个右矩阵当作AIC直接从HBM读取的字节。首版选择**两次GMM左矩阵的FP8读取量**作为一份压力预算，包含N256投影之间的重复读取：

```text
R = BS×TopK = 65,536
GMM1 A读取 = R×3584×10 = 2,348,810,240 B
GMM2 A读取 = R×1280×14 = 1,174,405,120 B
合计 = 3,523,215,360 B = 3.28125 GiB / 卡 / pass
```

没有计入scale、AIV0权重读取/反量化、FIXPIPE输出写入或实际L2命中。
为了单独施加HBM压力，这些字节来自**独立、连续、每pass只读一次**的大缓冲区，并设置L2 bypass；没有复刻真实tile地址的复用和命中率。所以这是受控读压力实验，不是完整GMM的流量/性能模拟。

off/on都分配相同缓冲区，只切换AIC是否执行搬运。`--l1-repeat`改变总读取量；`--l1-gap-cycles`改变下发密度：大于0时每提交4×64KiB，等待该批MTE2完成再退避指定时钟周期。0时无内部Scalar完成等待，只在整个搬运结束时等待一次。

## 编译

使用当前DeepEP的CANN和已安装SHMEM SDK，无需重编译MegaMoE、安装wheel或改第三方源码：

```bash
cd /home/t00953803/projects/DeepEP
# 先source本机CANN环境，使ASCEND_HOME_PATH正确
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
bash experiments/urma_combine_l1/build.sh
```

仅支持Ascend950，SDK需要提供主线使用的RO/SO UDMA接口（本分支锁定PR944）。CMake单独构建demo；不修改系统安装。

S920的`libtiling_api.a`需要在链接顺序上位于`platform`之前，本demo已适配。若旧版本出现`PlatformInfoManager::Instance`/`DSO missing from command line`，拉取更新后重跑上述build脚本即可；无需删除SHMEM SDK。前面的宏重定义、deprecated warning不是这次链接失败的原因。

## 运行对照

所选卡必须已分配且空闲。启动器保存npu-smi快照，只清理自己创建的进程组，不重置卡。不要与真实MegaMoE同时占卡：AIC负载已在同一个demo kernel内产生。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
python3 experiments/urma_combine_l1/run.py \
  --ep-size 16 --bs 8192 --l1-load off,on \
  --warmup 10 --iters 20 --exp-repeat 50
```

分别运行：

```bash
# 仅Combine
python3 experiments/urma_combine_l1/run.py --ep-size 16 --bs 8192 --l1-load off
# Combine + 28 AIC HBM→L1搬运
python3 experiments/urma_combine_l1/run.py --ep-size 16 --bs 8192 --l1-load on
# 同样读取量，更慢下发；与gap=0比较
python3 experiments/urma_combine_l1/run.py --ep-size 16 --bs 8192 --l1-load on --l1-gap-cycles 10000
# 读压力覆盖得不够长时，增加读取量；此时已不是单份8192预算
python3 experiments/urma_combine_l1/run.py --ep-size 16 --bs 8192 --l1-load on --l1-repeat 2
```

默认devices为0..EP−1，可用 `--devices 12,13,14,15 --ep-size 4` 指定所分配的卡。`--group-size 4`按实际device ID连续4卡统计拓扑。ABBA顺序可用 `--l1-load off,on,on,off` 检查顺序/频率影响。

EP4示例：

```bash
python3 experiments/urma_combine_l1/run.py \
  --ep-size 4 --devices 0,1,2,3 --bs 8192 --l1-load off,on \
  --warmup 10 --iters 20 --exp-repeat 50
```

EP4每卡3个通信AIV1；每peer为16,384行、64次data+SO、112MiB，每卡总发送336MiB。AIC仍为28个，开启时每卡左矩阵读预算仍为3.28125GiB。0..3或12..15同属一个物理四卡组，CROSS显示N/A；如果选择0、4、8、12（分别来自四个物理组），这3条edge均统计为跨组。必须按实际物理拓扑选择device列表。

## 拉齐和计时

每种模式新进程/新通信域。设备全卡同步→50次Exp→连续10次warmup和20次测量，每轮都由Host启动一个新的混合kernel，前后有SHMEM设备全卡barrier，中间不做Host同步，最终统一回收数据。Exp默认FP32 `(100000,20000)`，两个缓冲区14.90GiB/卡；off/on同样分配。使用设备SHMEM barrier，不创建HCCL域，不等同于原测试器的HCCL all-reduce。

每个活动核仅记录开始/结束时间，所有核本轮工作结束后才写记录；没有逐CV打点或计时引入的PIPE_ALL。AIC结束必须等MTE2完成，通信结束必须quiet完成。时间戳默认1000tick/us，可用 `--ticks-per-us` 修改。

```text
每卡某类带宽 = 该类发送量 / mean_round(max(peer quiet end) − min(peer first PUT start))
round-max-mean = mean_round(max_rank(该rank本轮通信包络))
round-min-mean = mean_round(min_rank(该rank本轮通信包络))
```

TOTAL/INTRA/CROSS的包络各自计算，不能直接相加；L1是28核总读取字节/28核包络。不同rank绝对时间戳从不直接相减。另输出 `AIC active coverage of Combine`：28核搬运区间与通信包络的交集总长/(28×通信包络)。该值低说明读压力没有覆盖大部分通信，不能据“on没变慢”排除竞争。

结果含每rank日志、CSV、`config.json`、`comparison.csv`和`summary.json`。若充分重叠时on显著低于off，且减小读压力后恢复，支持HBM/片上通路竞争的解释；不能仅由这个实验断言具体瓶颈就是HBM控制器，也不能把该结果直接当真实GMM带宽。

2026-11-02：在177.22.25.1使用CANN `/home/pkg/b080/cann-9.2.0` 和已有隔离SHMEM SDK完成设备kernel及Host可执行程序编译/链接。3项离线统计与数据量检查通过。未运行NPU，不声称设备功能、数据正确性或性能已验证。
