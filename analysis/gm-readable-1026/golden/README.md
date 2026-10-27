# 独立Golden准入

候选为`8118a76`，即可读性GM＋slice完成通知；本轮只增加测试脚本，不改原生算法或误差阈值。

1. `prepare.py`先用冻结包中的独立NumPy参考生成CPU缓存；每rank全量token、全量H7168，不抽样。固定H7168/I3072/E384/top-k6、seed20260922、FP8/FP4、无共享专家。
2. EP8和EP32均覆盖balanced的BS32/96/256/512/1024/2048/4096，以及BS96 random/skewed。后两项覆盖空专家、负路由权重、偏斜与不同有效slice长度；capacity4096。
3. `launch.py`核对脚本/CPU准备身份，经全组空闲和锁门禁后才放行；使用已部署候选，不重建或安装共享环境。最终测试控制用标准库认证TCP协调器，真实设备通信仍是SHMEM/MTE。只做精度，不重复采性能。
4. `worker.py`调用公开API、仓库`benchmark.accuracy`与`golden.compare`：非零初值7，两次累加，检查完整输出、完整专家计数、重复输出与累加计数。仓库阈值保持`diff_thd=0.01,pct_thd=0.05,max_diff=0.1`；混合误差的定义以`golden.py`为准，不声称逐元素都小于1%。
5. `audit.py`核对全部rank退出、实际身份、九个case的完整报告和独立参考合同，拒绝缺测、抽样、阈值改变或失败。仅当EP8/EP32均为`full_golden_pass`，才允许备份旧main并以正常fast-forward更新；远端main若已变化，重新核对，绝不force覆盖。

已完成并通过，见[完整结果与提升依据](RESULTS.md)。原Gloo/Store失败和门禁取消保留原始记录。最终控制面先通过32rank纯CPU探针，再运行NPU；仍调用同一`benchmark.accuracy`/`golden.compare`，不修改阈值、数值或设备同步。

NumPy参考建模BF16/MXFP8舍入，不是Cube累加或exp的逐位模拟。通过范围不外推为EP64/128、共享专家或所有可能输入均已验证。原始日志、失败JSON、库/脚本哈希全部保留；输出等价测试与独立Golden分开记录。

本地证据根目录：`C:\Project\Infra\ops-transformer-hs\outputs\gm-readable-1026\golden-1027`。准入已完成，主干提升前保留旧main备份。
