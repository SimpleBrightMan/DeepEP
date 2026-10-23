# R31两轮wave：实验复现

算法源码`7512c6c`，父版`d986ac9`；后续提交仅补编排/审计/结果，不改变被测生产代码。
只改host `MakeTiling`，保留配对、原数学、FP4打包、全部同步及原Combine服务粒度。

结果：EP32/BS4096原R31 4791.54475→4257.0945μs，降低11.154%；每版两次ABBA。
计时为完整通信kernel Duration，warm10/采20取后10的逐轮rank-min均值，另存rank-max。
没有同窗DS对照，不据此宣称全域最优；未跑独立Golden。

## 构建与运行边界

- `build.py`在新隔离目录构建候选与原R31 host扩展，共用同一设备库；包不安装进共享Conda。
- 控制host来自父提交仅包含`forward.cpp`的Git归档；生产源码除该文件外逐项核对一致。
- 原R31 Python/API与native协议8完整匹配；不混用旧prepared权重缓存。
- `run.py`固定原→新→新→原，复用冻结的四节点门禁/采集器，不自动重试或停止他人进程。
- 门禁/Store/采集器来自`codex/megamoe-paired-two-round-wave-1023`提交`2e18f4e`的`analysis/paired-wave-1023`；本轮记录每文件SHA。
- CLI中的`paired`标签表示本轮原R31，不是历史49e8174；`paired2`表示本轮7512c6c。身份以包注册表为准。
- `report.py`复核256份原始CSV、全部rank输入输出、实际16/32 wave和2304/2688任务覆盖；失败不写完成状态。

完整被动timer仅rank0，覆盖32组AIC/AIV0/AIV1；on测试的20次调用全部打点，off与on数据成对保留。
默认可视化隐藏wait、不压缩时间；native事件和稀疏原始buffer保留真实等待。
实际device库与原R31完全同SHA，排除“计算后端也被换了”这一混杂因素。

## 验证与限制

真实host预算C++测试1项、timer解析5项通过；设备构建、32rank预检、四次完整性能与原始证据审计通过。
指纹一致不是Golden；其他EP/BS和路由仍待验证。四节点退出后均无本次worker残留。
首次构建后的导入检查漏了先导入Torch；只修复构建脚本，复用已完成编译，失败日志保留。
一次恢复校验遇pytest生成的`__pycache__`，改为排除非生产缓存后继续，原始归档中的源文件逐项哈希未变。
R31解释器退出`__del__`告警保留；buffer显式destroy、预检记录和所有worker退出码均已核查。

本地证据`outputs/r31-two-round-wave-1023`包含包/编译日志；其`r05-7341-a`包含运行、CSV、timer和SHA清单。
凭据从已有授权清单只读进入内存，不随源码、日志或网页发布。
