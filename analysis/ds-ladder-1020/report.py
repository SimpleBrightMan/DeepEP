"""Render a concise report only from a locally verified ladder checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

TITLES={1:'Combine Scalar读metadata',2:'N半块提前FIX',3:'GMM2同向分核',
        4:'DS lookahead顺序＋旧完成发布',5:'共享两层异步epilogue协议',
        6:'CV原地压缩＋整stage写出',7:'保留数学合同的融合VF',
        8:'Dispatch Scalar写metadata',9:'Dispatch Scalar读索引',
        10:'Dispatch strided owner',11:'Dispatch源行互素打散',12:'小M L0C半槽固定64KiB间距'}

p=argparse.ArgumentParser()
p.add_argument('checkpoint',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--screen',type=Path)
a=p.parse_args()
checkpoint_label=str(a.checkpoint.resolve())
if os.name=='nt':a.checkpoint=Path('\\\\?\\'+checkpoint_label)
for item in json.loads((a.checkpoint/'manifest.json').read_text()):
    rel=PurePosixPath(item['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe checkpoint path')
    f=a.checkpoint/item['path']
    if f.stat().st_size!=item['size'] or hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError(f'Corrupt checkpoint: {f}')
summaries=list(a.checkpoint.glob('*-summary.json'))
if len(summaries)!=1:raise ValueError('Expected one checkpoint summary')
data=json.loads(summaries[0].read_text())
nodes=sorted(data['nodes'],key=lambda r:int(r['name'].split('-')[0][1:]))
best=max(nodes,key=lambda r:r['l0_reduction_pct'])
gap=100*(best['methods']['candidate']['mean_us']/best['methods']['ds']['mean_us']-1)
text=[
    '# DS逐级迁移/消融实验报告', '',
    f'完成{len(nodes)}个有效累计节点。当前本轮BS512候选中，{best["name"]}相对同窗口L0降低'
    f'{best["l0_reduction_pct"]:.2f}%，仍比同窗口布局优化DS慢{gap:.2f}%。**尚未追平DS，未合入master。**', '',
    '不是预设迁移必然有效：有效但变慢的节点继续保留在累计链；构建失败和修复重试另记，不计为优化实验。'
    '数学舍入、scale算法和route weight位置未悄悄改成DS，不把少做数值工作当成等价收益。', '',
    '## EP8 BS512累计结果', '',
    '完整通信融合kernel Duration，timer-off；H7168/I3072/E384/TopK6/shared0、FP8/FP4、balanced、seed20260922。'
    '预热10、采20取后10，逐轮rank-min再对两次独立启动求均值；rank-max及逐rank样本保留。'
    '顺序L0→父版→本版→DS→DS→本版→父版→L0（L1父版即L0，不重复启动）。', '',
    '|节点 / 本步迁移|L0 μs|直接父版 μs|本版 μs|布局DS μs|较父版降耗|较L0降耗|',
    '|---|---:|---:|---:|---:|---:|---:|']
for n in nodes:
    m=n['methods'];parent=m.get('parent',m['l0']);i=int(n['name'].split('-')[0][1:])
    text.append(f'|L{i} {TITLES[i]}|{m["l0"]["mean_us"]:.2f}|{parent["mean_us"]:.2f}|'
                f'{m["candidate"]["mean_us"]:.2f}|{m["ds"]["mean_us"]:.2f}|'
                f'{n["parent_reduction_pct"]:+.2f}%|{n["l0_reduction_pct"]:+.2f}%|')
text += ['',
    '各行是不同时间窗口，收益以本行对照为准；不能直接相加单项百分比。小于波动量的差值不宣称稳定收益。'
    '本轮最低值也不等于已通过七shape验收的最佳统一版本。', '',
    '## 目前最重要的结论', '',
    '- L4单独迁移DS任务顺序反而变慢；L5配套共享epilogue序列/真正FIX full与empty生命周期后改善，'
    '说明任务顺序与完成协议必须作为闭合系统审查，不能把某个flag或排列视为独立“神优化”。'
    'L5相对较慢L4的约4%不等于相对初始版的约4%。',
    '- L6在新协议上整stage写出有小幅额外收益；旧协议上同名策略曾无收益，体现交互作用。',
    '- 融合VF、Scalar metadata/索引、通信owner/源行打散以及L0C半槽间距的收益必须看表中真实结果。'
    '机制更接近DS，并不保证本工程剩余数据布局、完成边界和生成指令也已经相同。',
    '- 大差距仍未闭合。下一批优先迁移完整AIC compute leaf及GMM2 SF布局/供数协议；'
    '不是再盲目增加buffer。具体差异、代码入口与风险见[差异账本](DIFFERENCES.md)。', '',
    '## 中间版本与阅读入口', '',
    '初始版：`50d3ebe`，设备`dd5b2d7`。中间版本均独立分支；总工作分支`codex/megamoe-ds-ladder-1020`。'
    '原生测量以表中的设备提交为准；总分支随后只补报告和注释，不冒充重测。', '',
    '|节点|设备提交|独立分支|完整被动timer开销|', '|---|---|---|---:|']
for n in nodes:
    branch=f'codex/megamoe-ds-ladder-{n["name"]}-1020'
    text.append(f'|{n["name"]}|`{n["revision"][:7]}`|[{branch}](https://gitcode.com/Yancy_/DeepEP/tree/{branch})|{n["trace"]["overhead"]:.2f}%|')
text += ['',
    '每个节点：`candidate-a/timer_trace.json`为完整32组×AIC/AIV0/AIV1的96轨派生图；'
    '`timer_all_events.json`保留原始等待事件，派生图仅留白，不重排时间。'
    '`candidate-a/setup/rank_i/.../kernel_details.csv`为正式off CSV，`.../profiler/timer_on/...`为配对on CSV。'
    '图来自单次rank0观察，不是两次正式均值；ISSUE不是硬件完成，不能与PMU活跃时间混算。', '',
    f'本地证据：`{checkpoint_label}`，大小/SHA清单`manifest.json`已核验。'
    '保存各节点源码归档、实际kernel与host扩展、构建命令/日志、有效配置、逐rank输出差分和失败记录。'
    '统计摘要含每个节点实际kernel SHA及独立启动样本。', '',
    '## 验证与限制', '',
    '- 有效节点均通过8卡有限值/逐位差分；DS与其冻结DS输出差分，未声称我们与DS逐位相等。'
    '按用户要求未跑独立Golden；这不是正式数学精度验收，不能据此直接合入master。',
    '- CPU测试执行生产调度、行映射、credit或编排代码；不能证明硬件可见性。'
    '共享epilogue后32项、strided owner后33项、L0C容量检查后34项CPU测试，设备kernel和host扩展均需原生构建成功。',
    '- 本轮未新增task-block PMU，不复用历史PMU冒充本版计数器。正式性能、调试流水、PMU三种用途分开。',
    '- 小M闭合任务流仅在实际各专家M≤128、无共享专家且N2 tile数≤核数等条件成立时启用；'
    '其他场景保留原Wave。Host的CV选择仍使用均衡期望M，不保证偏斜性能。',
    '- 同一SO跨shape/路由筛选另列；不同分支的最低成绩不能拼接为一个不存在的统一实现。', '',
    '## 历史失败构建与修复重试', '',
    '以下均有对应修复后的有效节点，不是仍在运行或等待用户处理的实验。', '']
for n in data['incomplete']:
    text.append(f'- `{n["name"]}`：build={n["build_rc"]}，campaign={n["campaign_rc"]}。原样保留，不计为有效优化。')
text += ['', '执行顺序见[PLAN](PLAN.md)；调度、同步、UB边界的详细说明见[差异账本](DIFFERENCES.md)。', '']
for case in data.get('auxiliary',[]):
    text+=['## 最佳候选与DS的同shape配对流水', '',
        f'BS{case["bs"]}补采，仅用于定位；不计为新的优化实验，也不替换上面的正式累计表。', '',
        '|方法|timer-off μs|完整被动timer开销|', '|---|---:|---:|']
    for name,r in case['methods'].items():
        text.append(f'|{name}|{r["mean_us"]:.2f}|{r["trace"]["overhead"]:.2f}%|')
    text+=['', f'两份图分别在检查点的`trace-pair/l6/timer_trace.json`和`trace-pair/ds/timer_trace.json`；'
           '同目录保留各自原始事件及16份off/on CSV。仅rank0记录，但覆盖全部32组与AIC/AIV0/AIV1。', '']
    from trace_summary import summarize
    windows={name:summarize(a.checkpoint/r['label']/'timer_all_events.json')['group_means']
             for name,r in case['methods'].items()}
    text+=['rank0的32组AIC提交窗口均值：', '',
           '|位置|L6 μs|DS μs|差值 μs|','|---|---:|---:|---:|']
    for key,label in [('gmm1_issue_us','GMM1 ISSUE累计'),('gmm2_issue_us','GMM2 ISSUE累计'),
                      ('outside_issue_us','两类ISSUE之外'),('submit_window_us','首GMM1至末GMM2 ISSUE窗口')]:
        v,d=windows['l6'][key],windows['ds'][key]
        text.append(f'|{label}|{v:.2f}|{d:.2f}|{v-d:+.2f}|')
    text+=['','ISSUE不是硬件纯计算；首SF预取等细节边界未完全一致。此表只用于导航，'
           '不能把它与timer-off的端到端差值再次相加；优先核对GMM1供数/提交与其余控制区间，而非断言MMAD慢了30μs。','']
if a.screen is not None:
    label=str(a.screen.resolve())
    screen=Path('\\\\?\\'+label) if os.name=='nt' else a.screen
    for item in json.loads((screen/'manifest.json').read_text()):
        rel=PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe screening path')
        f=screen/item['path']
        if f.stat().st_size!=item['size'] or hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Corrupt screening evidence')
    if (screen/'campaign.rc').read_text().strip()!='0':raise ValueError('Screening incomplete')
    s=json.loads((screen/'summary.json').read_text())
    text+=['## 同一SO跨shape复查', '',
        'L6和累计L11各使用自己的同一个SO。下面每方法每shape仅一次启动，是筛选而非稳定收益证明；'
        'BS512的重复结果见前表。不将各shape最低值拼成一个虚拟版本。', '',
        '|BS/卡|初始L0 μs|L6 μs|累计L11 μs|布局DS μs|L6较L0降耗|',
        '|---|---:|---:|---:|---:|---:|']
    for r in s['cases']:
        m=r['methods']
        text.append(f'|{r["bs"]}|'+ '|'.join(f'{m[k]["mean_us"]:.2f}' for k in ('l0','l6','l11','ds'))+
            f'|{100*(1-m["l6"]["mean_us"]/m["l0"]["mean_us"]):+.2f}%|')
    text+=['', 'BS96路由回归：', '', '|路由|L0 μs|L6 μs|L11 μs|输出检查|','|---|---:|---:|---:|---|']
    for r in s['routing_regression']:
        text.append('|'+r['routing']+'|'+'|'.join(f'{r["methods"][k]["mean_us"]:.2f}' for k in ('l0','l6','l11'))+
                    '|8卡有限值/逐位差分通过，非独立Golden|')
    text+=['',f'跨shape本地证据：`{label}`。L11 BS1024与新DS BS512完整96轨流水、各自off/on配套CSV和DS JIT实物一并保存。', '']
a.output.write_text('\n'.join(text),encoding='utf-8')
print('REPORT_COMPLETE',len(nodes),best['name'],f'net={best["l0_reduction_pct"]:.2f}%',f'ds_gap={gap:.2f}%')
