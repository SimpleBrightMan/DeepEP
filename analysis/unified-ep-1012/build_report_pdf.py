"""Render the frozen-candidate report from verified measurement JSON."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

W=Path('C:/Project/Infra/ops-transformer-hs')
ROOT=Path(__file__).resolve().parents[2]
data=json.loads((W/'outputs/megamoe-best-unified-1012/report-data.json').read_text(encoding='utf-8'))
catalog=json.loads((Path(__file__).parent/'evidence/experiment-catalog.json').read_text(encoding='utf-8'))
out=ROOT/'output/pdf/MegaMoE_最佳候选_EP8_EP32_EP128_优化报告.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/msyhbd.ttc'))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')
INK=colors.HexColor('#16324F'); TEAL=colors.HexColor('#087E8B'); PALE=colors.HexColor('#EDF5F7')
styles={
 'title':ParagraphStyle('title',fontName='CNB',fontSize=22,leading=31,textColor=INK,spaceAfter=12),
 'h1':ParagraphStyle('h1',fontName='CNB',fontSize=15,leading=22,textColor=INK,spaceAfter=10),
 'h2':ParagraphStyle('h2',fontName='CNB',fontSize=11.5,leading=17,textColor=TEAL,spaceBefore=6,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='CN',fontSize=9,leading=15,spaceAfter=6,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CN',fontSize=7.5,leading=11,spaceAfter=5,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CN',fontSize=8,leading=12,wordWrap='CJK'),
}
story=[]
def p(text,style='body'):
    return Paragraph(escape(str(text)),styles[style])
def add(text,style='body'):
    story.append(p(text,style))
def table(rows,widths,green=None):
    head=ParagraphStyle('head',parent=styles['cell'],textColor=colors.white,fontName='CNB')
    cells=[[Paragraph(escape(str(v)),head if i==0 else styles['cell']) for v in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    commands=[('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
      ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
      ('LINEBELOW',(0,0),(-1,0),.7,INK)]
    for r in range(1,len(rows)):
        if r%2==0: commands.append(('BACKGROUND',(0,r),(-1,r),PALE))
    for col,row in green or []: commands.append(('BACKGROUND',(col,row),(col,row),colors.HexColor('#D6EEDB')))
    t.setStyle(TableStyle(commands)); story.append(t); story.append(Spacer(1,8))
def page(title):
    story.append(PageBreak()); add(title,'h1')

add('MegaMoE 性能优化探索报告','title')
add('EP8 / EP32 / EP128  |  2026-10-12  |  Ascend950 + CANN B070','h2')
if data['completed_cases']<15:
    add('阶段性报告：EP8/32 已完成；EP128 性能启动失败、待补测。精度按用户要求取消。历史 EP128 参考值不是本次候选结果。','h2')
add('冻结的是当前已验证的计算优化组合，不是所有实验的堆叠。本次不更改 main；整合分支包含原生代码 db2c017 和可复现统测脚本。')
add(data['plan']['branch'],'small')
rows=[['EP','BS/卡','本次候选 us','main参考 us','DS参考 us','较main降耗','较DS降耗']]
green=[]
for i,r in enumerate(data['rows'],1):
    fmt=lambda key: '待测' if r[key] is None else f"{r[key]:.2f}"
    pct=lambda key: 'n.a.' if r[key] is None else f"{r[key]:+.2%}"
    rows.append([r['ep'],r['bs'],fmt('mean_us'),fmt('baseline_us'),fmt('ds_us'),pct('gain_main'),pct('gain_ds')])
    if r['mean_us'] is not None:
        best=min(r['mean_us'],r['baseline_us'],r['ds_us'])
        green.append((2+[r['mean_us'],r['baseline_us'],r['ds_us']].index(best),i))
table(rows,[29,47,81,81,81,81,81],green)
add('绿色：该行列出的数值最低者。正百分比代表延迟降低，负值代表回退。参考列来自历史记录，并非本次同窗口成组消融，因此绿色不代表统计显著优胜。','small')
add('已完成的配置独立运行 2 轮，均为带通信的完整 MegaMoE kernel；每轮预热 10 次、测 20 次，取后 10 次逐轮 rank-min 的均值，再平均两轮。不是 Python/主机调用端到端。','small')
add('EP8 DS 参考为 10-12 三组交替复测均值；EP8 main 为 10-12 单轮筛选；EP32/128 main 与 DS 为 10-11 历史值。DS 历史数据未重新跑 CPU Golden。','small')

page('01  探索结论：做了什么，解决了什么')
add(f'已核验 {len(catalog)} 个不同原生候选的完整五 shape 筛选记录；预热、采样轮次、重复运行及 timer 诊断不计为额外优化实验。编号不连续，计划中的 e06/e07/e08/e10 与未做 NPU 验证的 e60 不计入完成数。')
add('结论一：小 BS 的主要突破来自供数和固定成本，而不是更换矩阵乘数学算法。wave 预算、FP4 位运算、流式权重缓存、UB 生产消费解耦及协同供数组合后，EP8 BS32/96/256 已在三组交替比较中快于 DS。')
add('结论二：大 BS 的重要有效方向是延长流水生命周期、保留异步真实完成通知；但纯计算约3924 us并不能直接推导完整通信也达到3924 us。最终归约、资源竞争、阶段推进和测量口径仍有影响。')
add('结论三：EP8全shape追平目标尚未完成。e55 与 DS 三组交替比较中，BS1024仍慢约9.94%，BS4096仍慢约2.01%；不能仅挑最快的一次或跨候选拼接结果宣称达标。')
add('结论四：迁移 DS 的思路必须匹配我们的资源协议。单独增加槽位、减少通知、立即消费GMM2、直接CV或更早归约，都没有自动转化为端到端收益。')
add('结论五：跨 EP 泛化需要重新验证。当前候选 EP32/BS4096 相对历史main回退约5.40%，且还不是同窗口A/B结论，因此暂不以“全EP最佳”名义合入main。')
add('报告阅读路线','h2')
add('先看首页统测结果与本页结论，再读测量口径；B1/B2是已在main的基础，O1-O11是本次保留组合。失败实验说明为何停止某方向；最后的逐候选账本可追溯每个实验、父版本和五shape数据。')
add('下一阶段补测范围','h2')
add('用户已确认：新增BS512、2048只用于EP128，双方都测，不是EP256。我们的EP128补齐32/96/256/512/1024/2048/4096；DS新增512/2048。其他五shape的DS参考仍按历史数据标注，除非另行完成新鲜复测。本次只测性能，不跑精度。')

page('02  统测口径与可信度边界')
add('配置与环境','h2')
add('每卡 BS=32、96、256、1024、4096；H=7168，专家中间维度=3072，总专家数=384，top-k=6，无共享专家；FP8 输入、FP4 权重、balanced 路由，seed=20260922。')
add('EP8：07柜 CPU7；EP32：07柜 CPU7/CPU1/CPU2/CPU3；EP128配置：07柜全部8节点 + 05柜全部8节点。已知07柜为 turbo2，未改硬件模式；05柜模式未独立核验。因此与历史值比较必须保留机器/日期差异。')
if data['completed_cases']<15:
    add('EP128 CPU Golden已生成，用户随后取消精度。性能首轮在05柜CPU8、rank120、设备0出现TsdOpen/rtSetDevice初始化失败，未产生有效性能汇总；需要先排查设备状态，再以独立输出目录补测，保留失败日志，不覆盖旧证据。不得重置设备或终止他人作业。')
add('所有节点使用同一 SHA256 的 wheel，解包到独立目录，不覆盖已安装环境。每个阶段先检查设备空闲；未终止他人任务。')
add('正确性与 timer','h2')
add('本次EP8/32共10个配置检查所有rank，每rank采样16 token与CPU量化感知Golden对比，并检查专家count/cumulative count；均通过。它是抽样精度，不等于全token验证。候选此前EP8五shape全量8-rank及skew BS1/random BS1031已通过。EP128本次按用户要求不验证精度，不能继承EP8的PASS。')
add('性能主数据来自 timer-off。第一轮性能测量后另跑 rank0 timer，第二轮不打点；timer 不混入表中时间，也不需要假设 timer 开销始终小于1%。rank0 仅能反映一张卡的流水，不能代表全部跨卡临界路径。')
add('为什么同时保留 rank-max','h2')
add('主指标沿用历史 rank-min 口径，便于对照；它不是集群最慢 rank 的完成延迟。Excel 同时给出逐轮 rank-max 均值及两轮均值之差，防止多卡尾部或测量波动被平均掩盖。只有两轮，不能宣称统计显著。')
add('版本与追溯','h2')
add('基线 main：'+data['plan']['base_main'],'small')
add('候选原生代码：'+data['plan']['native_commit'],'small')
add('统一 wheel SHA256：'+data['plan']['wheel_sha256'],'small')
add('服务器数据根目录：/home/zyx/unified-best-db2c017-1012；各阶段完整日志、全 rank profiler 和 timer 原始数据保留。仓库 analysis/unified-ep-1012/evidence 中保存统测汇总、配置、manifest 和精度/性能报告。','small')

page('03  已在 main 的基础优化：不要重复计算收益')
add('B1  GMM2 wave 内复用流水','h2')
add('出发点：每个专家 slice 创建/销毁流水，重复等待 FIX 并回收 L1/L0 credit。')
add('做法：在同一 wave 的非量化 routed GMM2 之间保留状态，wave 收尾统一等待 FIX，再发布 Combine 就绪；不把提交当完成。共享专家及其他路径保持原生命周期。')
add('分支：codex/megamoe-gmm2-wave-lifetime；设备提交 14a7a3e，验证归档 9eda05f。','small')
add('历史本地纯计算三轮均值：4118.181 → 4026.676 us，减少 91.505 us，约2.22%。这是纯计算收益，不等于带通信场景收益。')
add('B2  GMM1 异步完成通知 + wave 内复用','h2')
add('出发点：生产端 AIC 每 tile 等 FIX 完成后才继续提交，会压缩输出与下一 tile 的重叠窗口。')
add('做法：AIC 提交后继续推进；SwiGLU 通过真正的 FIX full 事件等待数据，用 empty/credit 通知缓冲可复用。流水在 wave 内保持，最终归还；禁止用 Scalar 发射时间代替数据完成。')
add('分支：codex/megamoe-gmm1-async-completion；设备提交 f0eb5c6，验证归档 00afaca。','small')
add('历史纯计算 A/B/A：父版本六轮均值4027.000 us，候选3923.563 us，减少103.437 us，约2.57%。这是异步协议及必要生命周期延长的组合收益。')
add('已经兑现的完整通信收益','h2')
add('10-10 EP8 三组交替验证：旧 main 的 BS4096 为4455.515 us，合入 B1/B2 等计算优化后的 main 为4254.149 us，降耗4.52%；五 shape 都通过全量精度且都有收益。这两项已包含在本报告 main24 参考列中。')
add('因此本报告 O1-O11 是在上述 main 基础上的后续优化，不能把 B1/B2 的收益再累加一次。纯计算3924 us和本次完整通信结果的输入及测量场景不同，不能直接相减当作通信时间。')
add('来源：experiments/megamoe/LOCAL_COMPUTE.md；EP8_MAIN_VALIDATION.md。','small')

for start in range(0,len(data['optimizations']),3):
    page(f'04  保留优化 {start+1:02d}-{min(start+3,len(data["optimizations"])):02d}')
    for item in data['optimizations'][start:start+3]:
        add(item['id']+'  '+item['title'],'h2')
        add('出发点：'+item['reason'])
        add('做法：'+item['method'])
        add('历史收益：'+item['benefit'])
        add('分支：'+item['branch']+'  @ '+item['commit'],'small')
        add('代码：'+item['code'],'small')
        add('约束：'+item['caveat'],'small')
        add('实验/来源：'+item['experiment']+'；历史分析分支 analysis/'+item['source'],'small')
        story.append(Spacer(1,4))

page('05  组合关系与未采用的实验')
add('收益不能简单相加','h2')
add('主线：main24 → wave2 + FP4 → 权重缓存 → UB 解耦 → 生命周期 handoff → 小 M 协同供数 → Unpermute 首项剥离 → 中大 BS 配对 → FP8 打包 + 归约缓存 → 激活 LAST。各实验的父版本不同，报告标明父版本才能区分真正增益与修复自身回退。')
add('例如：UB 解耦相对“单独整块转换”的收益很大，但该父版本本来就严重回退；应同时看相对 e22 的净收益。CV 输出改善单 credit 版本，也不代表比此前双 credit GM 更好。')
table([
 ['实验','结论 / 为什么未纳入'],
 ['e56 UB 预留调整','未证实有意义的收益，不扩大内存与编译布局变化。'],
 ['e57 单 credit GM','BS4096 约 4182 us，退于 e49 的约 4130 us。'],
 ['e58 直接 CV','BS4096 约 4130 us，较 e49 无明显净收益；不能把恢复单 credit 回退作为优化。'],
 ['e59 K-loop 特化','五 shape 精度通过，BS4096 4133.273 us；没有优于已有组合。'],
 ['e60 scale 转置','只有 CPU 几何检查；未构建、未 NPU 验证，严禁纳入“最好版本”。'],
 ['GMM2 立即消费 / 更细通知','去掉一 wave lag 或新增生产端 FIX 排空曾退化；更早通知不必然缩短端到端。'],
 ],[117,464])
add('历史 profiler 提示，而非确定归因','h2')
add('BS1024 rank0 样本：e49/DS kernel 1172.456/1063.946 us；MAC 1030.793/974.664，Scalar 466.204/271.207，MTE2 937.568/749.876，FIX 252.143/194.355 us。多个流水重叠，不能把这些差值相加，也不能把 Scalar 差值直接当作关键路径差值。')
add('仍需重点关注短批次内部供数接续、指令/同步固定成本和输出重叠；缓存提示有效不等于已经证明 cache-bound。统计字段缺少 cache hit/带宽时，不作缓存命中率结论。')

page('06  为什么 DS 更快：哪些假设被实验修正')
add('最初假设：GMM2短K放大每tile固定开销','h2')
add('GMM1的K=7168，GMM2的K=3072；启动供数、同步和输出成本不随K等比例减少。因此优先验证生命周期、UB轮转和L1/L0供数。这是有效起点，但不是证明全部差距都来自FIX。')
add('证据更新：e34之后，大BS不再表现为累计GMM ISSUE慢数百微秒','h2')
add('8-rank BS4096流水对比中，最晚AIC轨道的GMM1累计ISSUE与DS之差为-38.25至+4.54 us，GMM2为-25.07至-12.30 us；但完整kernel仍慢约150 us。这说明应把重点转向任务推进和尾段，不能沿用早期“GMM2多285 us”的账本描述优化后的版本。')
add('rank0 e34/DS：GMM窗口内非ISSUE累计150.771/92.570 us。但ISSUE不等于MMAD完成，非ISSUE也不等于硬件空转；该比较是单次诊断，不是可相加的因果归因。','small')
add('实验：DS提前归约是否解释剩余差距？','h2')
add('对DS原实现做on/off/off/on：BS1024重叠归约开启1061.421 us、关闭1045.463 us；BS4096开启4004.080 us、关闭4006.800 us。每种只有两轮，大BS约2.72 us差异落在波动量级。这个实验不支持“照搬提前归约就能省130-170 us”。')
add('禁用修改的是整个编译期协议，含metadata/readiness/最终归约分工，并非只移动一个sum；输出与原版指纹相同且有限值，但该DS实验未做数学Golden。它只能约束本EP8配置下的假设。','small')
add('反例：更早通知和更多重叠也可能更慢','h2')
add('e11去掉GMM2一wave滞后，BS4096增加337.036 us；e19每slice早发布Combine需额外FIX排空，增加32.849 us；e48缩短中等wave，BS1024增加133.510 us。e50把GMM2完成等待转移至AIV0，小BS反增约7 us，可能损害权重预取。')
add('因此下一步应减少关键路径上的真实依赖成本，而不是追求图上更早发出通知。新的异步设计须同时证明生产者推进、消费者可见性和缓冲释放；把等待移到另一个核，不代表等待消失。')
add('来源：ep8-parity-dstrace21-1012、dsreduce23-1012、round3/34/35/37 的原始记录与反思。','small')

page('07  EP32 大 BS 回退：已有线索与边界')
add('这版不能直接认定为全 EP 最佳','h2')
ep32large=next(r for r in data['rows'] if r['ep']==32 and r['bs']==4096)
add(f"EP32 / BS4096 两轮均值 {ep32large['mean_us']:.3f} us；历史 main {ep32large['baseline_us']:.3f} us，增加 {(ep32large['mean_us']/ep32large['baseline_us']-1):.2%}。其他 EP32 shape 的净收益见首页。此次优先交付真实反例，不把 EP8 收益外推。")
table([
 ['rank0 流水 / us','历史 main','本次候选','变化'],
 ['首个GMM之前',169.294,145.579,'-23.715'],
 ['GMM窗口',4047.547,4189.486,'+141.939'],
 ['末个GMM ISSUE之后尾段',424.676,562.675,'+137.999'],
 ['可见总跨度',4641.517,4897.740,'+256.223'],
 ],[200,108,108,117])
add('以上是两次 rank0 的独立 timer 调用，不是表中的 timer-off kernel，也不是同窗口 A/B。最终 AIC group 不同（main为7，候选为19），GMM窗口含等待；ISSUE终点不是真实FIX完成。只能用于提出假设，不能按表中增量直接归因硬件成本。')
add('优先验证顺序','h2')
add('1. 在空闲的同环境重新测 main24 / db2c017，排除日期、硬件状态和运行序列差异。目前未拿到独占窗口，不伪装成已做完。')
add('2. 先消融中大 BS gate/up 配对阈值，再消融跨阶段 handoff、归约 VF/缓存；每次只动一组边界，保持通信与输入一致。EP8每卡48专家，EP32每卡12专家，同样BS下每专家M更大，复用/调度代价不同。')
add('3. 尾段可能包含 FIX 完成、Combine、跨卡等待、最终归约；不能仅凭尾段增大说是网络慢。需要同rank、同事件语义配对，再决定是改计算后端还是收尾调度。')
add('来源：本次 ep32-perf1/bs_4096/timer/timer_breakdown.json；历史 /home/zyx/ep32-main-20261011-1030/bs_4096/timer/timer_breakdown.json。','small')

page('08  交付、使用与后续建议')
add('分支与来源','h2')
add('最终整合分支：'+data['plan']['branch'])
add('仓库：https://gitcode.com/Yancy_/DeepEP')
add('单项分支均在同一仓库。完整候选原生代码已经包含所有保留项；不要再把单项分支依次 cherry-pick，否则容易重复补丁或引入不同实验父版本。')
add('历史实验档案：codex/megamoe-ep8-parity-1012，analysis/ep8-parity-*；本次档案：analysis/unified-ep-1012。')
add('拉取与本机 EP8','h2')
for line in ['git fetch origin','git switch -c best-unified --track origin/'+data['plan']['branch'],
 'source /home/pkg/b070/cann-9.2.0/set_env.sh',
 'conda activate zyx',
 'python3 experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 32,96,256,1024,4096']:
    add(line,'small')
add('上述命令按公开脚本默认配置运行；报告中的严格复现协议是 analysis/unified-ep-1012/plan.json + run_node.sh。多机必须使用明确节点列表及可达的 master/SHMEM 地址；不要直接在其他机器照抄本次固定地址。')
add('下一步应该如何使用这份结果','h2')
add('1. 先看每个 EP 的五个 shape，而不是只看 BS4096。若某 EP 回退，先用同一窗口重新 A/B main 与候选；历史参考仅用于定位风险。')
add('2. 若要继续追 DS：优先复测 BS1024 的连续供数/Scalar 固定成本，再看 BS4096；EP8 小 BS 的优势不能推导为 EP128 必然更优。')
add('3. 若跨 EP 的小 BS 回退，应先隔离 wave2 提交预算和激活缓存策略的规模敏感性，不立即追加更多无条件缓冲。使用同一输入、同一 wheel 依赖、交替顺序测试。')
add('4. 合入 main 前补目标部署硬件下的全量/不均匀路由精度，并验证 B070 地址缓存提示的可移植性。此次只冻结候选、统测和交付，不自动替换 main。')

for start in range(0,len(catalog),14):
    page(f'附录  逐候选实验账本 {start+1}-{min(start+14,len(catalog))} / {len(catalog)}')
    add('单位us；每行是该候选完整五shape的一次timer-off筛选，不是多次运行取最小值。父版本不同，不能按相邻行计算单项收益。原始全rank报告、wheel哈希与来源哈希见 evidence/experiment-catalog.json。','small')
    rows=[['实验 / 父版本','32','96','256','1024','4096']]
    for item in catalog[start:start+14]:
        parent=item['parent'] or '见原始记录'
        rows.append([item['id']+' / '+parent]+[f"{item['means'][str(bs)]:.2f}" for bs in (32,96,256,1024,4096)])
    table(rows,[228,61,61,61,61,61])
    add('“screened”只说明构建、五shape精度与性能筛选完成，不等于保留或达标。保留的组合及分支见O1-O11；其他候选包括回退、无收益和弱证据项。早期无独立提交号的实验以父版本+补丁哈希+wheel哈希定位，不虚构分支。','small')
    add('e06/e07/e08/e10没有完整五shape筛选证据，e60仅CPU检查，均未计入。DS对照、消融、timer诊断和不规则路由回归另记，不为凑实验数重复计数。','small')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D3DFE7')); canvas.line(31,36,A4[0]-31,36)
    canvas.setFont('CN',7); canvas.setFillColor(INK)
    canvas.drawString(31,23,'MegaMoE | db2c017 | 全通信、timer-off 主指标 | 2026-10-12')
    canvas.drawRightString(A4[0]-31,23,str(doc.page))

# Landscape-width tables are intentionally fitted to portrait A4 usable width.
# Scale all table widths to the same 533 pt text column.
for item in story:
    if isinstance(item,Table):
        factor=533/sum(item._argW)
        item._argW=[v*factor for v in item._argW]
        item._colWidths=item._argW[:]
SimpleDocTemplate(str(out),pagesize=A4,rightMargin=31,leftMargin=31,topMargin=32,bottomMargin=49,
    title='MegaMoE 最佳候选 EP8/32/128 优化报告',author='MegaMoE performance study').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
