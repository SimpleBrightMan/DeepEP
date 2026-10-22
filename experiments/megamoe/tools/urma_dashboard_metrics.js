/* Offline URMA trace analysis. No dependencies or network access. */
(function (root) {
  'use strict';
  const DEFAULTS = {bs:4096, hidden:3584, intermediate:1280, experts:384, topk:8,
    ep:16, groupSize:4, dispatchBytes:1, combineBytes:2, peak:765, links:4,
    lineBW:56, efficiency:0.8, overhead:512, byteMode:'trace'};
  const NAMES = ['Pack','DispatchPutX','DispatchPutScale','DispatchQuiet','DispatchPublish',
    'SelfDispatch','WaitStartPeer','PublishStart','WaitLocalDispatch','WaitCombineReady',
    'CombinePut','CombineQuiet','SelfCombine','FinalPeerHandshake','WaitDispatch','Gmm1Tile',
    'Gmm1TileDrain','Gmm1Dequant','WaitActivation','Gmm2Tile','Gmm2Dequant','Gmm2Drain',
    'WaitGmm1','ActivationSlice','RankBarrier','Reduce','LocalSync','Gmm1Drain',
    'InputScaleGather','WaitInputScale','CombinePublish','WaitReduceReady','ReduceDrain',
    'Gmm1Tile','Gmm2Tile','Gmm1Batch','Gmm2Batch','Gmm1DequantBatch','Gmm2DequantBatch'];
  const COMPOSITE_IDS = {
    33:[15,16,27],34:[19,21],35:[15,16,27,14,29],36:[19,21,18],37:[17,27],38:[20,21]
  };
  const fmt = n => Number.isFinite(n) ? n.toLocaleString('en-US',{maximumFractionDigits:6}) : '不可计算';
  const finite = n => typeof n === 'number' && Number.isFinite(n);
  const metric = (label,value,unit,detail,estimated=false) => ({label,value:finite(value)?value:null,unit,detail,estimated});
  function union(intervals) {
    let sum=0, end=-Infinity;
    for (const [a,b] of intervals.slice().sort((x,y)=>x[0]-y[0])) {
      sum += Math.max(0,b-Math.max(a,end)); end=Math.max(end,b);
    }
    return sum;
  }
  const sum = list => list.reduce((a,b)=>a+b,0);
  function bounds(rows) {
    let start=Infinity,end=-Infinity;
    for(const row of rows){start=Math.min(start,row.ts);end=Math.max(end,row.end);}
    return {start,end,span:rows.length?end-start:0};
  }
  function validateParams(input,doc,rankIds) {
    const p={...DEFAULTS,...input};
    for(const key of Object.keys(DEFAULTS).filter(k=>typeof DEFAULTS[k]==='number')) {
      p[key]=Number(p[key]);
      if(!finite(p[key]) || p[key]<(key==='overhead'?0:Number.MIN_VALUE)) throw Error(`参数 ${key} 必须为有效${key==='overhead'?'非负':'正'}数。`);
    }
    for(const key of ['bs','hidden','intermediate','experts','topk','ep','groupSize','links'])
      if(!Number.isSafeInteger(p[key])) throw Error(`参数 ${key} 必须为正整数。`);
    if(doc.metadata?.kind==='urma_allrank' || rankIds.length>1)
      p.ep=Math.max(rankIds.length,Math.max(...rankIds)+1,Number(doc.metadata?.rank_count)||0);
    if(p.ep<=Math.max(...rankIds)) throw Error('EP 必须大于最大 rank 编号。');
    if(p.groupSize>p.ep) throw Error('每组卡数不能大于 EP。');
    if(p.topk>p.experts) throw Error('TopK 不能大于专家数。');
    if(p.efficiency>1) throw Error('协议效率应在 (0,1]。');
    if(!['trace','model'].includes(p.byteMode)) throw Error('无效的计算模式。');
    return p;
  }
  function sourceScopes(e,clock) {
    if(!Array.isArray(e.args.source_records)) {
      if(e.id>=35&&e.id<=38)throw Error('GMM batch 缺少 source_records，不能把含等待的跨度当作核时。');
      return [e];
    }
    if(!e.args.source_records.length)throw Error('GMM batch 的 source_records 不能为空。');
    if(!finite(clock)||clock<=0) throw Error('合并的 GMM 记录缺少 clock_mhz，不能准确拆解核时。');
    const base=BigInt(e.args.start_tick),limit=BigInt(e.args.end_tick);
    let previous=base;
    return e.args.source_records.map(r=>{
      const start=BigInt(r.start_tick), end=BigInt(r.end_tick);
      const allowed=COMPOSITE_IDS[e.id]||[];
      if(!allowed.includes(r.event_id)||end<start||start<previous||end>limit) throw Error('GMM 源记录非法、越界或重叠。');
      previous=end;
      const ts=e.ts+Number(start-base)/clock;
      // Legacy 33/34 records only carried ticks; new batches carry each tile's
      // complete geometry. Never reuse one batch's geometry for every tile.
      return {...e,args:{...e.args,...r},id:r.event_id,name:NAMES[r.event_id],ts,end:ts+Number(end-start)/clock,dur:Number(end-start)/clock};
    });
  }
  function coreBusyTimes(rows) {
    const lanes=new Map();
    for(const e of rows){if(!lanes.has(e.tid))lanes.set(e.tid,[]);lanes.get(e.tid).push([e.ts,e.end]);}
    return [...lanes].map(([tid,ivs])=>[tid,union(ivs)]).sort((a,b)=>a[0]-b[0]);
  }
  function busy(rows,label,detail) {
    const totals=coreBusyTimes(rows);
    const best=totals.reduce((a,b)=>b[1]>a[1]?b:a,[-1,0]);
    return metric(label,totals.length?best[1]:null,'µs',[
      detail,'按核心将所选条累加；若输入存在重叠，先取同核区间并集，防止双计。',
      ...totals.map(([tid,t])=>`block ${Math.floor(tid/3)} / ${['AIC','AIV0','AIV1'][tid%3]}：${fmt(t)} µs`),
      totals.length?`max(以上 ${totals.length} 个核心) = ${fmt(best[1])} µs，tid=${best[0]}。`:'未找到对应记录。',
      '排除条与条之间的空隙和独立 Wait；条内的等待/流水阻塞无法由此 JSON 再细分，不能视为纯硬件忙碌时间。']);
  }
  function aicBusyRange(scopes) {
    const totals=coreBusyTimes(scopes.filter(e=>e.tid%3===0&&[15,16,27,19,21].includes(e.id)));
    const longest=totals.reduce((a,b)=>b[1]>a[1]?b:a,[-1,-Infinity]);
    const shortest=totals.reduce((a,b)=>b[1]<a[1]?b:a,[-1,Infinity]);
    return metric('AIC 不带空泡时长极差',totals.length?longest[1]-shortest[1]:null,'µs',[
      '先对本 rank 每个 AIC 分别累计 GMM1 + GMM2（包含各自 Drain），再计算最长核时 − 最短核时；不是分别取两个阶段的最大值再相加。',
      '合并条按 source_records 拆解；同核区间取并集去重，排除条间空隙和独立 Wait，不计 AIV 反量化。',
      ...totals.map(([tid,t])=>`AIC block ${Math.floor(tid/3)}（tid=${tid}）：${fmt(t)} µs。`),
      totals.length?`最长：block ${Math.floor(longest[0]/3)}，${fmt(longest[1])} µs；最短：block ${Math.floor(shortest[0]/3)}，${fmt(shortest[1])} µs（并列时显示编号最小的核）。`:'未找到 AIC GMM 记录。',
      totals.length?`极差 = ${fmt(longest[1])} − ${fmt(shortest[1])} = ${fmt(longest[1]-shortest[1])} µs。`:'极差不可计算。',
      `统计 ${totals.length} 个有 GMM 记录的 AIC；缺失核心不按 0 补齐。条内等待无法进一步扣除。allrank 均值是各 rank 极差的算术均值。`
    ]);
  }
  function analyzeRank(rank,rows,meta,p) {
    const warnings=[], metrics={};
    const clock=meta.clock_mhz;
    const scopes=rows.flatMap(e=>sourceScopes(e,clock));
    const aic=scopes.filter(e=>e.tid%3===0 && [15,16,19,21,27].includes(e.id));
    const aicBounds=bounds(aic), allBounds=bounds(rows);
    const total=finite(meta.envelope_us)&&meta.envelope_us>=0?meta.envelope_us:allBounds.span;
    metrics.fused=metric('融合算子诊断时间',total,'µs',[
      finite(meta.envelope_us)?`使用 metadata.envelope_us = ${fmt(total)} µs；保留已隐藏 Wait 的原始首尾范围。`:
        `可见事件最后结束 ${fmt(allBounds.end)} − 最早开始 ${fmt(allBounds.start)} = ${fmt(total)} µs。`,
      '这是 JSON 对应的诊断调用跨度，包括 Pack、计算、通信、Reduce 及已记录的尾部同步；不含未打点的 kernel 入口/出口。']);
    metrics.aicSpan=metric('AIC 工作跨度（带空泡）',aic.length?aicBounds.span:null,'µs',[
      `本 rank 全部 AIC 的首个 GMM 下发开始：${fmt(aicBounds.start)} µs。`,
      `全部 AIC 的最后一个 GMM/drain 结束：${fmt(aicBounds.end)} µs。`,
      `T_AIC = ${fmt(aicBounds.end)} − ${fmt(aicBounds.start)} = ${fmt(aicBounds.span)} µs。`,
      '取整个 AIC 阵列的首尾包络，包含中间等待和空泡；不累计并行核心时长，也不含首个 GMM 之前的输入等待。']);
    metrics.dBusy=busy(scopes.filter(e=>[1,2,3,4,5].includes(e.id)),'Dispatch 最长累计核时','范围：DispatchPutX / PutScale / Quiet / Publish；历史 SelfDispatch 如存在也计入。');
    metrics.cBusy=busy(scopes.filter(e=>[10,11,12,30].includes(e.id)),'Combine 最长累计核时','范围：CombinePut / Quiet / Publish；历史 SelfCombine 如存在也计入。');
    metrics.g1Busy=busy(scopes.filter(e=>e.tid%3===0&&[15,16,27].includes(e.id)),'GMM1 最长累计核时','范围：AIC Gmm1Tile + TileDrain + Gmm1Drain。合并条按 source_records 拆解，避免把打点间隙算进来。');
    metrics.g2Busy=busy(scopes.filter(e=>e.tid%3===0&&[19,21].includes(e.id)),'GMM2 最长累计核时','范围：AIC Gmm2Tile + Gmm2Drain；AIV 反量化不重复算作 AIC 计算。');
    metrics.aicBusyRange=aicBusyRange(scopes);
    function bandwidth(stage) {
      const putIds=stage==='D'?[1,2]:[10],quietId=stage==='D'?3:11;
      const pending=new Map(), batches={intra:[],cross:[]};
      const payload={intra:0,cross:0},incomplete={intra:0,cross:0},unknown={intra:false,cross:false};
      const estimated={intra:false,cross:false}, peers={intra:new Set(),cross:new Set()};
      let missingPeer=0;
      for(const e of rows.filter(e=>putIds.includes(e.id)||e.id===quietId).sort((a,b)=>a.ts-b.ts)) {
        const peer=e.args.peer;
        if(!Number.isInteger(peer)||peer<0||peer>=p.ep){missingPeer++;continue;}
        if(peer===rank)continue;
        const kind=Math.floor(peer/p.groupSize)===Math.floor(rank/p.groupSize)?'intra':'cross';
        const key=`${e.tid}/${peer}`;
        if(putIds.includes(e.id)) {
          let bytes=e.args.bytes;
          if(finite(bytes)&&bytes<0)throw Error('PUT payload 字节数不能为负。');
          if(p.byteMode==='model'||!finite(bytes)||bytes<=0) {
            estimated[kind]=true;
            const n=e.args.rows;
            bytes=finite(n)&&n>=0?n*(e.id===2?p.hidden/32:p.hidden*(stage==='D'?p.dispatchBytes:p.combineBytes)):NaN;
          }
          if(!finite(bytes)||bytes<0){unknown[kind]=true;bytes=0;}
          payload[kind]+=bytes;peers[kind].add(peer);
          if(!pending.has(key))pending.set(key,{start:e.ts,bytes:0,kind,cvs:new Set()});
          pending.get(key).bytes+=bytes;
          pending.get(key).cvs.add(e.args.cv);
        } else if(pending.has(key)) {
          const item=pending.get(key);
          if(e.args.cv!=null&&!item.cvs.has(e.args.cv))continue;
          batches[item.kind].push([item.start,e.end]);pending.delete(key);
        }
      }
      for(const item of pending.values())incomplete[item.kind]++;
      for(const kind of ['intra','cross']) {
        const ivs=batches[kind],time=union(ivs),env=ivs.length?bounds(ivs.map(([ts,end])=>({ts,end}))).span:0;
        const unavailable=unknown[kind]||incomplete[kind]||missingPeer||!ivs.length||time<=0;
        const key=(stage==='D'?'d':'c')+(kind==='intra'?'Intra':'Cross');
        metrics[key]=metric(`${stage==='D'?'Dispatch':'Combine'} ${kind==='intra'?'组内':'跨组'}带宽`,unavailable?null:payload[kind]/time/1000,'GB/s',[
          `rank ${rank}；本组 floor(${rank}/${p.groupSize}) = ${Math.floor(rank/p.groupSize)}；对端：${[...peers[kind]].sort((a,b)=>a-b).join(', ')||'无'}。`,
          `${p.byteMode==='trace'?'优先取 args.bytes；缺失时按 rows 与参数估算':'按 rows 与参数重算（模型）'}。实际发送有效字节 = ${fmt(payload[kind])} B。`,
          stage==='D'?`包含 X 与 Scale。模型 X = rows × ${p.hidden} × ${p.dispatchBytes} B；Scale = rows × (${p.hidden}/32) B。`:`Combine 模型 = rows × ${p.hidden} × ${p.combineBytes} B。`,
          `${ivs.length} 个通信窗口：[首个 PUT 开始，同一核心/peer 的后续 Quiet 结束]。一次 Quiet 可完成多个此前 PUT。`,
          `全部该类窗口时间并集 U = ${fmt(time)} µs；不同核心同时通信只计一次，窗口之间空隙不计。`,
          `单向聚合有效带宽 = ${fmt(payload[kind])} ÷ (${fmt(time)} × 1000) = ${unavailable?'不可计算':fmt(payload[kind]/time/1000)} GB/s。`,
          env>0?`另供参考：该类首尾跨度 ${fmt(env)} µs；含空隙带宽 = ${fmt(payload[kind]/env/1000)} GB/s。`:'没有完整通信窗口。',
          '不乘 2；不把各核心时长求均值。组内只包含其它卡，本卡路径融合在 Pack/GMM2 中，无法从 trace 单独分离。',
          '这是观察到 Quiet 完成的有效带宽，包含软件下发/轮询成本；不是线速。信号字节和理论 512B 附加量不计入实测 payload。',
          ...(unavailable?[`不可计算原因：无窗口=${!ivs.length}；缺字节=${unknown[kind]}；缺 Quiet=${incomplete[kind]}；缺/非法 peer=${missingPeer}。`]:[])
        ],estimated[kind]);
      }
      if(incomplete.intra+incomplete.cross)warnings.push(`${stage==='D'?'Dispatch':'Combine'} 存在未找到后续 Quiet 的 PUT，相关带宽不输出。`);
    }
    bandwidth('D');bandwidth('C');
    let f1=0,f2=0,tiles1=0,tiles2=0,invalidTiles=false;
    for(const e of scopes.filter(e=>e.tid%3===0&&[15,19].includes(e.id))) {
      const {rows:n,columns:c}=e.args;
      if(!finite(n)||n<=0||!finite(c)||c<=0){invalidTiles=true;continue;}
      if(e.id===15){f1+=4*p.hidden*n*c;tiles1++;}
      else {f2+=2*p.intermediate*n*c;tiles2++;}
    }
    const actual=tiles1>0&&tiles2>0&&!invalidTiles;
    const f=actual?f1+f2:6*p.bs*p.topk*p.hidden*p.intermediate;
    const flopDetail=actual?[
      `按本 rank 的真实 AIC tile 计数（可反映路由不均衡）：GMM1 ${tiles1} tiles，GMM2 ${tiles2} tiles。`,
      `GMM1 gate/up 两路：Σ(4 × H × tile.rows × tile.columns) = ${fmt(f1)} FLOP。`,
      `GMM2：Σ(2 × I × tile.rows × tile.columns) = ${fmt(f2)} FLOP。`,
      `F = ${fmt(f1)} + ${fmt(f2)} = ${fmt(f)} FLOP = ${fmt(f/1e9)} GFLOP。`
    ]:[
      'tile 行列信息不全，回退到本 rank 接收负载绝对均衡的估计。BS 指每 rank 原始 token 数。',
      `F = 6 × BS × TopK × H × I = 6 × ${p.bs} × ${p.topk} × ${p.hidden} × ${p.intermediate} = ${fmt(f)} FLOP。`,
      `EP 个来源合计后每 rank 处理 BS×TopK；不再乘 EP 或除 EP，专家数 ${p.experts} 只决定分布。`
    ];
    flopDetail.push('按一次乘加=2 FLOP，只计两次矩阵乘；不计激活、量化、Pack/Reduce，也不计 padding 的无效计算。');
    metrics.flops=metric('有效矩阵计算量',f/1e9,'GFLOP',flopDetail,!actual);
    const tcompute=f/(p.peak*1e6);
    metrics.theoryCompute=metric('理论计算时间',tcompute,'µs',[
      ...flopDetail,`峰值 = ${p.peak} TFLOPS/card；请匹配实际数据类型与稠密计算峰值。`,
      `T_compute = ${fmt(f)} ÷ (${p.peak} × 10⁶) = ${fmt(tcompute)} µs。`
    ],!actual);
    const groupBegin=Math.floor(rank/p.groupSize)*p.groupSize;
    const localSize=Math.min(p.groupSize,p.ep-groupBegin),crossPeers=p.ep-localSize;
    const d=(p.hidden*p.dispatchBytes+p.overhead)*p.topk*p.bs;
    const c=p.hidden*p.combineBytes*p.topk*p.bs;
    const capacity=p.links*p.lineBW*p.efficiency;
    const tcD=d/p.ep*crossPeers/capacity/1000,tcC=c/p.ep*crossPeers/capacity/1000;
    const tcomm=tcD+tcC;
    const tcDetail=[
      '理论网络模型假设每 rank 发向各 rank 均匀；按跨组流量/协议上限计算，未计本组链路瓶颈或软件开销。',
      `D = (H × dispatchBytes + 附加字节) × TopK × BS = (${p.hidden} × ${p.dispatchBytes} + ${p.overhead}) × ${p.topk} × ${p.bs} = ${fmt(d)} B。`,
      `每原始 token ${fmt((p.hidden*p.dispatchBytes+p.overhead)*p.topk)} B = ${fmt((p.hidden*p.dispatchBytes+p.overhead)*p.topk/1024)} KiB。`,
      `跨组比例 = (EP − 本组卡数) / EP = (${p.ep} − ${localSize}) / ${p.ep} = ${fmt(crossPeers/p.ep)}。`,
      `有效上限 = 链路数 × 单向线速 × 协议效率 = ${p.links} × ${p.lineBW} × ${p.efficiency} = ${fmt(capacity)} GB/s。`,
      `T_D = ${fmt(d)} / ${p.ep} × ${crossPeers} / (${fmt(capacity)} × 1000) = ${fmt(tcD)} µs。`,
      `C = H × combineBytes × TopK × BS = ${p.hidden} × ${p.combineBytes} × ${p.topk} × ${p.bs} = ${fmt(c)} B（未额外增加 512B）。`,
      `T_C = ${fmt(c)} / ${p.ep} × ${crossPeers} / (${fmt(capacity)} × 1000) = ${fmt(tcC)} µs。`,
      `通信合计：T_comm = T_D + T_C = ${fmt(tcD)} + ${fmt(tcC)} = ${fmt(tcomm)} µs。`,
      '512B 是理论随路开销假设，与实际 trace 的 Scale 字节（当前 H/32）并不相同。'
    ];
    metrics.theoryComm=metric('理论通信时间',tcomm,'µs',tcDetail,true);
    metrics.referenceCover=metric('参考掩盖率（计算占比）',aic.length&&total>0?aicBounds.span/total*100:null,'%',[
      `参考掩盖率 = T_AIC / T_fused = ${fmt(aicBounds.span)} / ${fmt(total)} × 100% = ${fmt(aicBounds.span/total*100)}%。`,
      '按指定定义，这是带空泡的 AIC 时间覆盖率；高占比不等于 Cube 高利用率，也不等于真实通算重叠率。']);
    const mfu=aicBounds.span>0?f/aicBounds.span/(p.peak*1e6)*100:null;
    metrics.mfu=metric('MFU（带空泡）',mfu,'%',[
      ...flopDetail,`MFU = F / T_AIC / Peak = ${fmt(f)} / (${fmt(aicBounds.span)} × ${p.peak} × 10⁶) × 100% = ${fmt(mfu)}%。`,
      `等价写法：T_compute / T_AIC = ${fmt(tcompute)} / ${fmt(aicBounds.span)} × 100% = ${fmt(mfu)}%。`,
      '分母使用本 rank 全部 AIC 的工作跨度与单卡峰值；不再乘 AIC 核数。'],!actual);
    const denom=Math.min(tcomm,tcompute),overlap=denom>0?(tcomm+tcompute-total)/denom*100:null;
    metrics.theoryOverlap=metric('理论掩盖率',overlap,'%',[
      `T_serial = T_comm + T_compute = ${fmt(tcomm)} + ${fmt(tcompute)} = ${fmt(tcomm+tcompute)} µs。`,
      `分子 = T_serial − T_fused = ${fmt(tcomm+tcompute)} − ${fmt(total)} = ${fmt(tcomm+tcompute-total)} µs。`,
      `分母 = min(${fmt(tcomm)}, ${fmt(tcompute)}) = ${fmt(denom)} µs。`,
      `理论掩盖率 = ${fmt(tcomm+tcompute-total)} / ${fmt(denom)} × 100% = ${fmt(overlap)}%。`,
      `通信口径：Dispatch + Combine；T_comm = ${fmt(tcD)} + ${fmt(tcC)} = ${fmt(tcomm)} µs，属于理想带宽/峰值模型。`,
      '不裁剪到 0–100%。负值表示实际总耗时超过理想串行时间（可能包含 Pack、Reduce、同步、吞吐损失等），不是负的物理重叠；>100% 需检查模型/参数。'
    ],true);
    if(mfu>100)warnings.push('MFU 超过 100%，请核对峰值、维度、时钟或 trace 是否包含重复调用。');
    if(!actual)warnings.push('缺少完整 tile 几何信息；计算量使用绝对均衡模型。');
    if(!finite(meta.envelope_us))warnings.push('缺少原始 envelope_us；总时间只能使用可见事件的首尾跨度。');
    if(meta.hidden_wait_count)warnings.push(`已隐藏 ${meta.hidden_wait_count} 条 Wait；原始总跨度仍保留。`);
    return {rank,metrics,warnings};
  }
  function analyze(doc,params={},filename='trace.json') {
    if(!doc||!Array.isArray(doc.traceEvents)) throw Error('需要 Chrome trace JSON（含 traceEvents），不是 summary.json 或 timer.json。');
    if(doc.metadata?.schema!=null&&doc.metadata.schema!==1)throw Error('仅支持 URMA trace schema 1。');
    const byRank=new Map();
    for(const raw of doc.traceEvents) {
      if(raw.ph!=='X')continue;
      const {pid:rank,tid,ts,dur}=raw;
      if(!Number.isInteger(rank)||rank<0||!Number.isInteger(tid)||tid<0||tid>=84||!finite(ts)||!finite(dur)||ts<0||dur<0)
        throw Error('trace 含非法 rank/引擎/时间；仅支持 URMA 的 block×3+AIC/AIV0/AIV1 布局。');
      const args=raw.args||{},name=String(raw.name||'').split(' [')[0];
      let id=args.event_id;
      if(!Number.isInteger(id))id=NAMES.indexOf(name);
      if(id<0||id>38)continue;
      if(!byRank.has(rank))byRank.set(rank,[]);
      byRank.get(rank).push({rank,tid,ts,dur,end:ts+dur,args,id,name});
    }
    const rankIds=[...byRank.keys()].sort((a,b)=>a-b);
    if(!rankIds.length)throw Error('未发现支持的 URMA 逐事件记录。');
    const p=validateParams(params,doc,rankIds),warnings=[];
    if(doc.metadata?.kind==='urma_allrank' && doc.metadata.rank_count!==rankIds.length)throw Error('allrank 的 rank 数与元数据不符，可能缺失 rank。');
    const ranks=rankIds.map(rank=>{
      const meta=(doc.metadata?.per_rank||[]).find(x=>x.rank===rank)||doc.metadata||{};
      const rows=byRank.get(rank);
      if(Number.isInteger(meta.rank)&&meta.rank!==rank)throw Error(`rank ${rank} 与元数据 rank 不一致。`);
      if(meta.dropped>0)throw Error(`rank ${rank} trace 溢出，不可用于完整指标。`);
      const declared=meta.visible_record_count??meta.record_count;
      if(Number.isInteger(declared)&&declared!==rows.length)throw Error(`rank ${rank} 记录数不匹配：应有 ${declared}，实际 ${rows.length}。`);
      return analyzeRank(rank,rows,meta,p);
    });
    const metrics={};
    for(const key of Object.keys(ranks[0].metrics)) {
      const available=ranks.filter(r=>r.metrics[key].value!==null),m=ranks[0].metrics[key];
      const value=available.length?sum(available.map(r=>r.metrics[key].value))/available.length:null;
      metrics[key]=metric(m.label,value,m.unit,[
        '各 rank 分别计算指标后取算术均值；不会把不同 rank 的独立时钟拼成一条时间轴。',
        ...ranks.map(r=>`rank ${r.rank}：${fmt(r.metrics[key].value)} ${m.unit}`),
        `均值 = (${available.map(r=>fmt(r.metrics[key].value)).join(' + ')||'无有效值'}) / ${available.length} = ${fmt(value)} ${m.unit}。`,
        `有效 ${available.length}/${ranks.length} ranks；缺失项不按 0 处理。`
      ],available.some(r=>r.metrics[key].estimated));
    }
    warnings.push('诊断跨度来自 JSON；各 rank 时钟独立。GB/s 按十进制 10⁹ B/s。');
    warnings.push('“无空泡”指标排除独立 Wait 与条间空隙，但条内阻塞无法从此 trace 分离。');
    warnings.push('组内带宽只计其它卡；当前 self 路径已融合进 Pack/GMM2。');
    return {ranks,mean:{metrics},meta:{warnings,rankCount:ranks.length,ep:p.ep,filename,params:p}};
  }
  function inferParams(doc,filename='') {
    const params={},notes=[],events=(doc?.traceEvents||[]).filter(e=>e.ph==='X');
    const ids=[...new Set(events.map(e=>e.pid).filter(Number.isInteger))];
    if(ids.length>1||doc?.metadata?.kind==='urma_allrank')params.ep=Math.max(ids.length,Math.max(...ids)+1,doc?.metadata?.rank_count||0);
    // Dimension inference needs original tile geometries too, but not clocks.
    const tiles=events.filter(e=>e.tid%3===0).flatMap(e=>Array.isArray(e.args?.source_records)
      ?e.args.source_records.map(r=>({...e.args,...r})): [e.args||{}]);
    for(const [key,id] of [['intermediate',15],['hidden',19]]) {
      const edges=tiles.filter(a=>a.event_id===id&&finite(a.tile_n)&&finite(a.columns))
        .map(a=>a.tile_n+a.columns);
      if(edges.length)params[key]=edges.reduce((a,b)=>Math.max(a,b),0);
    }
    const pack=events.filter(e=>e.pid===ids[0]&&e.args?.event_id===0&&finite(e.args.rows));
    const packed=sum(pack.map(e=>e.args.rows));
    if(packed>0&&Number.isInteger(packed/DEFAULTS.topk)) {
      params.bs=packed/DEFAULTS.topk;notes.push(`BS 由 rank ${ids[0]} 的 Pack.rows 合计 ${packed} / 默认 TopK ${DEFAULTS.topk} 推断；若 TopK 不同请修正。`);
    } else {
      const match=filename.match(/bs[_=\-]?(\d+)/i);
      if(match)params.bs=Number(match[1]);
    }
    for(const [key,id] of [['dispatchBytes',1],['combineBytes',10]]) {
      const first=events.find(e=>e.args?.event_id===id&&e.args.rows>0&&e.args.bytes>0);
      if(first&&params.hidden) {
        const n=first.args.bytes/first.args.rows/params.hidden;
        if([0.5,1,2,4,8].includes(n))params[key]=n;
      }
    }
    if(params.hidden||params.intermediate)notes.push('H/I 由 AIC tile 的 N 方向末端推断；缺失末尾 tile 时请人工核对。');
    return {params,notes};
  }
  const api={DEFAULTS,analyze,inferParams,union};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.URMAAnalyzer=api;
})(typeof window!=='undefined'?window:globalThis);
