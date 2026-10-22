'use strict';

// CPU-only, dependency-free checks: node --test test_urma_dashboard_metrics.cjs.
const assert = require('node:assert/strict');
const test = require('node:test');
const { DEFAULTS, analyze, inferParams, union } = require('../tools/urma_dashboard_metrics.js');

const PARAMETERS = { bs: 8, hidden: 32, intermediate: 16, experts: 32, topk: 2, peak: 1 };
const NAMES = {
  0: 'Pack', 1: 'DispatchPutX', 2: 'DispatchPutScale', 3: 'DispatchQuiet',
  4: 'DispatchPublish', 5: 'SelfDispatch', 10: 'CombinePut', 11: 'CombineQuiet',
  12: 'SelfCombine', 15: 'Gmm1Tile', 16: 'Gmm1TileDrain', 17: 'Gmm1Dequant',
  19: 'Gmm2Tile', 20: 'Gmm2Dequant', 21: 'Gmm2Drain', 25: 'Reduce',
  27: 'Gmm1Drain', 30: 'CombinePublish', 33: 'Gmm1Tile', 34: 'Gmm2Tile',
  35: 'Gmm1Batch', 36: 'Gmm2Batch', 37: 'Gmm1DequantBatch', 38: 'Gmm2DequantBatch'
};

function close(actual, expected, message = '') {
  assert.ok(Number.isFinite(actual), `${message}: expected finite value, got ${actual}`);
  assert.ok(Math.abs(actual - expected) <= Math.max(1e-12, Math.abs(expected) * 1e-10),
    `${message}: ${actual} != ${expected}`);
}

function event(rank, tid, id, ts, dur, args = {}) {
  return {
    ph: 'X', pid: rank, tid, name: NAMES[id] || `event_${id}`, ts, dur,
    args: { event_id: id, bytes: 0, ...args }
  };
}

function seal(doc) {
  doc.metadata.record_count = doc.traceEvents.filter(e => e.ph === 'X').length;
  doc.metadata.visible_record_count = doc.metadata.record_count;
  return doc;
}

function rankTrace(rank = 0, timeScale = 1) {
  const group = Math.floor(rank / 4) * 4;
  const near1 = group + (rank % 4 + 1) % 4;
  const near2 = group + (rank % 4 + 2) % 4;
  const far = (rank + 4) % 16;
  const out = [];
  const add = (tid, id, ts, dur, args = {}) => {
    out.push(event(rank, tid, id, ts * timeScale, dur * timeScale, args));
  };
  add(1, 0, 0, 5, { rows: 16, bytes: 528 });
  for (const [tid, peer, start, quiet, quietDur] of [
    [2, near1, 10, 15, 5], [5, near2, 14, 18, 6], [8, far, 10, 14, 4]
  ]) {
    const meta = { peer, cv: 0, rows: 4 };
    add(tid, 1, start, 2, { ...meta, bytes: 128 });
    add(tid, 2, start + 2, 1, { ...meta, bytes: 4 });
    add(tid, 3, quiet, quietDur, meta);
  }
  for (const [tid, rows, start, submit, drain, teardown] of [
    [0, 2, 25, 5, 2, 1], [3, 1, 26, 6, 1, 1]
  ]) {
    const meta = { fragment: tid / 3, expert: 0, row: 0, rows, tile_m: 0, tile_n: 0, columns: 16 };
    add(tid, 15, start, submit, meta);
    add(tid, 16, start + submit + 1, drain, meta);
    add(tid, 27, start + submit + drain + 2, teardown, meta);
  }
  for (const [tid, rows, start, submit, drainStart, drainDur] of [
    [0, 2, 40, 4, 45, 3], [3, 1, 42, 5, 49, 2]
  ]) {
    const meta = { fragment: tid / 3, expert: 0, row: 0, rows, tile_m: 0, tile_n: 0, columns: 32 };
    add(tid, 19, start, submit, meta);
    add(tid, 21, drainStart, drainDur, meta);
  }
  // Long paired AIV0 dequant must not become AIC time or duplicate tile FLOPs.
  add(1, 17, 25, 30, { rows: 999, columns: 999 });
  add(4, 20, 40, 30, { rows: 999, columns: 999 });
  for (const [tid, peer, start, quiet, quietDur] of [
    [2, near1, 60, 62, 8], [5, near2, 66, 68, 6], [8, far, 60, 64, 8]
  ]) {
    const meta = { peer, cv: 0, rows: 4 };
    add(tid, 10, start, 1, { ...meta, bytes: 256 });
    add(tid, 11, quiet, quietDur, meta);
  }
  add(1, 25, 75, 10, { rows: 8 });
  return seal({ traceEvents: out, metadata: {
    schema: 1, rank, dropped: 0, clock_mhz: 1000,
    envelope_us: 100 * timeScale, hidden_wait_count: 7
  } });
}

function metrics(doc = rankTrace(), overrides = {}) {
  return analyze(doc, { ...PARAMETERS, ...overrides }).ranks[0].metrics;
}

function coalesce(doc) {
  const result = structuredClone(doc), remove = new Set();
  for (const tile of result.traceEvents.filter(e => [15, 19].includes(e.args.event_id))) {
    const ids = tile.args.event_id === 15 ? [15, 16, 27] : [19, 21];
    const parts = ids.map(id => result.traceEvents.find(e => e.tid === tile.tid &&
      e.args.fragment === tile.args.fragment && e.args.event_id === id));
    tile.args.source_records = parts.map(part => ({
      event_id: part.args.event_id,
      start_tick: String(Math.round(part.ts * 1000)),
      end_tick: String(Math.round((part.ts + part.dur) * 1000))
    }));
    tile.args.start_tick = tile.args.source_records[0].start_tick;
    tile.args.end_tick = tile.args.source_records.at(-1).end_tick;
    tile.dur = parts.at(-1).ts + parts.at(-1).dur - tile.ts;
    tile.args.event_id = ids[0] === 15 ? 33 : 34;
    tile.name += ' [compute + drain]';
    parts.slice(1).forEach(part => remove.add(part));
  }
  result.traceEvents = result.traceEvents.filter(e => !remove.has(e));
  return seal(result);
}

function multiFragmentTrace() {
  const doc = rankTrace();
  doc.traceEvents = doc.traceEvents.filter(e => ![15,16,17,19,20,21,27].includes(e.args.event_id));
  const shapes = {0:{fragment:0,expert:0,rows:2,columns:16,tile_n:0},
    1:{fragment:1,expert:0,rows:1,columns:4,tile_n:16}};
  for(const [id,ts,dur,fragment] of [
    [15,10,2,0],[16,13,1,0],[29,15,4,1],[15,21,3,1],[16,25,1,1],[27,28,2,1],
    [19,35,2,0],[18,38,5,1],[19,45,3,1],[21,49,2,1]
  ]) {
    const args={...shapes[fragment]};
    if([19,21].includes(id)){args.columns*=2;args.tile_n*=2;}
    doc.traceEvents.push(event(0,0,id,ts,dur,args));
  }
  for(const [id,ts,dur,fragment] of [[17,10,25,0],[17,36,5,1],[27,43,2,1],
    [20,50,5,0],[20,60,15,1],[21,79,2,1]])
    doc.traceEvents.push(event(0,1,id,ts,dur,{...shapes[fragment],rows:999,columns:999}));
  return seal(doc);
}

function coalesceBatches(input) {
  const doc=structuredClone(input);
  const specs=[[0,35,[15,16,29,27]],[0,36,[19,18,21]],[1,37,[17,27]],[1,38,[20,21]]];
  for(const [tid,id,sourceIds] of specs) {
    const parts=doc.traceEvents.filter(e=>e.tid===tid&&sourceIds.includes(e.args.event_id)).sort((a,b)=>a.ts-b.ts);
    const first=parts[0],last=parts.at(-1);
    const source_records=parts.map((e,index)=>({...e.args,record_index:index,
      start_tick:String(Math.round(e.ts*1000)),end_tick:String(Math.round((e.ts+e.dur)*1000))}));
    const batch=event(0,tid,id,first.ts,last.ts+last.dur-first.ts,{expert:0,
      fragment:null,rows:null,columns:null,tile_n:null,source_records,
      start_tick:source_records[0].start_tick,end_tick:source_records.at(-1).end_tick});
    doc.traceEvents=doc.traceEvents.filter(e=>!parts.includes(e));
    doc.traceEvents.push(batch);
  }
  return seal(doc);
}

test('multi-fragment GMM and Dequant batches preserve metrics without counting internal waits or gaps', () => {
  const doc=multiFragmentTrace(),raw=metrics(doc),merged=metrics(coalesceBatches(doc));
  for(const key of Object.keys(raw)) {
    if(raw[key].value===null)assert.equal(merged[key].value,null,key);
    else close(merged[key].value,raw[key].value,key);
  }
  close(merged.g1Busy.value,9); // 2+1+3+1+2; excludes 4us Wait and 7us gaps.
  close(merged.g2Busy.value,7); // 2+3+2; excludes 5us Wait and 4us gaps.
  close(merged.aicSpan.value,41); // 10..51, unaffected by AIV0 reaching 81.
  close(merged.flops.value,(4*32*(2*16+1*4)+2*16*(2*32+1*8))/1e9);
  assert.equal(merged.flops.estimated,false);
});

test('batch shape inference uses individual raw tile geometry with null outer fields', () => {
  const doc=multiFragmentTrace();
  assert.deepEqual(inferParams(coalesceBatches(doc)),inferParams(doc));
  assert.equal(inferParams(coalesceBatches(doc)).params.hidden,40);
  assert.equal(inferParams(coalesceBatches(doc)).params.intermediate,20);
});

test('new batch cannot silently become busy time when provenance is missing or invalid', () => {
  for(const mutate of [
    batch=>delete batch.args.source_records,
    batch=>{batch.args.source_records=[];},
    batch=>{batch.args.source_records[0].event_id=17;}
  ]) {
    const doc=coalesceBatches(multiFragmentTrace());
    mutate(doc.traceEvents.find(e=>e.args.event_id===35));
    assert.throws(()=>analyze(doc,PARAMETERS),/GMM|source_records/);
  }
});

test('interval union counts concurrency once and preserves real gaps', () => {
  assert.equal(union([[10, 20], [14, 24]]), 14);
  assert.equal(union([[20, 30], [0, 10], [5, 15]]), 25);
  assert.equal(union([[0, 1], [1, 2], [0.5, 1.5]]), 2);
  assert.equal(union([]), 0);
});

test('AIC range sums both GMM stages per core and ignores waits and AIV work', () => {
  const doc = seal({ metadata: { schema: 1, rank: 0 }, traceEvents: [
    event(0, 0, 15, 0, 10), event(0, 0, 19, 100, 2),
    event(0, 3, 15, 0, 3), event(0, 3, 19, 100, 12),
    event(0, 0, 18, 10, 90), event(0, 1, 17, 0, 1000),
    event(0, 6, 18, 0, 100)
  ] });
  const m = metrics(doc);
  close(m.aicBusyRange.value, 3); // core 0 = 12, core 1 = 15; missing GMM cores are not zero.
  assert.ok(m.aicBusyRange.detail.some(s => s.includes('15 − 12 = 3')));
  doc.traceEvents.push(event(0, 0, 15, 2, 3)); // Duplicate/overlapping scope must not inflate work.
  close(metrics(seal(doc)).aicBusyRange.value, 3);
  doc.traceEvents = doc.traceEvents.filter(e => e.tid === 0);
  close(metrics(seal(doc)).aicBusyRange.value, 0);
  doc.traceEvents = doc.traceEvents.filter(e => e.args.event_id === 18);
  assert.equal(metrics(seal(doc)).aicBusyRange.value, null);
});

test('allrank reports mean of local AIC ranges instead of a pooled range', () => {
  const traceEvents = [event(0, 0, 15, 0, 10), event(0, 3, 15, 0, 14),
    event(1, 0, 15, 0, 100), event(1, 3, 15, 0, 108)];
  const result = analyze({ traceEvents, metadata: { schema: 1, kind: 'urma_allrank', rank_count: 2 } },
    { ...PARAMETERS, groupSize: 1 });
  close(result.ranks[0].metrics.aicBusyRange.value, 4);
  close(result.ranks[1].metrics.aicBusyRange.value, 8);
  close(result.mean.metrics.aicBusyRange.value, 6);
});

test('bandwidth uses single-direction bytes divided by cohort union, not average core time', () => {
  const m = metrics();
  close(m.dIntra.value, 264 / 14 / 1000);
  close(m.dCross.value, 132 / 8 / 1000);
  close(m.cIntra.value, 512 / 14 / 1000);
  close(m.cCross.value, 256 / 12 / 1000);
  assert.notEqual(m.dIntra.value, 264 / 10 / 1000);
  assert.ok(m.dIntra.detail.some(s => s.includes('14 µs')));
});

test('separate CVs have separate completion windows and intervening producer idle is excluded', () => {
  const doc = rankTrace();
  const peer = doc.traceEvents.find(e => e.tid === 8 && e.args.event_id === 1).args.peer;
  doc.traceEvents.push(
    event(0, 8, 1, 30, 2, { peer, cv: 1, rows: 4, bytes: 128 }),
    event(0, 8, 2, 32, 1, { peer, cv: 1, rows: 4, bytes: 4 }),
    event(0, 8, 3, 35, 3, { peer, cv: 1, rows: 4 })
  );
  const m = metrics(seal(doc));
  close(m.dCross.value, 264 / 16 / 1000);
  assert.notEqual(m.dCross.value, 264 / 28 / 1000);
});

test('PUT without subsequent Quiet cannot produce a completed bandwidth', () => {
  const doc = rankTrace();
  doc.traceEvents = doc.traceEvents.filter(e => !(e.tid === 8 && e.args.event_id === 3));
  const result = analyze(seal(doc), PARAMETERS);
  assert.equal(result.ranks[0].metrics.dCross.value, null);
  assert.ok(result.ranks[0].warnings.some(s => s.includes('Quiet')));
  assert.notEqual(result.ranks[0].metrics.dIntra.value, null);
});

test('explicit mismatched Quiet CV does not complete the previous CV', () => {
  const doc = rankTrace();
  doc.traceEvents.find(e => e.tid === 8 && e.args.event_id === 3).args.cv = 99;
  assert.equal(metrics(doc).dCross.value, null);
});

test('historical peer-level Quiet without CV may complete multiple outstanding PUTs', () => {
  const doc = rankTrace();
  const quiet = doc.traceEvents.find(e => e.tid === 8 && e.args.event_id === 3);
  delete quiet.args.cv;
  doc.traceEvents.push(event(0, 8, 1, 13, 1, { peer: quiet.args.peer, cv: 1, rows: 2, bytes: 64 }));
  close(metrics(seal(doc)).dCross.value, 196 / 8 / 1000);
});

test('self traffic does not inflate same-group network payload or timing', () => {
  const doc = rankTrace();
  doc.traceEvents.push(
    event(0, 11, 1, 0, 10, { peer: 0, cv: 0, rows: 999, bytes: 99999 }),
    event(0, 11, 3, 10, 80, { peer: 0, cv: 0 }),
    event(0, 11, 10, 0, 10, { peer: 0, cv: 0, rows: 999, bytes: 99999 }),
    event(0, 11, 11, 10, 80, { peer: 0, cv: 0 }),
    event(0, 14, 5, 0, 90, { rows: 999, bytes: 99999 }),
    event(0, 14, 12, 0, 90, { rows: 999, bytes: 99999 })
  );
  const m = metrics(seal(doc));
  close(m.dIntra.value, 264 / 14 / 1000);
  close(m.cIntra.value, 512 / 14 / 1000);
});

test('trace byte mode preserves actual bytes; model mode responds to datatype overrides', () => {
  const actual = metrics(rankTrace(), { dispatchBytes: 2, combineBytes: 4 });
  close(actual.dIntra.value, 264 / 14 / 1000);
  close(actual.cIntra.value, 512 / 14 / 1000);
  assert.equal(actual.dIntra.estimated, false);
  const model = metrics(rankTrace(), { byteMode: 'model', dispatchBytes: 2, combineBytes: 4 });
  close(model.dIntra.value, 520 / 14 / 1000);
  close(model.cIntra.value, 1024 / 14 / 1000);
  assert.equal(model.dIntra.estimated, true);
});

test('missing PUT bytes use explicit row-size estimate and are marked estimated', () => {
  const doc = rankTrace();
  for (const e of doc.traceEvents.filter(e => [1, 2, 10].includes(e.args.event_id))) delete e.args.bytes;
  const m = metrics(doc);
  close(m.dCross.value, 132 / 8 / 1000);
  close(m.cCross.value, 256 / 12 / 1000);
  assert.equal(m.dCross.estimated, true);
  assert.equal(m.cCross.estimated, true);
});

test('busy time is longest sum of raw scopes per core, distinct from transfer envelope', () => {
  const m = metrics();
  close(m.dBusy.value, 9); // 2 + 1 + 6, on tid 5.
  close(m.cBusy.value, 9); // 1 + 8, on tid 2 or 8.
  close(m.g1Busy.value, 8);
  close(m.g2Busy.value, 7);
});

test('merged and unmerged GMM records give identical busy time, FLOPs and AIC span', () => {
  const raw = metrics(), merged = metrics(coalesce(rankTrace()));
  for (const key of ['g1Busy', 'g2Busy', 'aicBusyRange', 'flops', 'aicSpan', 'mfu'])
    close(merged[key].value, raw[key].value, key);
  close(merged.g1Busy.value, 8); // The 2 us gaps inside the merged bar are excluded.
});

test('actual rank tile work counts gate/up twice and excludes AIV dequant duplicates', () => {
  const m = metrics();
  const expectedF1 = 4 * 32 * (2 * 16 + 1 * 16);
  const expectedF2 = 2 * 16 * (2 * 32 + 1 * 32);
  close(m.flops.value, (expectedF1 + expectedF2) / 1e9);
  assert.equal(m.flops.estimated, false);
  close(m.theoryCompute.value, (expectedF1 + expectedF2) / 1e6);
  close(m.aicSpan.value, 51 - 25);
  close(m.mfu.value, (expectedF1 + expectedF2) / 26 / 1e6 * 100);
  close(m.mfu.value, m.theoryCompute.value / m.aicSpan.value * 100);
});

test('missing tile geometry falls back to per-rank balanced model without EP or expert multipliers', () => {
  const doc = rankTrace();
  delete doc.traceEvents.find(e => e.args.event_id === 15).args.columns;
  const m = metrics(doc);
  close(m.flops.value, 6 * 8 * 2 * 32 * 16 / 1e9);
  assert.equal(m.flops.estimated, true);
});

test('unfiltered diagnostic envelope controls fused time despite hidden Wait records', () => {
  const m = metrics();
  close(m.fused.value, 100);
  close(m.referenceCover.value, 26);
  const doc = rankTrace();
  delete doc.metadata.envelope_us;
  close(metrics(doc).fused.value, 85);
});

test('theoretical communication always includes dispatch and combine, including in overlap', () => {
  const m = metrics();
  const td = (32 + 512) * 2 * 8 / 16 * 12 / (4 * 56 * 0.8) / 1000;
  const tc = 32 * 2 * 2 * 8 / 16 * 12 / (4 * 56 * 0.8) / 1000;
  const compute = 9216 / 1e6;
  close(m.theoryComm.value, td + tc);
  close(m.theoryOverlap.value, (td + tc + compute - 100) / Math.min(td + tc, compute) * 100);
  assert.ok(m.theoryOverlap.value < 0);
  const wider = metrics(rankTrace(), { combineBytes: 4 });
  close(wider.theoryComm.value, td + 2 * tc);
  assert.ok(wider.theoryOverlap.value > m.theoryOverlap.value);
  close(metrics(rankTrace(), { theoryMode: 'dispatch' }).theoryComm.value, td + tc);
  assert.match(m.theoryComm.detail.join('\n'), /T_comm = T_D \+ T_C/);
});

test('production default EP16 BS4096 model includes 2-byte combine traffic', () => {
  const m = analyze(rankTrace()).ranks[0].metrics;
  // 100,663,296 B dispatch + 176,160,768 B combine, at 179.2 GB/s.
  close(m.theoryComm.value, 1544.777142857143);
});

test('allrank uses each rank local clock and arithmetic mean of per-rank metrics', () => {
  const inputs = Array.from({ length: 16 }, (_, rank) => rankTrace(rank, rank === 0 ? 1 : 2));
  const doc = {
    traceEvents: inputs.flatMap(item => item.traceEvents),
    metadata: { schema: 1, kind: 'urma_allrank', rank_count: 16, ranks: inputs.map((_, i) => i),
      per_rank: inputs.map(item => item.metadata) }
  };
  const result = analyze(doc, { ...PARAMETERS, ep: 4 });
  assert.equal(result.ranks.length, 16);
  assert.equal(result.meta.ep, 16);
  close(result.mean.metrics.fused.value, (100 + 15 * 200) / 16);
  close(result.mean.metrics.dIntra.value, (264 / 14 / 1000 + 15 * 264 / 28 / 1000) / 16);
  close(result.mean.metrics.aicSpan.value, (26 + 15 * 52) / 16);
  close(result.mean.metrics.referenceCover.value, 26);
  for (const rank of result.ranks) assert.notEqual(rank.metrics.dIntra.value, null);
});

test('topology changes correctly reclassify remote peers without inventing self bytes', () => {
  const m = metrics(rankTrace(), { groupSize: 8 });
  close(m.dIntra.value, 396 / 14 / 1000);
  assert.equal(m.dCross.value, null);
});

test('inference extracts tile dimensions, actual datatypes and Pack-based BS assumption', () => {
  const inferred = inferParams(rankTrace());
  assert.equal(inferred.params.hidden, 32);
  assert.equal(inferred.params.intermediate, 16);
  assert.equal(inferred.params.dispatchBytes, 1);
  assert.equal(inferred.params.combineBytes, 2);
  assert.equal(inferred.params.bs, 16 / DEFAULTS.topk);
  assert.ok(inferred.notes.some(note => note.includes('TopK')));
});

test('inference accepts merged GMM IDs and filename fallback when Pack is absent', () => {
  const doc = coalesce(rankTrace());
  doc.traceEvents = doc.traceEvents.filter(e => e.args.event_id !== 0);
  const inferred = inferParams(doc, 'rank_3_bs_8192.json');
  assert.equal(inferred.params.hidden, 32);
  assert.equal(inferred.params.intermediate, 16);
  assert.equal(inferred.params.bs, 8192);
});

test('invalid time, identity, schema, incomplete records and invalid parameters are rejected', () => {
  assert.throws(() => analyze({}), /traceEvents|Chrome/);
  assert.throws(() => analyze({ traceEvents: [] }), /记录/);
  for (const [field, value] of [['dur', -1], ['ts', NaN], ['tid', 84], ['pid', -1]]) {
    const doc = rankTrace();
    doc.traceEvents[0][field] = value;
    assert.throws(() => analyze(doc, PARAMETERS), /非法/);
  }
  const dropped = rankTrace(); dropped.metadata.dropped = 1;
  assert.throws(() => analyze(dropped, PARAMETERS), /溢出/);
  const missing = rankTrace(); missing.traceEvents.pop();
  assert.throws(() => analyze(missing, PARAMETERS), /记录数/);
  const schema = rankTrace(); schema.metadata.schema = 99;
  assert.throws(() => analyze(schema, PARAMETERS), /schema/);
  const bytes = rankTrace(); bytes.traceEvents.find(e => e.args.event_id === 1).args.bytes = -1;
  assert.throws(() => analyze(bytes, PARAMETERS), /bytes|字节/);
  assert.throws(() => analyze(rankTrace(), { ...PARAMETERS, bs: 0 }), /bs/);
  assert.throws(() => analyze(rankTrace(), { ...PARAMETERS, efficiency: 1.1 }), /效率/);
  assert.throws(() => analyze(rankTrace(), { ...PARAMETERS, ep: 2, groupSize: 4 }), /卡数/);
});

test('incomplete allrank metadata cannot silently become a partial comparison', () => {
  const doc = rankTrace();
  doc.metadata.kind = 'urma_allrank'; doc.metadata.rank_count = 16;
  assert.throws(() => analyze(doc, PARAMETERS), /rank/);
});

test('malformed composite source records are rejected instead of falsifying GMM work', () => {
  for (const mutate of [
    part => { part.end_tick = String(Number(part.start_tick) - 1); },
    part => { part.event_id = 10; },
    part => { part.start_tick = '0'; }
  ]) {
    const doc = coalesce(rankTrace());
    const combined = doc.traceEvents.find(e => e.args.event_id === 33);
    mutate(combined.args.source_records[0]);
    assert.throws(() => analyze(doc, PARAMETERS), /GMM|源记录/);
  }
  const doc = coalesce(rankTrace()); delete doc.metadata.clock_mhz;
  assert.throws(() => analyze(doc, PARAMETERS), /clock_mhz/);
});
