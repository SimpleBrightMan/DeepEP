"""EP8/32真实通信七BS；独立进程、配套Python/native ABI，不运行Golden。"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from coordinator import open_control_group
MODEL = dict(hidden=7168,intermediate_hidden=3072,num_experts=384,num_topk=6,num_shared_experts=0)
SEED = 20260922
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method')
    p.add_argument('--bs',type=int,choices=(32,96,256,512,1024,2048,4096),required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--ep-size',type=int,choices=(8,32),default=32)
    p.add_argument('--num-experts',type=int,choices=(96,384),default=384)
    p.add_argument('--package-root',type=Path,default=ROOT)
    p.add_argument('--trace-ranks',help='Separate timer-on profiling pass, e.g. 0-31; default remains timer-off')
    a=p.parse_args()
    if Path(a.label).name!=a.label: raise ValueError('Unsafe label')
    rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    if world!=a.ep_size: raise ValueError(f'EP{a.ep_size} required, got {world}')
    model=dict(MODEL,num_experts=a.num_experts)
    diagnostic=False
    if world not in (8,32) or a.num_experts!=384:
        raise ValueError('Only real EP8/32 E384 targets supported')
    identities=json.loads((a.package_root/'packages.json').read_text())
    if a.method not in identities:
        raise ValueError(f'Unknown method {a.method}; frozen methods: {list(identities)}')
    out=ROOT/'runs'/a.label
    rank_dir=out/f'rank_{rank}'
    rank_dir.mkdir(parents=True,exist_ok=False)
    identity=identities[a.method]
    package=a.package_root/'packages'/a.method
    for name,expected in identity['native'].items():
        if sha(package/name)!=expected: raise RuntimeError(f'{a.method}: native changed {name}')
    sys.path.insert(0,str(package))
    import numpy as np
    import torch
    import torch_npu
    from deep_ep_experimental.megamoe import api
    from deep_ep_experimental.megamoe.benchmark import inputs
    from deep_ep_experimental.megamoe.reference import make_weights,quantize_input
    from deep_ep_experimental.megamoe.performance import collect_profile,summarize_rounds,DEFAULT_ALIGNMENT
    from deep_ep_experimental.megamoe.timer.__main__ import parse_ranks
    trace_ranks=parse_ranks(a.trace_ranks) if a.trace_ranks is not None else []
    if any(r>=world for r in trace_ranks):
        raise ValueError(f'EP{world}: timer rank exceeds world: {trace_ranks}')
    if trace_ranks and not api._extension().timer_enabled:
        raise ValueError('Tracing requires a timer-enabled matching native package')
    if api._extension().api_protocol!=identity['protocol'] or not Path(api.__file__).is_relative_to(package):
        raise RuntimeError('Wrong loaded package')
    def stage(name): print(f'STAGE rank={rank} method={a.method} bs={a.bs} {name}',flush=True)
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(90,repeat=True)
    stage('device_init')
    torch.npu.set_device(local)
    # StoreGroup仍是生产API；只隔离启动器Store，真正Dispatch/Combine仍走SHMEM/MTE。
    stage('host_store_init')
    group=open_control_group(api,rank,world,a.label)
    stage('symm_buffer_init')
    buffer=api.SymmBuffer(group,num_max_tokens_per_rank=a.bs,**model,ranks_per_node=8)
    def digest(value):
        raw=np.ascontiguousarray(value)
        return dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(memoryview(raw)).hexdigest())
    stage('weights')
    raw=make_weights(model['num_experts']//world,model['hidden'],model['intermediate_hidden'],SEED+rank)
    fingerprints={f'w{i}_{kind}':digest(v) for i,pair in enumerate(raw,1) for kind,v in zip(('fp4','sf'),pair)}
    def device_pair(pair):
        data,scale=(torch.from_numpy(v).npu() for v in pair)
        return data,scale.view(torch.float8_e8m0fnu)
    first,second=api.transform_weights_for_mega_moe(*(device_pair(pair) for pair in raw))
    x,ids,weights=inputs(dict(model=model,seed=SEED),rank,a.bs,'balanced',world=world)
    data,scale=quantize_input(x)
    fingerprints.update({n:digest(v) for n,v in zip(('x_fp8','x_sf','route_ids','route_weights'),(data,scale,ids,weights))})
    buffer.x[:a.bs].copy_(torch.from_numpy(data).npu().view(torch.float8_e4m3fn))
    buffer.x_sf[:a.bs].copy_(torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))
    buffer.topk_idx[:a.bs].copy_(torch.from_numpy(ids).npu())
    buffer.topk_weights[:a.bs].copy_(torch.from_numpy(weights).npu())
    buffer.validate_inputs(a.bs)
    buffer.prepare(a.bs)
    stage('inputs_ready')
    # 只诊断计时外准备；进入预热前关闭，避免采样阶段输出栈或新增同步。
    faulthandler.cancel_dump_traceback_later()
    y=torch.empty((a.bs,model['hidden']),device='npu',dtype=torch.bfloat16)
    def call(timer=None):
        if timer is not None and not trace_ranks: raise ValueError('Timer not requested')
        api.fp8_fp4_mega_moe(y,first,second,buffer,timer=timer)
    write(rank_dir/'inputs.json',fingerprints)
    write(rank_dir/'identity.json',dict(method=a.method,bs=a.bs,ep=world,rank=rank,local_rank=local,
        native=identity['native'],model=model,seed=SEED,routing='balanced',timer=False,
        accuracy='not_run_user_requested',torch=torch.__version__,torch_npu=torch_npu.__version__,
        benchmark_sha256=sha(Path(__file__)),coordinator_sha256=sha(ROOT/'coordinator.py'),
        host_coordination='standalone_tcpstore',host_store_backend='legacy',device_communication='shmem_mte',
        diagnostic_equivalent_expert_m=diagnostic))
    del raw,x,data,scale,ids,weights
    prefix=torch.zeros((DEFAULT_ALIGNMENT['rows'],DEFAULT_ALIGNMENT['cols']),device='npu',dtype=torch.float32)
    torch.exp(prefix,out=prefix)
    for _ in range(10): call()
    torch.npu.synchronize()
    group.barrier('warmup_ready')
    stage('profiling')
    samples=collect_profile(call,group,torch,dict(alignment=DEFAULT_ALIGNMENT,iters=20),prefix,rank_dir/'profiler',a.bs)
    stage('profile_complete')
    write(rank_dir/'kernel_samples.json',samples)
    rows=group.all_gather([r['duration_us'] for r in samples],'performance_rounds')
    finite=bool(torch.isfinite(y).all().item())
    write(rank_dir/'output.json',dict(finite=finite,sha256=digest(y.view(torch.int16).cpu().numpy())['sha256']))
    if not finite:
        message=f'EP{world} BS{a.bs} {a.method} rank={rank}: nonfinite'
        group.abort(message)
        raise RuntimeError(message)
    if rank==0:
        report=summarize_rounds(rows,8)
        report.update(ep=world,bs=a.bs,method=a.method,measurement='full_moe',timer=False,model=model,
                      accuracy='not_run_user_requested',native=identity['native'],seed=SEED,routing='balanced',
                      host_coordination='standalone_tcpstore',host_store_backend='legacy',device_communication='shmem_mte',
                      diagnostic_equivalent_expert_m=diagnostic,
                      expected_tokens_per_expert=world*a.bs*model['num_topk']//model['num_experts'])
        write(out/'performance.json',report)
        print('RESULT',a.label,report['mean_us'],report['rank_max_mean_us'],flush=True)
    if trace_ranks:
        stage('timer_allocation')
        timer=None
        try:
            timer=buffer.allocate_timer() if rank in trace_ranks else None
        except Exception as error:
            group.abort(f'timer allocation rank={rank}: {error}')
            raise
        torch.npu.synchronize()
        group.barrier('timer_allocated')
        for _ in range(10): call(timer)
        torch.npu.synchronize()
        group.barrier('timer_warmup_ready')
        stage('timer_profiling')
        # 原collector会逐轮传入None；这里必须显式忽略该参数，确保20轮全部打点。
        timer_samples=collect_profile(lambda _unused=None:call(timer),group,torch,
            dict(alignment=DEFAULT_ALIGNMENT,iters=20),prefix,rank_dir/'timer_profiler',a.bs)
        write(rank_dir/'timer_kernel_samples.json',timer_samples)
        timer_rows=group.all_gather([r['duration_us'] for r in timer_samples],'timer_performance_rounds')
        if rank==0:
            timer_report=summarize_rounds(timer_rows,8)
            timer_report.update(ep=world,bs=a.bs,method=a.method,measurement='full_moe',
                timer=True,timer_ranks=trace_ranks,all_measured_invocations_instrumented=True,
                native=identity['native'],trace_policy=identity.get('trace_policy'))
            write(out/'timer_on_performance.json',timer_report)
            print('TIMER_RESULT',a.label,timer_report['mean_us'],flush=True)
        on_output=dict(finite=bool(torch.isfinite(y).all().item()),
                       sha256=digest(y.view(torch.int16).cpu().numpy())['sha256'])
        off_output=json.loads((rank_dir/'output.json').read_text())
        write(rank_dir/'timer_output.json',on_output)
        if on_output!=off_output:
            group.abort(f'rank={rank}: timer output differs from timer-off')
            raise RuntimeError('Timer on/off output mismatch (not a Golden test)')
        if timer is not None:
            from export_native_trace import export
            raw_timer=timer.cpu().numpy().astype('<i8',copy=False).reshape(-1)
            exported=export(raw_timer,rank_dir/'timer',rank)
            write(rank_dir/'trace_identity.json',dict(**exported,native=identity['native'],
                trace_policy=identity.get('trace_policy'),trace_only_changes=identity.get('trace_only_changes'),
                exporter_sha256=sha(ROOT/'export_native_trace.py')))
            stage('timer_saved')
            del raw_timer
        del timer
        group.barrier('timer_export_ready')
    group.barrier('output_ready')
    buffer.destroy()
    group.close()

if __name__=='__main__': main()
