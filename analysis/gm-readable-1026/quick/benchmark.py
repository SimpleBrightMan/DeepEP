"""EP8/32真实通信；同组worker连续测多个BS，整套进程独立复测，不运行Golden。"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from coordinator import open_control_group
MODEL = dict(hidden=7168,intermediate_hidden=3072,num_experts=384,num_topk=6,num_shared_experts=0)
SEED = 20260922
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
SHAPES = (32,96,256,512,1024,2048,4096)


def resolve_cases(bs, values, capacity):
    batches=[bs] if bs is not None else list(values or [])
    if not batches or len(set(batches))!=len(batches) or any(v not in SHAPES for v in batches):
        raise ValueError('Expected distinct supported BS values')
    capacity=max(batches) if capacity is None else capacity
    if not max(batches)<=capacity<=4096:
        raise ValueError('Buffer capacity must cover every BS and be <=4096')
    return batches,capacity


def case_label(bs, label):
    if re.fullmatch(r'(control|combined)-[ab]',label):return f'bs{bs}-{label}'
    if re.fullmatch(fr'bs{bs}-(control|combined)-[ab]',label):return label
    raise ValueError(f'Invalid case label for BS{bs}: {label}')

def write(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method')
    cases=p.add_mutually_exclusive_group(required=True)
    cases.add_argument('--bs',type=int,choices=SHAPES,help='Legacy isolated case')
    cases.add_argument('--bs-values',type=int,nargs='+',help='All cases in one worker session')
    p.add_argument('--buffer-capacity',type=int,help='Default max(BS); recorded separately from actual BS')
    p.add_argument('--label',required=True)
    p.add_argument('--ep-size',type=int,choices=(8,32),default=32)
    p.add_argument('--num-experts',type=int,choices=(96,384),default=384)
    p.add_argument('--package-root',type=Path,default=ROOT)
    p.add_argument('--trace-ranks',help='Separate timer-on profiling pass, e.g. 0-31; default remains timer-off')
    a=p.parse_args()
    batches,capacity=resolve_cases(a.bs,a.bs_values,a.buffer_capacity)
    for bs in batches:case_label(bs,a.label)
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
    def stage(name,bs=0): print(f'STAGE rank={rank} method={a.method} bs={bs} {name}',flush=True)
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(90,repeat=True)
    stage('device_init')
    torch.npu.set_device(local)
    # 测试主机控制使用CPU Gloo，真正Dispatch/Combine仍走SHMEM/MTE，不引入HCCL。
    stage('host_store_init')
    group=open_control_group(api,rank,world,a.label)
    stage('symm_buffer_init')
    buffer=api.SymmBuffer(group,num_max_tokens_per_rank=capacity,**model,ranks_per_node=8)
    def digest(value):
        raw=np.ascontiguousarray(value)
        return dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(memoryview(raw)).hexdigest())
    stage('weights')
    raw=make_weights(model['num_experts']//world,model['hidden'],model['intermediate_hidden'],SEED+rank)
    weight_fingerprints={f'w{i}_{kind}':digest(v) for i,pair in enumerate(raw,1) for kind,v in zip(('fp4','sf'),pair)}
    def device_pair(pair):
        data,scale=(torch.from_numpy(v).npu() for v in pair)
        return data,scale.view(torch.float8_e8m0fnu)
    first,second=api.transform_weights_for_mega_moe(*(device_pair(pair) for pair in raw))
    del raw
    prefix=torch.zeros((DEFAULT_ALIGNMENT['rows'],DEFAULT_ALIGNMENT['cols']),device='npu',dtype=torch.float32)
    torch.exp(prefix,out=prefix)
    def run_case(bs):
        # 权重/SHMEM归整套session所有；输入、输出、计划和证据按实际BS隔离。
        label=case_label(bs,a.label)
        out=ROOT/'runs'/label
        rank_dir=out/f'rank_{rank}'
        rank_dir.mkdir(parents=True,exist_ok=False)
        faulthandler.dump_traceback_later(90,repeat=True)
        x,ids,weights=inputs(dict(model=model,seed=SEED),rank,bs,'balanced',world=world)
        data,scale=quantize_input(x)
        fingerprints=dict(weight_fingerprints)
        fingerprints.update({n:digest(v) for n,v in zip(('x_fp8','x_sf','route_ids','route_weights'),(data,scale,ids,weights))})
        buffer.x[:bs].copy_(torch.from_numpy(data).npu().view(torch.float8_e4m3fn))
        buffer.x_sf[:bs].copy_(torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))
        buffer.topk_idx[:bs].copy_(torch.from_numpy(ids).npu())
        buffer.topk_weights[:bs].copy_(torch.from_numpy(weights).npu())
        buffer.validate_inputs(bs)
        buffer.prepare(bs)
        stage('inputs_ready',bs)
        faulthandler.cancel_dump_traceback_later()
        y=torch.empty((bs,model['hidden']),device='npu',dtype=torch.bfloat16)
        def call(timer=None):
            if timer is not None and not trace_ranks:raise ValueError('Timer not requested')
            api.fp8_fp4_mega_moe(y,first,second,buffer,timer=timer)
        write(rank_dir/'inputs.json',fingerprints)
        resources=dict(buffer_capacity=capacity,workspace_bytes=buffer.workspace_bytes,
            symmetric_buffer_bytes=buffer.symmetric_buffer_bytes)
        write(rank_dir/'identity.json',dict(method=a.method,bs=bs,ep=world,rank=rank,local_rank=local,
            native=identity['native'],model=model,seed=SEED,routing='balanced',timer=False,
            accuracy='not_run_user_requested',torch=torch.__version__,torch_npu=torch_npu.__version__,
            benchmark_sha256=sha(Path(__file__)),coordinator_sha256=sha(ROOT/'coordinator.py'),
            host_coordination='cpu_gloo',device_communication='shmem_mte',
            diagnostic_equivalent_expert_m=diagnostic,session_label=a.label,session_bs_values=batches,**resources))
        del x,data,scale,ids,weights
        for _ in range(10):call()
        torch.npu.synchronize()
        group.barrier(f'BS{bs}/warmup_ready')
        stage('profiling',bs)
        samples=collect_profile(call,group,torch,dict(alignment=DEFAULT_ALIGNMENT,iters=20),prefix,rank_dir/'profiler',bs)
        stage('profile_complete',bs)
        write(rank_dir/'kernel_samples.json',samples)
        rows=group.all_gather([r['duration_us'] for r in samples],f'BS{bs}/performance_rounds')
        finite=bool(torch.isfinite(y).all().item())
        write(rank_dir/'output.json',dict(finite=finite,sha256=digest(y.view(torch.int16).cpu().numpy())['sha256']))
        if not finite:raise RuntimeError(f'EP{world} BS{bs} {a.method} rank={rank}: nonfinite')
        if rank==0:
            report=summarize_rounds(rows,8)
            report.update(ep=world,bs=bs,method=a.method,measurement='full_moe',timer=False,model=model,
                accuracy='not_run_user_requested',native=identity['native'],seed=SEED,routing='balanced',
                host_coordination='cpu_gloo',device_communication='shmem_mte',
                diagnostic_equivalent_expert_m=diagnostic,session_label=a.label,session_bs_values=batches,
                expected_tokens_per_expert=world*bs*model['num_topk']//model['num_experts'],**resources)
            write(out/'performance.json',report)
            print('RESULT',label,report['mean_us'],report['rank_max_mean_us'],flush=True)
        if trace_ranks:
            stage('timer_allocation',bs)
            timer=buffer.allocate_timer() if rank in trace_ranks else None
            torch.npu.synchronize()
            group.barrier(f'BS{bs}/timer_allocated')
            for _ in range(10):call(timer)
            torch.npu.synchronize()
            group.barrier(f'BS{bs}/timer_warmup_ready')
            stage('timer_profiling',bs)
            # collector会传None；忽略该参数，保证每次测量均打点。
            timer_samples=collect_profile(lambda _unused=None:call(timer),group,torch,
                dict(alignment=DEFAULT_ALIGNMENT,iters=20),prefix,rank_dir/'timer_profiler',bs)
            write(rank_dir/'timer_kernel_samples.json',timer_samples)
            timer_rows=group.all_gather([r['duration_us'] for r in timer_samples],f'BS{bs}/timer_performance_rounds')
            if rank==0:
                timer_report=summarize_rounds(timer_rows,8)
                timer_report.update(ep=world,bs=bs,method=a.method,measurement='full_moe',
                    timer=True,timer_ranks=trace_ranks,all_measured_invocations_instrumented=True,
                    native=identity['native'],trace_policy=identity.get('trace_policy'),**resources)
                write(out/'timer_on_performance.json',timer_report)
                print('TIMER_RESULT',label,timer_report['mean_us'],flush=True)
            on_output=dict(finite=bool(torch.isfinite(y).all().item()),
                sha256=digest(y.view(torch.int16).cpu().numpy())['sha256'])
            off_output=json.loads((rank_dir/'output.json').read_text())
            write(rank_dir/'timer_output.json',on_output)
            if on_output!=off_output:raise RuntimeError(f'BS{bs} rank={rank}: timer output mismatch (not Golden)')
            if timer is not None:
                from export_native_trace import export
                raw_timer=timer.cpu().numpy().astype('<i8',copy=False).reshape(-1)
                exported=export(raw_timer,rank_dir/'timer',rank)
                write(rank_dir/'trace_identity.json',dict(**exported,native=identity['native'],
                    trace_policy=identity.get('trace_policy'),trace_only_changes=identity.get('trace_only_changes'),
                    exporter_sha256=sha(ROOT/'export_native_trace.py')))
                stage('timer_saved',bs)
                del raw_timer
            del timer
            group.barrier(f'BS{bs}/timer_export_ready')
        group.barrier(f'BS{bs}/output_ready')
    try:
        for bs in batches:run_case(bs)
    except BaseException as error:
        group.abort(f'EP{world} rank={rank} session={a.label}: {error}')
        raise
    buffer.destroy()
    group.close()

if __name__=='__main__': main()
