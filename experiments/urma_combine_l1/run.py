"""Production-size data-only Dispatch then Combine, with optional real GMM1/GMM2 and coupled W4 dequant."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from report import analyze,display


def stop_owned(processes):
    for p in processes:
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and any(p.poll() is None for p in processes):time.sleep(.1)
    for p in processes:
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        p.wait()


def launch(args,root,out,mode):
    out.mkdir()
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));endpoint=f'tcp://127.0.0.1:{s.getsockname()[1]}'
    env=os.environ.copy()
    shmem=Path(env.get('SHMEM_ROOT',root.parent.parent/'third-party/shmem/install/shmem')).resolve()
    env['LD_LIBRARY_PATH']=f"{root/'build'}:{shmem/'lib'}:"+env.get('LD_LIBRARY_PATH','')
    procs=[];logs=[];start=time.monotonic()
    try:
        for rank,device in enumerate(args.devices):
            log=(out/f'rank{rank}.log').open('w');logs.append(log)
            options={'rank':rank,'device':device,'ep-size':args.ep_size,'endpoint':endpoint,
                     'output':str(out),'bs':args.bs,'hidden':args.hidden,'intermediate':args.intermediate,'topk':args.topk,'shared-experts':args.shared_experts,'shared-intermediate':args.shared_intermediate,'compute-enable':int(mode=='on'),'communication-enable':int(args.communication=='on'),
                     'warmup':args.warmup,'iters':args.iters,'exp-repeat':args.exp_repeat,
                     'exp-rows':args.exp_shape[0],'exp-cols':args.exp_shape[1]}
            cmd=[str(root/'build/combine_l1')]
            for k,v in options.items():cmd.extend(['--'+k,str(v)])
            procs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True))
        while True:
            codes=[p.poll() for p in procs]
            failures=[i for i,c in enumerate(codes) if c not in (None,0)]
            if failures:raise RuntimeError(f'Failed ranks {failures}; inspect {out}/rank*.log')
            if all(c==0 for c in codes):break
            if time.monotonic()-start>args.timeout:raise TimeoutError(f'Timeout; logs: {out}')
            time.sleep(.2)
    finally:
        stop_owned(procs)
        for f in logs:f.close()
    result=analyze(out,args.devices,args.iters,args.group_size,args.ticks_per_us,mode=='on',
                   communication_enable=args.communication=='on',model=dict(bs=args.bs,hidden=args.hidden,intermediate=args.intermediate,topk=args.topk,shared_experts=args.shared_experts,shared_intermediate=args.shared_intermediate),peak_tflops=args.peak_tflops)
    result.update(compute=mode,dequant=mode,communication=args.communication)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    display(result,mode)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ep-size',type=int,choices=(4,8,16),default=16)
    p.add_argument('--bs',type=int,default=8192)
    p.add_argument('--hidden',type=int,default=5120)
    p.add_argument('--intermediate',type=int,default=2304)
    p.add_argument('--topk',type=int,choices=range(1,9),default=6)
    p.add_argument('--shared-experts',type=int,choices=(0,1),default=1)
    p.add_argument('--shared-intermediate',type=int,default=2304)
    p.add_argument('--devices',help='Comma-separated ACL device IDs; default 0..EP-1')
    p.add_argument('--compute',default='off,on',help='GMM1+GMM2+dequant together: off / on / off,on / off,on,on,off')
    p.add_argument('--communication',choices=('on','off'),default='on',help='off isolates GMM/dequant with all inputs prefilled; requires --compute on')
    p.add_argument('--peak-tflops',type=float,default=756.0,help='Dense FP8 peak per card for MFU, default756')
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--iters',type=int,default=20)
    p.add_argument('--exp-repeat',type=int,default=50)
    p.add_argument('--exp-shape',type=int,nargs=2,default=(100000,20000))
    p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--ticks-per-us',type=float,default=1000)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    args.devices=[int(x) for x in args.devices.split(',')] if args.devices else list(range(args.ep_size))
    modes=args.compute.split(',')
    if (not 256<=args.hidden<=8192 or args.hidden%256 or not 256<=args.intermediate<=16384 or args.intermediate%256 or not 256<=args.shared_intermediate<=16384 or args.shared_intermediate%256 or not 1024<=args.bs<=16384 or args.bs%128 or len(args.devices)!=args.ep_size or
        len(set(args.devices))!=len(args.devices) or min(args.devices)<0 or not all(m in ('off','on') for m in modes) or
        not 0<=args.warmup<=1000 or
        not 1<=args.iters<=1000 or not 0<=args.exp_repeat<=1000 or min(args.exp_shape)<=0 or
        args.exp_shape[0]*args.exp_shape[1]>2000000000 or args.timeout<=0 or
        args.group_size<=0 or not 0<args.ticks_per_us<1000000):p.error('Invalid configuration')
    if not math.isfinite(args.peak_tflops) or args.peak_tflops<=0:p.error('--peak-tflops must be positive and finite')
    if args.communication=='off' and 'off' in modes:p.error('--communication off requires --compute on (or on,on)')
    if os.name!='posix':p.error('Run on Linux UBX')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
        p.error('Unset ASCEND_RT_VISIBLE_DEVICES and select allocated devices explicitly with --devices (avoid double mapping)')
    root=Path(__file__).resolve().parent
    if not (root/'build/combine_l1').is_file():p.error('Run bash experiments/urma_combine_l1/build.sh first')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=(args.output or root.parent.parent/f'benchmark-runs/combine_l1/{stamp}_{os.getpid()}').resolve()
    out.mkdir(parents=True,exist_ok=False)
    config=vars(args).copy();config['output']=str(out)
    config.update(protocol='QP0 RO data only; Dispatch X+scale then Combine Y; production all-ready batching over CV256; no SO; one final quiet',
                  h=args.hidden,experts=384,
                  compute_schedule='production GMM classes, tile ownership, batch boundaries; routed production task order plus shared head/tail; independent nonzero prepared inputs',
                  cache='production default L2; no artificial bypass',
                  excluded='Pack/SwiGLU/Reduce/SO and stage-readiness gates; compute switch covers routed+shared GEMMs and coupled dequant',
                  operands='nonzero FP8/W4/scales; independent communication and GMM operands; post-run nonzero finite GMM check',
                  alignment='SHMEM device barrier -> Exp -> (device barrier + mixed kernel + device barrier)*rounds',
                  timer='stage envelopes only; timestamp writes after local completion join')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    snapshot=subprocess.run(['npu-smi','info'],text=True,capture_output=True,check=True)
    (out/'npu_smi.txt').write_text(snapshot.stdout+snapshot.stderr)
    print(f'[RESOURCE] Devices {args.devices} must be allocated and idle; snapshot: {out}/npu_smi.txt',flush=True)
    results=[]
    for index,mode in enumerate(modes):
        print(f'[RUN] {index} compute+dequant={mode}',flush=True)
        results.append(launch(args,root,out/f'{index:02d}_{mode}',mode))
    (out/'summary.json').write_text(json.dumps({'config':config,'runs':results},indent=2))
    fields=['run','compute','total_gbps','intra_gbps','cross_gbps',
            'total_round_max_mean_us','total_round_min_mean_us','gmm_round_max_mean_us','gmm_combine_coverage_percent','communication','peak_tflops','gmm_flops_per_card','mfu_percent']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for index,result in enumerate(results):
            classes=result['classes'];total=classes['TOTAL']
            writer.writerow(dict(zip(fields,[index,modes[index],total['mean_gbps'] if total else None,
                classes['INTRA']['mean_gbps'] if classes['INTRA'] else None,
                classes['CROSS']['mean_gbps'] if classes['CROSS'] else None,
                total['round_max_mean_us'] if total else None,total['round_min_mean_us'] if total else None,
                classes['GMM']['round_max_mean_us'] if classes['GMM'] else None,
                result['gmm_coverage_combine']*100 if result['gmm_coverage_combine'] is not None else None,
                args.communication,args.peak_tflops,
                result['mfu']['flops_per_card'] if result['mfu'] else None,
                result['mfu']['percent'] if result['mfu'] else None])))
    baseline=next((r for m,r in zip(modes,results) if m=='off'),None)
    if baseline:
        for index,(mode,result) in enumerate(zip(modes,results)):
            if mode!='on':continue
            for kind in ('TOTAL','INTRA','CROSS','COMBINE_TOTAL','COMBINE_CROSS'):
                off=baseline['classes'][kind];on=result['classes'][kind]
                if off and on:
                    change=100*(on['mean_gbps']/off['mean_gbps']-1)
                    print(f'[COMPARE] run={index} {kind} bandwidth vs first off: {change:+.2f}%')
    print(f'[PASSED] {out}/summary.json')


if __name__=='__main__':main()
