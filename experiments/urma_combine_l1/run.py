"""Fixed CV256 data+SO Combine, with optional simultaneous AIC GM->L1 read traffic."""
import argparse
import csv
import datetime
import json
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
                     'output':str(out),'bs':args.bs,'l1-enable':int(mode=='on'),
                     'l1-repeat':args.l1_repeat,'l1-gap-cycles':args.l1_gap_cycles,
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
    result=analyze(out,args.devices,args.iters,args.group_size,args.ticks_per_us,mode=='on')
    result.update(l1_load=mode,l1_repeat=args.l1_repeat,l1_gap_cycles=args.l1_gap_cycles)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    display(result,mode)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ep-size',type=int,choices=(4,8,16),default=16)
    p.add_argument('--bs',type=int,default=8192)
    p.add_argument('--devices',help='Comma-separated ACL device IDs; default 0..EP-1')
    p.add_argument('--l1-load',default='off,on',help='off / on / off,on; ABBA also allowed: off,on,on,off')
    p.add_argument('--l1-repeat',type=int,default=1)
    p.add_argument('--l1-gap-cycles',type=int,default=0,help='Throttle after each 4x64KiB AIC copy batch; 0 = no inner wait')
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--iters',type=int,default=20)
    p.add_argument('--exp-repeat',type=int,default=50)
    p.add_argument('--exp-shape',type=int,nargs=2,default=(100000,20000))
    p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--ticks-per-us',type=float,default=1000)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    args.devices=[int(x) for x in args.devices.split(',')] if args.devices else list(range(args.ep_size))
    modes=args.l1_load.split(',')
    if (not 1024<=args.bs<=8192 or args.bs%128 or len(args.devices)!=args.ep_size or
        len(set(args.devices))!=len(args.devices) or min(args.devices)<0 or not all(m in ('off','on') for m in modes) or
        not 1<=args.l1_repeat<=64 or not 0<=args.l1_gap_cycles<=10000000 or not 0<=args.warmup<=1000 or
        not 1<=args.iters<=1000 or not 0<=args.exp_repeat<=1000 or min(args.exp_shape)<=0 or
        args.exp_shape[0]*args.exp_shape[1]>2000000000 or args.timeout<=0 or
        args.group_size<=0 or not 0<args.ticks_per_us<1000000):p.error('Invalid configuration')
    if os.name!='posix':p.error('Run on Linux UBX')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
        p.error('Unset ASCEND_RT_VISIBLE_DEVICES and select allocated devices explicitly with --devices (avoid double mapping)')
    root=Path(__file__).resolve().parent
    if not (root/'build/combine_l1').is_file():p.error('Run bash experiments/urma_combine_l1/build.sh first')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=(args.output or root.parent.parent/f'benchmark-runs/combine_l1/{stamp}_{os.getpid()}').resolve()
    out.mkdir(parents=True,exist_ok=False)
    config=vars(args).copy();config['output']=str(out)
    config.update(protocol='main QP0 RO data + SO bitmap; CV256; one final quiet',
                  h=3584,intermediate=1280,experts=384,topk=8,
                  l1_base_bytes=args.bs*8*(3584*10+1280*14),
                  l1_scope='GMM1/GMM2 A operand reads per N256 projection, unique bypass-L2 HBM stream; no B/scales/FIX',
                  alignment='SHMEM device barrier -> exp50 -> (reset + device barrier + mixed kernel + device barrier)*rounds',
                  timer='two timestamps per active worker; written after local completion join, no per-CV timer')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    snapshot=subprocess.run(['npu-smi','info'],text=True,capture_output=True,check=True)
    (out/'npu_smi.txt').write_text(snapshot.stdout+snapshot.stderr)
    print(f'[RESOURCE] Devices {args.devices} must be allocated and idle; snapshot: {out}/npu_smi.txt',flush=True)
    results=[]
    for index,mode in enumerate(modes):
        print(f'[RUN] {index} l1={mode}',flush=True)
        results.append(launch(args,root,out/f'{index:02d}_{mode}',mode))
    (out/'summary.json').write_text(json.dumps({'config':config,'runs':results},indent=2))
    fields=['run','l1_load','combine_total_gbps','combine_intra_gbps','combine_cross_gbps',
            'combine_round_max_mean_us','combine_round_min_mean_us','l1_gbps','aic_active_coverage_percent']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for index,result in enumerate(results):
            classes=result['classes'];total=classes['TOTAL']
            writer.writerow(dict(zip(fields,[index,modes[index],total['mean_gbps'],
                classes['INTRA']['mean_gbps'] if classes['INTRA'] else None,
                classes['CROSS']['mean_gbps'] if classes['CROSS'] else None,
                total['round_max_mean_us'],total['round_min_mean_us'],
                classes['L1']['mean_gbps'] if classes['L1'] else None,
                result['aic_active_coverage_mean']*100])))
    baseline=next((r for m,r in zip(modes,results) if m=='off'),None)
    if baseline:
        for index,(mode,result) in enumerate(zip(modes,results)):
            if mode!='on':continue
            for kind in ('TOTAL','INTRA','CROSS'):
                off=baseline['classes'][kind];on=result['classes'][kind]
                if off and on:
                    change=100*(on['mean_gbps']/off['mean_gbps']-1)
                    print(f'[COMPARE] run={index} {kind} bandwidth vs first off: {change:+.2f}%')
    print(f'[PASSED] {out}/summary.json')


if __name__=='__main__':main()
