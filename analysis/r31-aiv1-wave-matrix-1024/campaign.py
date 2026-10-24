"""逐case全组空闲门禁、ABBA七BS；失败停止、保留日志，不占用他人NPU。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import tarfile
import threading


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True)
    p.add_argument('--host-keys',required=True)
    p.add_argument('--dependencies',required=True)
    p.add_argument('--packages',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rack',choices=('01','05','06','07'),required=True)
    p.add_argument('--nodes',required=True)
    p.add_argument('--ep',type=int,choices=(8,32),required=True)
    p.add_argument('--remote-root',required=True)
    a=p.parse_args()
    if not re.fullmatch('/home/codex-deepep/[a-zA-Z0-9_-]+',a.remote_root):
        raise ValueError('Invalid isolated remote root')
    sys.path.insert(0,a.dependencies)
    import openpyxl
    import paramiko
    book=openpyxl.load_workbook(a.inventory,read_only=True,data_only=True)
    sheet=next(s for s in book if s.title.endswith(a.rack))
    inventory={r[0]:r[:4] for r in sheet.iter_rows(min_row=2,values_only=True)}
    nodes=a.nodes.split(','); count=a.ep//8
    if len(nodes)!=count or len(set(nodes))!=count or any(n not in inventory for n in nodes):
        raise ValueError('EP placement mismatch')
    source=Path(__file__).resolve().parent
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(a.packages) as t:
        packages=json.loads(t.extractfile('packages.json').read())
    if set(packages)!={'control','combined'} or any(v['protocol']!=8 for v in packages.values()):
        raise ValueError('Package ABI mismatch')
    manifest=dict(status='preparing',ep=a.ep,nodes=nodes,rack=a.rack,bs_values=[32,96,256,512,1024,2048,4096],
        packages=packages,accuracy='not_run_user_requested',remote_root=a.remote_root,
        package_archive_sha256=hashlib.sha256(a.packages.read_bytes()).hexdigest(),
        schedule=['control-a','combined-a','combined-b','control-b'],completed=[],
        scope='full communication kernel Duration; timer-off; last10/20; mean per-round rank-min',
        timer='separate rank0 full passive pass on a repeats only',
        harness_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir() if f.is_file()})
    lock=threading.Lock()
    def state(stage,**fields):
        event=dict(stage=stage,time_utc=datetime.now(timezone.utc).isoformat(),**fields)
        with lock:
            with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    def save(status):
        manifest['status']=status
        (out/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def connect(i):
        _,ip,user,password=inventory[nodes[i]]
        c=paramiko.SSHClient();c.load_host_keys(a.host_keys)
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
        c.connect(ip,username=user,password=password,timeout=15,look_for_keys=False,allow_agent=False)
        c.get_transport().set_keepalive(20)
        return c
    def command(i,cmd,label,barrier=None):
        with connect(i) as c:
            ch=c.get_transport().open_session(timeout=15);ch.set_combine_stderr(True);ch.settimeout(960);ch.exec_command(cmd)
            with (out/f'{label}-{nodes[i]}.log').open('xb') as log:
                if barrier is not None:
                    pending=b'';ch.settimeout(30)
                    try:
                        while b'COHORT_READY\n' not in pending:
                            b=ch.recv(65536)
                            if not b:raise RuntimeError('Node not idle/ready')
                            log.write(b);log.flush();pending+=b
                        state('cohort_ready',label=label,node=nodes[i]);barrier.wait(timeout=30)
                        ch.sendall(b'GO\n');state('cohort_released',label=label,node=nodes[i])
                    except Exception:
                        barrier.abort()
                        try:ch.sendall(b'CANCEL\n')
                        except OSError:pass
                        state('cohort_abort',label=label,node=nodes[i])
                    ch.shutdown_write();ch.settimeout(960)
                while b:=ch.recv(65536):log.write(b);log.flush()
            code=ch.recv_exit_status();state('node_exit',label=label,node=nodes[i],exit=code)
            return code
    def parallel(fn):
        with ThreadPoolExecutor(max_workers=count) as pool:return list(pool.map(fn,range(count)))
    def deploy(i):
        if command(i,'mkdir '+a.remote_root,'prepare'):raise RuntimeError('Remote root exists/unavailable')
        with connect(i) as c,c.open_sftp() as s:
            for n in ('benchmark.py','coordinator.py','cohort_gate.sh','export_native_trace.py','probe.py','run_node.sh'):
                s.put(str(source/n),a.remote_root+'/'+n)
            s.put(str(a.packages),a.remote_root+'/packages.tar.gz')
        check="import pathlib,json,hashlib;p=pathlib.Path('.');rs=json.loads((p/'packages.json').read_text());assert all(hashlib.sha256((p/'packages'/m/n).read_bytes()).hexdigest()==h for m,r in rs.items() for n,h in r['native'].items())"
        cmd=f'cd {a.remote_root} && test $(sha256sum packages.tar.gz | cut -d" " -f1) = {manifest["package_archive_sha256"]} && tar -xzf packages.tar.gz && /root/miniconda3/envs/zyx/bin/python -I -c {shlex.quote(check)}'
        if command(i,cmd,'deploy'):raise RuntimeError('Deployment identity mismatch')
    def collect(i,label,probe=False):
        relative='preflight/'+label if probe else 'runs/'+label
        archive='archive-'+label+'-'+nodes[i]+'.tar.gz'
        cmd=(f'cd {a.remote_root} && find {relative} -type f \\( -name "kernel_details*.csv" -o -name "native_sparse.npz" '
             '-o -name "timer_summary.csv" -o \\( -name "*.json" ! -path "*/profiler/*" ! -path "*/timer_profiler/*" \\) \\) '
             f'-print0 | tar --null -T - -czf {archive}')
        if command(i,cmd,'collect-'+label):raise RuntimeError('Collection failed')
        target=out/archive
        with connect(i) as c,c.open_sftp() as s:s.get(a.remote_root+'/'+archive,str(target))
        with tarfile.open(target) as t:t.extractall(out/f'node{i}',filter='data')
        state('evidence_saved',label=label,node=nodes[i],sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    def launch(method,bs,label,trace):
        barrier=threading.Barrier(count)
        master=inventory[nodes[0]][1]
        cmd=lambda i:f'bash {a.remote_root}/run_node.sh {i} {master} {a.ep} {method} {bs} {label} {trace}'
        state('starting',label=label,method=method,bs=bs,ep=a.ep)
        if any(parallel(lambda i:command(i,cmd(i),label,barrier))):raise RuntimeError('Failed node; no next case')
        parallel(lambda i:collect(i,label,trace=='probe'))
        if trace!='probe':
            folder=out/label;folder.mkdir()
            for name in ['performance.json']+(['timer_on_performance.json'] if trace=='0' else []):
                with connect(0) as c,c.open_sftp() as s:s.get(a.remote_root+'/runs/'+label+'/'+name,str(folder/name))
            result=json.loads((folder/'performance.json').read_text())
            state('run_complete',label=label,mean_us=result['mean_us'],rank_max_mean_us=result['rank_max_mean_us'])
        manifest['completed'].append(label);save('running')
    save('preparing')
    try:
        parallel(deploy)
        launch('combined',32,'preflight','probe')
        for bs in manifest['bs_values']:
            for job in manifest['schedule']:
                method,repeat=job.split('-')
                launch(method,bs,f'bs{bs}-{job}','0' if repeat=='a' else 'none')
        save('collected_not_yet_audited')
    except Exception as error:
        save('failed');state('phase_failed',error_type=type(error).__name__)
        raise


if __name__=='__main__':main()
