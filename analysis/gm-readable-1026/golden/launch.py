"""Gate and collect an already-deployed full Golden campaign; credentials only in memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import sys
import tarfile
import threading


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ep',type=int,choices=(8,32),required=True)
    p.add_argument('--rack',choices=('01','05'),default='05')
    p.add_argument('--cpu-only',action='store_true',help='32-rank CPU metadata probe; no NPU initialization')
    p.add_argument('--nodes',required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+',a.label):raise ValueError('Unsafe label')
    nodes=a.nodes.split(',')
    if len(nodes)!=a.ep//8 or len(set(nodes))!=len(nodes):raise ValueError('EP placement mismatch')
    workspace=Path('C:/Project/Infra/ops-transformer-hs')
    sys.path.insert(0,str(workspace/'tmp/rack-audit-deps'))
    import openpyxl,paramiko
    sheet=next(s for s in openpyxl.load_workbook('C:/Project/Agent/乌兰512p_8柜.xlsx',read_only=True,data_only=True)
        if s.title.endswith(a.rack))
    inventory={row[0]:row[:4] for row in sheet.iter_rows(min_row=2,values_only=True)}
    if any(node not in inventory for node in nodes):raise ValueError('Unknown node')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    control_token=secrets.token_hex(32) # Only in memory and SSH stdin, not in logs/manifests.
    root='/home/codex-deepep/gm-readable-golden-a-1027'
    lock=threading.Lock();barrier=threading.Barrier(len(nodes));clients=[None]*len(nodes)
    names=(('common.py','coordinator.py','probe_cpu.py','run_cpu.sh','local_workers.py') if a.cpu_only else
        ('common.py','worker.py','coordinator.py','cohort_gate.sh','run_accuracy.sh','local_workers.py'))
    manifest=dict(status='preparing',ep=a.ep,rack=a.rack,nodes=nodes,label=a.label,port=a.port,cpu_only=a.cpu_only,
        candidate='8118a7647f43db5e1fe86116a1d473514bcb6134',
        check_tokens=0,precision='repository CPU mixed error; unchanged thresholds',
        hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names})
    def save():
        (out/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def event(stage,**fields):
        value=dict(stage=stage,time_utc=datetime.now(timezone.utc).isoformat(),**fields)
        with lock:
            with (out/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
            print(json.dumps(value),flush=True)
    def connect(node):
        _,ip,user,password=inventory[node]
        client=paramiko.SSHClient();client.load_host_keys(str(workspace/'tmp/cpu7_known_hosts'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        extra=paramiko.HostKeys(str(workspace/'tmp/r31-continuous-jump-public-known-hosts')).lookup(ip)
        for kind,key in (extra or {}).items():
            known=client.get_host_keys().lookup(ip)
            if known and kind in known and known[kind]!=key:raise ValueError('Conflicting trusted host key')
            client.get_host_keys().add(ip,kind,key)
        client.connect(ip,username=user,password=password,timeout=15,look_for_keys=False,allow_agent=False)
        client.get_transport().set_keepalive(20);return client
    def run(index):
        node=nodes[index];client=clients[index]
        script='run_cpu.sh' if a.cpu_only else 'run_accuracy.sh'
        cmd=f'bash {root}/{script} {index} {inventory[nodes[0]][1]} {a.ep} {a.label} {a.port}'
        channel=client.get_transport().open_session(timeout=15);channel.set_combine_stderr(True)
        channel.exec_command(cmd);channel.settimeout(30)
        with (out/f'{node}.log').open('xb') as log:
            received=b''
            try:
                while b'COHORT_READY\n' not in received:
                    part=channel.recv(65536)
                    if not part:raise RuntimeError('Node not ready')
                    received+=part;log.write(part);log.flush()
                event('ready',node=node);barrier.wait(timeout=30)
                channel.sendall(('GO\n'+control_token+'\n').encode());event('released',node=node)
            except Exception:
                barrier.abort()
                try:channel.sendall(b'CANCEL\n')
                except OSError:pass
                event('cancelled',node=node)
            channel.shutdown_write();channel.settimeout(1250)
            while part:=channel.recv(65536):log.write(part);log.flush()
            code=channel.recv_exit_status();event('exit',node=node,code=code)
            return code
    save()
    try:
        def prepare(index):
            node=nodes[index];client=connect(node);clients[index]=client
            code='import hashlib,pathlib,json;p=pathlib.Path('+repr(root)+');expected='+repr(manifest['hashes'])+';assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in expected.items());m=json.loads((p/'+repr(f'cpu-ep{a.ep}/prepared-ep{a.ep}-r{index*8}.json')+').read_text());assert m["ep"]=='+str(a.ep)+' and m["ranks"]=='+repr(list(range(index*8,index*8+8)))
            import shlex
            _,stdout,stderr=client.exec_command('/root/miniconda3/envs/zyx/bin/python -I -c '+shlex.quote(code),timeout=30)
            err=stderr.read().decode(errors='replace');rc=stdout.channel.recv_exit_status()
            if rc:raise RuntimeError(f'{node}: prepared script/CPU Golden mismatch: {err}')
            event('prepared_verified',node=node)
        # 节点认证/只读核对并行，缩短空闲探测到全组门禁之间的窗口；不省略任何门禁。
        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:list(pool.map(prepare,range(len(nodes))))
        manifest['status']='running';save()
        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:codes=list(pool.map(run,range(len(nodes))))
        manifest['exit_codes']=codes
        # 失败也收集已产生的JSON；门禁拒绝时没有输出目录，不能伪造完成证据。
        for index,node in enumerate(nodes):
            client=clients[index]
            archive=f'{a.label}-{node}.tar.gz'
            _,exists,_=client.exec_command(f'test -d {root}/{a.label}',timeout=15)
            if exists.channel.recv_exit_status():
                event('no_worker_artifacts',node=node);continue
            cmd=f'cd {root} && tar -czf {archive} {a.label}'
            _,stdout,stderr=client.exec_command(cmd,timeout=30)
            if stdout.channel.recv_exit_status():raise RuntimeError(f'{node}: evidence collection failed')
            with client.open_sftp() as sftp:sftp.get(root+'/'+archive,str(out/archive))
            with tarfile.open(out/archive) as tar:tar.extractall(out/f'node{index}',filter='data')
            event('collected',node=node)
        manifest['status']='collected_not_audited' if not any(codes) else 'failed';save()
        if any(codes):raise RuntimeError('Golden worker/gate failed; no promotion')
    except BaseException:
        manifest['status']='failed';save();raise
    finally:
        for client in clients:
            if client is not None:client.close()


if __name__=='__main__':main()
