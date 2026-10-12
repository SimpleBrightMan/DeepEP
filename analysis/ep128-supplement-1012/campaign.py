"""Isolated EP128 performance-only retry on authorized racks 07+01."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import time

W=Path('C:/Project/Infra/ops-transformer-hs')
HERE=Path(__file__).resolve().parent
OUT=W/'outputs/ep128-supplement-1012-r07-r01'
REMOTE='/home/zyx/ep128-supplement-1012-r07-r01'
WHEEL=W/'tmp/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl'
SHA='f39cc15c9c252a689b6e2bbdde4fcc78d9566be73d0a1cce5577048bd2962cc3'
ORDER=[('07','CPU7')]+[('07',f'CPU{i}') for i in (1,2,3,4,5,6,8)]+[('01',f'CPU{i}') for i in range(1,9)]
sys.path.insert(0,str(W/'tmp/rack-audit-deps'))
import openpyxl
import paramiko

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-preparation',action='store_true',help='Resume only before any performance stage was launched')
    parser.add_argument('--resume-preflight',action='store_true',help='Verify completed deployments without upload and retry the unstarted device preflight')
    args=parser.parse_args()
    assert not (args.resume_preparation and args.resume_preflight)
    resume=args.resume_preparation or args.resume_preflight
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest()==SHA
    OUT.mkdir(exist_ok=resume)
    book=openpyxl.load_workbook('C:/Project/Agent/乌兰512p_8柜.xlsx',read_only=True,data_only=True)
    secrets={}
    for rack in ('07','01'):
        sheet=next(s for s in book if s.title.endswith(rack))
        secrets.update({(rack,r[0]):r[1:4] for r in sheet.iter_rows(min_row=2,values_only=True) if r[0]})
    book.close()
    state=dict(status='preparing',remote=REMOTE,native_commit='db2c017',wheel_sha256=SHA,
        started=time.time(),accuracy='not_run_user_requested',topology='07+01; changed from historical07+05',
        hosts=[dict(rack=r,node=n,ip=secrets[r,n][0]) for r,n in ORDER],stages=[])
    if resume:
        previous=json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
        assert previous['status']=='stopped' and not previous['stages'], 'Partial/running performance must not be restarted'
        assert all(previous[k]==state[k] for k in ('remote','native_commit','wheel_sha256','hosts'))
        state=previous
        state.setdefault('preparation_retries',[]).append(dict(at=time.time(),previous_error=state.pop('error',None)))
        state['status']='preparing'
    def save():
        (OUT/'campaign.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    def connect(key):
        c=paramiko.SSHClient(); c.load_host_keys(str(W/'tmp/cpu7_known_hosts'))
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
        ip,user,password=secrets[key]
        c.connect(ip,username=user,password=password,timeout=20,look_for_keys=False,allow_agent=False)
        return c
    def command(key,cmd,timeout=60):
        c=connect(key)
        try:
            _,out,err=c.exec_command(cmd,timeout=timeout)
            stdout=out.read().decode(errors='replace'); stderr=err.read().decode(errors='replace')
            rc=out.channel.recv_exit_status()
            if rc: raise RuntimeError(f'{key} rc={rc}: {stdout[-1800:]} {stderr[-1800:]}')
            return stdout
        finally: c.close()
    def fetch(key,remote,local):
        c=connect(key)
        try:
            with c.open_sftp() as s: s.get(remote,str(local))
        finally: c.close()
    def parallel(fn,keys=ORDER):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(fn,keys))
    def idle(key):
        value=command(key,'npu-smi info')
        (OUT/f'idle-{key[0]}-{key[1]}.txt').write_text(value,encoding='utf-8')
        if set(re.findall(r'No running processes found in NPU (\d+)',value))!=set(map(str,range(8))):
            raise RuntimeError(f'{key} busy; no other job will be stopped')
    save()
    try:
        parallel(idle)
        deps=OUT/'dependencies'; deps.mkdir(exist_ok=resume)
        if not args.resume_preflight:
            for name in ('aclshmem_bootstrap_config_store.so','aclshmem_bootstrap_uid.so','libshmem.so','libshmem_utils.so'):
                fetch(ORDER[0],'/home/zyx/ep32-main-24b4221/shmem/lib/'+name,deps/name)
            for source,name in [('ep128-ds-test.py','test_mega_moe.py'),('ep128-ds-utils.py','utils.py'),('ep128-ds-measure.py','megamoe_measure.py')]:
                fetch(ORDER[0],'/home/zyx/'+source,deps/name)
        state['dependency_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in deps.iterdir()}
        save()
        def prepare(key):
            # 只创建隔离目录和缺失的zyx副本，不更新他人的hkq环境。
            if args.resume_preparation:
                actual=command(key,f'sha256sum {REMOTE}/{WHEEL.name}').split()[0]
                assert actual==SHA, (key,'deployed wheel changed')
            else:
                command(key,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/shmem {REMOTE}/ds-python {REMOTE}/ds-harness')
            c=connect(key)
            try:
                with c.open_sftp() as s:
                    files=[(WHEEL,REMOTE+'/'+WHEEL.name)]
                    files += [(p,REMOTE+'/'+p.name) for p in (HERE/'entry.py',HERE/'ds_entry.py',HERE/'run_node.sh',HERE/'config.json')]
                    files += [(p,REMOTE+('/shmem/' if p.suffix=='.so' else '/ds-harness/')+p.name) for p in deps.iterdir()]
                    files += [(W/'tmp'/name,REMOTE+'/'+name) for name in ('ep128-ds-package.tar.gz','ep128-ds-deps.tar.gz','ep128-ds-metadata.tar.gz')]
                    for local,remote in files: s.put(str(local),remote)
            finally: c.close()
            result=command(key,f'''set -e
if [ ! -d /root/miniconda3/envs/zyx ]; then
 /root/miniconda3/bin/conda create --offline -y --prefix /root/miniconda3/envs/zyx --clone /root/miniconda3/envs/hkq > {REMOTE}/clone-prefix.log 2>&1
fi
test -x /root/miniconda3/envs/zyx/bin/python
/root/miniconda3/envs/zyx/bin/python -I -m zipfile -e {REMOTE}/{WHEEL.name} {REMOTE}/package
tar -xzf {REMOTE}/ep128-ds-package.tar.gz -C {REMOTE}/ds-python
tar -xzf {REMOTE}/ep128-ds-deps.tar.gz -C {REMOTE}/ds-python
tar -xzf {REMOTE}/ep128-ds-metadata.tar.gz -C {REMOTE}/ds-python
sha256sum {REMOTE}/{WHEEL.name} {REMOTE}/ds-python/deep_gemm/include/deep_gemm/mega_moe.hpp
''',timeout=900)
            assert result.split()[0]==SHA,(key,result)
            (OUT/f'identity-{key[0]}-{key[1]}.txt').write_text(result)
            print('PREPARED',*key,flush=True)
        if args.resume_preflight:
            def verify(key):
                identity=(OUT/f'identity-{key[0]}-{key[1]}.txt').read_text().splitlines()
                expected={WHEEL.name:SHA,'ds-python/deep_gemm/include/deep_gemm/mega_moe.hpp':identity[1].split()[0]}
                expected.update({name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('entry.py','ds_entry.py','run_node.sh','config.json')})
                expected.update({('shmem/' if p.suffix=='.so' else 'ds-harness/')+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in deps.iterdir()})
                actual=command(key,'sha256sum '+' '.join(shlex.quote(REMOTE+'/'+name) for name in expected))
                hashes={line.split(maxsplit=1)[1].removeprefix(REMOTE+'/'):line.split()[0] for line in actual.splitlines()}
                assert hashes==expected,(key,'deployed file identity changed; do not overwrite automatically')
                print('VERIFIED',*key,flush=True)
            parallel(verify)
        else:
            parallel(prepare)
        state['status']='device_probe'; save()
        def probe(key):
            try:
                value=command(key,f'bash {REMOTE}/run_node.sh 0 probe probe',timeout=340)
                (OUT/f'probe-{key[0]}-{key[1]}.log').write_text(value)
                assert value.count('DEVICE_OK')==8,(key,value)
                print('PROBE_OK',*key,flush=True)
            except Exception as error:
                (OUT/f'probe-{key[0]}-{key[1]}-failed-{int(time.time())}.log').write_text(str(error),encoding='utf-8')
                return str(error)
        errors=[value for value in parallel(probe) if value]
        if errors: raise RuntimeError('Device preflight failures: '+'; '.join(errors))
        for method,run in [('ours','ours1'),('ds','ds1'),('ds','ds2'),('ours','ours2')]:
            parallel(idle)
            rec=dict(method=method,run=run,status='launching',jobs=[],started=time.time())
            state['stages'].append(rec); state['status']=run; save()
            def launch(pair):
                rank,key=pair
                inner=f'bash {REMOTE}/run_node.sh {rank} {method} {run}; rc=$?; echo "$rc" > {REMOTE}/{run}.rc; exit "$rc"'
                pid=int(command(key,f'nohup bash -c {shlex.quote(inner)} > {REMOTE}/{run}.log 2>&1 < /dev/null & echo $!').strip())
                return dict(rack=key[0],node=key[1],pid=pid,rank=rank)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for job in pool.map(launch,enumerate(ORDER)):
                    rec['jobs'].append(job); save()
            rec['status']='running'; save(); print('START',run,flush=True)
            pending=set(ORDER); deadline=time.monotonic()+1950
            while pending:
                for key in list(pending):
                    result=command(key,f'if test -f {REMOTE}/{run}.rc; then cat {REMOTE}/{run}.rc; else echo RUNNING; fi').strip()
                    if result=='RUNNING': continue
                    next(j for j in rec['jobs'] if (j['rack'],j['node'])==key)['returncode']=int(result)
                    fetch(key,f'{REMOTE}/{run}.log',OUT/f'{run}-{key[0]}-{key[1]}.log')
                    pending.remove(key); save()
                if pending:
                    if time.monotonic()>deadline: raise RuntimeError(f'{run} monitoring deadline; inspect own recorded PIDs')
                    time.sleep(10)
            if any(j['returncode'] for j in rec['jobs']):
                rec['status']='failed'; save(); raise RuntimeError(f'{run} failed; see logs, no auto retry')
            rec['reports']=[]
            for bs in ([32,96,256,512,1024,2048,4096] if method=='ours' else [512,2048]):
                remote=f'{REMOTE}/{run}/'+(f'rank_0/bs_{bs}/profiling.json' if method=='ours' else f'bs_{bs}/profiling.json')
                local=OUT/f'{run}-bs{bs}.json'; fetch(ORDER[0],remote,local)
                report=json.loads(local.read_text())
                assert len(report['per_rank_rounds_us'])==128 and report['aggregation']=='mean_of_per_round_min_rank'
                rec['reports'].append(dict(bs=bs,mean_us=report['mean_us'],file=local.name))
            rec['status']='complete'; rec['finished']=time.time(); save()
            print('DONE',run,rec['reports'],flush=True)
        state['status']='complete'; state['finished']=time.time(); save(); print('COMPLETE',flush=True)
    except BaseException as error:
        state['status']='stopped'; state['error']=str(error); save(); raise

if __name__=='__main__': main()
