"""Serial EP8/32/128 verification. Credentials stay in memory; never stop other jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import time

WORKSPACE = Path('C:/Project/Infra/ops-transformer-hs')
sys.path.insert(0, str(WORKSPACE/'tmp/rack-audit-deps'))
import paramiko
import openpyxl

HERE = Path(__file__).resolve().parent
REMOTE = '/home/zyx/unified-best-db2c017-1012'
WHEEL_NAME = 'ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl'
PYTHON = '/root/miniconda3/envs/zyx/bin/python'
BATCHES = [32, 96, 256, 1024, 4096]
ORDER = [('07', 'CPU7')] + [('07', f'CPU{i}') for i in (1,2,3,4,5,6,8)] + [('05', f'CPU{i}') for i in range(1,9)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Only resume after every launched stage completed; never restart a partial phase')
    args = parser.parse_args()
    plan = json.loads((HERE/'plan.json').read_text())
    wheel = WORKSPACE/'tmp'/WHEEL_NAME
    if digest(wheel) != plan['wheel_sha256']:
        raise RuntimeError('Candidate wheel SHA mismatch')
    book = openpyxl.load_workbook('C:/Project/Agent/乌兰512p_8柜.xlsx', read_only=True, data_only=True)
    secrets = {}
    for rack in ('07', '05'):
        sheet = next(s for s in book if s.title.endswith(rack))
        secrets.update({(rack,r[0]): r[1:4] for r in sheet.iter_rows(min_row=2, values_only=True) if r[0]})
    book.close()
    public = [dict(rack=r, node=n, ip=secrets[r,n][0], global_node_rank=i) for i,(r,n) in enumerate(ORDER)]
    if args.dry_run:
        print(json.dumps(dict(plan=plan, hosts=public), ensure_ascii=False, indent=2)); return
    if args.resume:
        state=json.loads((args.output/'campaign.json').read_text(encoding='utf-8'))
        def settled(s):
            return s['status']=='complete' or (s['stage']=='accuracy' and s['status']=='cancelled_by_user')
        if state['status']!='stopped' or not all(settled(s) for s in state['stages']):
            raise RuntimeError('Resume requires stopped state with no partial/running phase')
        if state['plan']!=plan or state['hosts']!=public or state['remote_root']!=REMOTE:
            raise RuntimeError('Resume configuration changed')
        state.setdefault('resumes',[]).append(dict(at=time.time(),previous_error=state.pop('error',None),runner_sha256=digest(Path(__file__))))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        state = dict(status='preparing', plan=plan, hosts=public, remote_root=REMOTE, stages=[],
                     runner_sha256=digest(Path(__file__)), started=time.time())

    def save():
        (args.output/'campaign.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    def connect(key):
        host,user,password = secrets[key]
        client = paramiko.SSHClient()
        client.load_host_keys(str(WORKSPACE/'tmp/cpu7_known_hosts'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(host, username=user, password=password, timeout=20, look_for_keys=False, allow_agent=False)
        return client

    def execute(key, command, timeout=50):
        client = connect(key)
        try:
            _, out, err = client.exec_command(command, timeout=timeout)
            output = out.read().decode(errors='replace')
            error = err.read().decode(errors='replace')
            code = out.channel.recv_exit_status()
            if code:
                raise RuntimeError(f'{key} command failed rc={code}: {error[-1500:]} {output[-1500:]}')
            return output
        finally:
            client.close()

    def parallel(keys, function):
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(function, keys))

    def idle(keys, label):
        def check(key):
            snapshot = execute(key, 'npu-smi info')
            (args.output/f'{label}-{key[0]}-{key[1]}.txt').write_text(snapshot, encoding='utf-8')
            if set(re.findall(r'No running processes found in NPU (\d+)', snapshot)) != set(map(str,range(8))):
                raise RuntimeError(f'{key} NPU occupied or unknown; no other job will be stopped')
            return True
        parallel(keys, check)

    def prepare(key):
        execute(key, f'test ! -e {REMOTE} && mkdir {REMOTE}')
        client = connect(key)
        try:
            with client.open_sftp() as transfer:
                for local in (wheel, HERE/'entry.py', HERE/'run_node.sh', HERE/'config.json', HERE/'plan.json'):
                    transfer.put(str(local), REMOTE+'/'+local.name)
        finally:
            client.close()
        script = ("from pathlib import Path; import hashlib,zipfile,json; "
                  f"r=Path({REMOTE!r}); w=r/{WHEEL_NAME!r}; "
                  f"assert hashlib.sha256(w.read_bytes()).hexdigest()=={plan['wheel_sha256']!r}; "
                  "zipfile.ZipFile(w).extractall(r/'package'); print('ISOLATED_WHEEL_OK')")
        result = execute(key, f'{PYTHON} -I -c {shlex.quote(script)}')
        if 'ISOLATED_WHEEL_OK' not in result:
            raise RuntimeError(f'{key} missing deployment verdict')
        return key

    def fetch(key, remote, local):
        client=connect(key)
        try:
            with client.open_sftp() as transfer:
                transfer.get(remote, str(local))
        finally:
            client.close()

    def phase(ep, stage, run):
        # 用户明确取消的精度阶段不重启；保留取消记录，不伪装为通过。
        if stage=='accuracy' and any(s['ep']==ep and s['run']==run and s['status']=='cancelled_by_user' for s in state['stages']):
            print('SKIP_USER_CANCELLED_ACCURACY', ep, flush=True)
            return
        if any(s['ep']==ep and s['run']==run and s['status']=='complete' for s in state['stages']):
            return
        keys = ORDER[:ep//8]
        idle(keys, f'ep{ep}-{run}-before')
        label=f'ep{ep}-{run}'
        record=dict(ep=ep, stage=stage, run=run, status='launching', jobs=[], started=time.time())
        state['stages'].append(record); state['status']=label; save()
        print('START', label, flush=True)
        def launch(pair):
            rank,key=pair
            command = f'bash {REMOTE}/run_node.sh {rank} {ep} {stage} {run}'
            inner = f'{command}; result=$?; printf "%s\\n" "$result" > {REMOTE}/{label}.rc; exit "$result"'
            output=execute(key, f'nohup bash -c {shlex.quote(inner)} > {REMOTE}/{label}.log 2>&1 < /dev/null & echo $!')
            return dict(rack=key[0], node=key[1], pid=int(output.strip()), rank=rank)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for job in pool.map(launch, enumerate(keys)):
                record['jobs'].append(job); save()
        record['status']='running'; save()
        started=time.monotonic()
        pending=set(keys)
        while pending:
            for key in list(pending):
                result=execute(key, f'if test -f {REMOTE}/{label}.rc; then cat {REMOTE}/{label}.rc; else echo RUNNING; fi')
                if result.strip()!='RUNNING':
                    rc=int(result.strip()); pending.remove(key)
                    job=next(j for j in record['jobs'] if (j['rack'],j['node'])==key)
                    job['returncode']=rc
                    fetch(key, f'{REMOTE}/{label}.log', args.output/f'{label}-{key[0]}-{key[1]}.log')
                    save()
            if pending:
                if time.monotonic()-started>1900:
                    raise RuntimeError(f'{label}: monitor deadline, inspect recorded own PIDs; not auto-restarted')
                time.sleep(12)
        if any(j['returncode'] for j in record['jobs']):
            record['status']='failed'; save(); raise RuntimeError(f'{label}: node failure, see per-node logs')
        if stage != 'golden':
            record['reports']=[]
            for bs in BATCHES:
                filename = 'accuracy.json' if stage=='accuracy' else 'profiling.json'
                remote=f'{REMOTE}/{label}/rank_0/bs_{bs}/{filename}'
                local=args.output/f'{label}-bs{bs}-{filename}'
                fetch(keys[0], remote, local)
                value=json.loads(local.read_text())
                if stage=='accuracy':
                    if len(value)!=ep or not all(v['passed'] for v in value):
                        raise RuntimeError(f'{label} BS{bs}: incomplete/failed accuracy')
                elif len(value['per_rank_rounds_us'])!=ep or value['measurement']!='full_moe':
                    raise RuntimeError(f'{label} BS{bs}: incorrect measurement population')
                record['reports'].append(dict(bs=bs, remote=remote, local=local.name, report=value))
            fetch(keys[0], f'{REMOTE}/{label}/rank_0/manifest.json', args.output/f'{label}-manifest.json')
        record['status']='complete'; record['finished']=time.time(); save()
        print('DONE', label, [(r['bs'],round(r['report']['mean_us'],3)) for r in record.get('reports',[]) if stage=='perf'], flush=True)

    save()
    try:
        idle(ORDER, f"resume{len(state.get('resumes',[]))}" if args.resume else 'initial')
        if args.resume:
            # 原测试均已正常退出；只读校验已有隔离环境，不覆盖代码、数据或输出。
            def verify(key):
                actual=execute(key, f'sha256sum {REMOTE}/{WHEEL_NAME}').split()[0]
                if actual!=plan['wheel_sha256']:
                    raise RuntimeError(f'{key}: resumed wheel hash mismatch')
                for name in ('entry.py','run_node.sh','config.json'):
                    actual=execute(key, f'sha256sum {REMOTE}/{name}').split()[0]
                    if actual!=digest(HERE/name):
                        raise RuntimeError(f'{key}: resumed {name} mismatch')
            parallel(ORDER,verify)
        else:
            parallel(ORDER, prepare)
        state['deployment']='all16 same isolated wheel'; save()
        for ep in (8,32,128):
            phase(ep, 'golden', 'golden')
            phase(ep, 'accuracy', 'accuracy')
            for repeat in (1,2):
                phase(ep, 'perf', f'perf{repeat}')
        state['status']='complete'; state['finished']=time.time(); save()
        print('UNIFIED_COMPLETE', flush=True)
    except BaseException as error:
        state['status']='stopped'; state['error']=str(error); save()
        raise


if __name__=='__main__':
    main()
