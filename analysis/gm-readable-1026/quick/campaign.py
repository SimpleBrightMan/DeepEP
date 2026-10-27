"""支持提前部署、校验后单轮B筛选；保留全组空闲门禁与连续多BS。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import tarfile
import threading


def measurement_schedule(single_start, method_only=None):
    if method_only not in (None,'control','combined'):raise ValueError('Unknown frozen method')
    if single_start:return [(method_only or 'combined')+'-a']
    return [method_only+'-a',method_only+'-b'] if method_only else ['control-a','combined-a','combined-b','control-b']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True)
    p.add_argument('--host-keys',required=True)
    p.add_argument('--dependencies',required=True)
    p.add_argument('--jump-rack',choices=tuple(f'{n:02d}' for n in range(1,9)))
    p.add_argument('--jump-node',default='CPU1')
    p.add_argument('--additional-host-keys',help='Public known_hosts from an authenticated trusted host')
    p.add_argument('--packages',type=Path,required=True)
    p.add_argument('--package-cache-root',help='Existing per-node archive cache; SHA verified before copying')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rack',choices=tuple(f'{n:02d}' for n in range(1,9)),required=True)
    p.add_argument('--nodes',required=True)
    p.add_argument('--ep',type=int,choices=(8,32),required=True)
    p.add_argument('--remote-root',required=True)
    p.add_argument('--bs-values',type=int,nargs='+',default=[512,1024,4096])
    p.add_argument('--trace',action='store_true',help='Separate passive timer pass for a repeats')
    phase=p.add_mutually_exclusive_group()
    phase.add_argument('--prepare-only',action='store_true',help='Deploy immutable packages now; no NPU worker')
    phase.add_argument('--run-prepared',action='store_true',help='Verify existing deployment; never upload/redeploy')
    p.add_argument('--single-start',action='store_true',help='One candidate start, no separate probe; screening only')
    p.add_argument('--method-only',choices=('control','combined'),help='Single frozen version, two independent starts per BS')
    p.add_argument('--isolated-bs',action='store_true',help='Capacity diagnostic only: restart workers per BS')
    p.add_argument('--buffer-capacity',type=int,help='Fixed max tokens; default max(BS) in continuous mode')
    p.add_argument('--port-base',type=int,required=True,help='Unique bootstrap range for this campaign')
    a=p.parse_args()
    if len(set(a.bs_values))!=len(a.bs_values) or any(bs not in (32,96,256,512,1024,2048,4096) for bs in a.bs_values):
        raise ValueError('Invalid BS set')
    if a.buffer_capacity is not None and not max(a.bs_values)<=a.buffer_capacity<=4096:
        raise ValueError('Buffer capacity must cover every BS and be <=4096')
    if not 1024<=a.port_base<=60000:
        raise ValueError('Invalid port base')
    if not re.fullmatch('/home/codex-deepep/[a-zA-Z0-9_-]+',a.remote_root):
        raise ValueError('Invalid isolated remote root')
    if a.package_cache_root and (not re.fullmatch('/home/codex-deepep/[a-zA-Z0-9_-]+',a.package_cache_root) or a.package_cache_root==a.remote_root):
        raise ValueError('Invalid distinct package cache root')
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
    manifest=dict(status='preparing',ep=a.ep,nodes=nodes,rack=a.rack,bs_values=a.bs_values,
        packages=packages,accuracy='not_run_user_requested',remote_root=a.remote_root,
        package_archive_sha256=hashlib.sha256(a.packages.read_bytes()).hexdigest(),
        schedule=measurement_schedule(a.single_start,a.method_only),
        completed=[],finished_workers=[],single_start=a.single_start,
        preparation='verify_existing' if a.run_prepared else 'deploy_only' if a.prepare_only else 'deploy_and_run',
        comparison='Historical A only; not same-window causal or repeated acceptance.' if a.single_start else 'Same-window repeats',
        scope='full communication kernel Duration; timer-off; last10/20; mean per-round rank-min',
        timer='separate rank0 full passive pass on a repeats only' if a.trace else 'off; screening only',
        timer_requested=a.trace,port_base=a.port_base,
        execution_mode='isolated' if a.isolated_bs else 'continuous',
        buffer_capacity=a.buffer_capacity or (None if a.isolated_bs else max(a.bs_values)),
        package_cache_root=a.package_cache_root,
        jump=dict(rack=a.jump_rack,node=a.jump_node) if a.jump_rack else None,
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
    clients={}
    jump_client=None
    connection_lock=threading.Lock()
    @contextmanager
    def connect(i):
        nonlocal jump_client
        # 每节点各阶段串行；复用一条已验证连接，减少频繁重新认证导致的启动失败。
        # 不在worker运行期间静默重连；连接失效仍停止本轮，最终统一关闭。
        if i not in clients:
            _,ip,user,password=inventory[nodes[i]]
            c=paramiko.SSHClient();c.load_host_keys(a.host_keys)
            c.set_missing_host_key_policy(paramiko.RejectPolicy())
            if a.additional_host_keys:
                extra=paramiko.HostKeys(a.additional_host_keys).lookup(ip)
                current=c.get_host_keys().lookup(ip)
                for kind,key in (extra or {}).items():
                    if current and kind in current and current[kind]!=key:raise ValueError('Conflicting trusted host keys')
                    c.get_host_keys().add(ip,kind,key)
            tunnel=None
            if a.jump_rack:
                with connection_lock:
                    if jump_client is None:
                        js=next(s for s in book if s.title.endswith(a.jump_rack))
                        _,jip,juser,jpassword=next(r[:4] for r in js.iter_rows(min_row=2,values_only=True) if r[0]==a.jump_node)
                        jump_client=paramiko.SSHClient();jump_client.load_host_keys(a.host_keys)
                        jump_client.set_missing_host_key_policy(paramiko.RejectPolicy())
                        jump_client.connect(jip,username=juser,password=jpassword,timeout=15,look_for_keys=False,allow_agent=False)
                        jump_client.get_transport().set_keepalive(20)
                tunnel=jump_client.get_transport().open_channel('direct-tcpip',(ip,22),('127.0.0.1',0),timeout=15)
            c.connect(ip,username=user,password=password,timeout=15,look_for_keys=False,allow_agent=False,sock=tunnel)
            c.get_transport().set_keepalive(20);clients[i]=c
        if not clients[i].get_transport().is_active():raise RuntimeError('Control connection lost')
        yield clients[i]
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
            if not a.package_cache_root:
                s.put(str(a.packages),a.remote_root+'/packages.tar.gz')
        if a.package_cache_root:
            # 缓存只减少上传；每次仍新建独立目录、验证原归档与展开后的native身份。
            cached=a.package_cache_root+'/packages.tar.gz'
            copy=(f'test $(sha256sum {cached} | cut -d" " -f1) = {manifest["package_archive_sha256"]} '
                  f'&& cp -- {cached} {a.remote_root}/packages.tar.gz')
            if command(i,copy,'cache'):raise RuntimeError('Package cache identity mismatch')
        check="import pathlib,json,hashlib;p=pathlib.Path('.');rs=json.loads((p/'packages.json').read_text());assert all(hashlib.sha256((p/'packages'/m/n).read_bytes()).hexdigest()==h for m,r in rs.items() for n,h in r['native'].items())"
        # 先确认每台节点的运行时依赖，避免到全组READY时才发现缺少共享锁目录/SHMEM。
        cmd=(f'cd {a.remote_root} && test -f /home/codex-deepep/ep32-r31-ds-1021/shmem/lib/libshmem.so '
             '&& test -f /home/pkg/b080/cann-9.2.0/set_env.sh '
             f'&& test $(sha256sum packages.tar.gz | cut -d" " -f1) = {manifest["package_archive_sha256"]} '
             f'&& tar -xzf packages.tar.gz && bash -n run_node.sh cohort_gate.sh '
             f'&& /root/miniconda3/envs/zyx/bin/python -I -c {shlex.quote(check)}')
        cmd+=' && source /home/pkg/b080/cann-9.2.0/set_env.sh'
        cmd+=' && export LD_LIBRARY_PATH="/home/codex-deepep/ep32-r31-ds-1021/shmem/lib:/root/miniconda3/envs/zyx/lib:${LD_LIBRARY_PATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0'
        for method in dict.fromkeys(job.split('-')[0] for job in manifest['schedule']):
            native_check=(f"import sys,torch,torch_npu;sys.path.insert(0,'packages/{method}');"
                "from deep_ep_experimental.megamoe.api import _extension;"
                "assert _extension().api_protocol==8;print('NATIVE_IMPORT_READY')")
            cmd+=' && /root/miniconda3/envs/zyx/bin/python -I -c '+shlex.quote(native_check)
        if command(i,cmd,'deploy'):raise RuntimeError('Deployment/native dependency check failed; see node deploy log')
    def verify_prepared(i):
        # 检测前已部署；这里只读核对包、原生库、执行脚本和运行时，失败不偷偷重部署。
        files={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
            ('benchmark.py','coordinator.py','cohort_gate.sh','export_native_trace.py','probe.py','run_node.sh')}
        check=("import pathlib,json,hashlib;p=pathlib.Path('.');"
            "rs=json.loads((p/'packages.json').read_text());"
            f"assert rs=={packages!r};"
            "assert all(hashlib.sha256((p/'packages'/m/n).read_bytes()).hexdigest()==h for m,r in rs.items() for n,h in r['native'].items());"
            f"assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {files!r}.items())")
        cmd=(f'cd {a.remote_root} && test -f /home/pkg/b080/cann-9.2.0/set_env.sh '
            '&& test -f /home/codex-deepep/ep32-r31-ds-1021/shmem/lib/libshmem.so '
            f'&& test $(sha256sum packages.tar.gz | cut -d" " -f1) = {manifest["package_archive_sha256"]} '
            f'&& /root/miniconda3/envs/zyx/bin/python -I -c {shlex.quote(check)}')
        if command(i,cmd,'verify-prepared'):raise RuntimeError('Prepared identity/runtime mismatch; no worker started')
    def collect(i,label,probe=False):
        relatives=['preflight/'+label] if probe else ([f'runs/bs{bs}-{label}' for bs in a.bs_values]
            if not a.isolated_bs else ['runs/'+label])
        archive='archive-'+label+'-'+nodes[i]+'.tar.gz'
        cmd=(f'set -o pipefail; cd {a.remote_root} && find {" ".join(relatives)} -type f \\( -name "kernel_details*.csv" -o -name "native_sparse.npz" '
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
        # preflight和每个独立启动不复用bootstrap/torchrun/control端口；不改变设备通信。
        launch_index=len(manifest['finished_workers'])
        port=a.port_base+launch_index*4
        values=[bs] if isinstance(bs,int) else bs
        capacity=a.buffer_capacity or max(values)
        csv_bs=','.join(map(str,values))
        cmd=lambda i:f'bash {a.remote_root}/run_node.sh {i} {master} {a.ep} {method} {csv_bs} {label} {trace} {port} {capacity}'
        state('starting',label=label,method=method,bs=bs,ep=a.ep,shmem_port=port,master_port=port+1,control_port=port+3)
        if any(parallel(lambda i:command(i,cmd(i),label,barrier))):raise RuntimeError('Failed node; no next case')
        manifest['finished_workers'].append(dict(label=label,trace=trace))
        save('running')
    def collect_finished():
        # 不在两个原生进程之间下载文件；避免几秒至十几秒空档吸引另一任务入卡。
        # 每次仍重新检查NPU空闲；失败时也归档此前已成功退出的调用，绝不重跑来掩盖失败。
        for item in manifest['finished_workers']:
            label,trace=item['label'],item['trace']
            case_labels=[label] if trace=='probe' or a.isolated_bs else [f'bs{bs}-{label}' for bs in a.bs_values]
            if all(case in manifest['completed'] for case in case_labels):continue
            parallel(lambda i:collect(i,label,trace=='probe'))
            if trace=='probe':
                manifest['completed'].append(label);save('collecting');continue
            for case in case_labels:
                if case in manifest['completed']:continue
                folder=out/case;folder.mkdir()
                for name in ['performance.json']+(['timer_on_performance.json'] if trace=='0' else []):
                    with connect(0) as c,c.open_sftp() as s:s.get(a.remote_root+'/runs/'+case+'/'+name,str(folder/name))
                result=json.loads((folder/'performance.json').read_text())
                state('run_complete',label=case,worker_session=label,mean_us=result['mean_us'],rank_max_mean_us=result['rank_max_mean_us'])
                manifest['completed'].append(case);save('collecting')
    save('preparing')
    try:
        parallel(verify_prepared if a.run_prepared else deploy)
        if a.prepare_only:
            save('prepared');state('prepared_only');return
        # 单轮B直接进入benchmark原有设备初始化、SHMEM初始化和预热，不多启动一套probe。
        if not a.single_start:launch(a.method_only or 'combined',32,'preflight','probe')
        if a.isolated_bs:
            for bs in manifest['bs_values']:
                for job in manifest['schedule']:
                    method,repeat=job.split('-')
                    launch(method,bs,f'bs{bs}-{job}','0' if a.trace and repeat=='a' else 'none')
        else:
            for job in manifest['schedule']:
                method,repeat=job.split('-')
                launch(method,a.bs_values,job,'0' if a.trace and repeat=='a' else 'none')
        collect_finished()
        save('collected_not_yet_audited')
    except Exception as error:
        save('failed');state('phase_failed',error_type=type(error).__name__)
        try:
            collect_finished()
        except Exception as collect_error:
            state('evidence_collect_failed',error_type=type(collect_error).__name__)
        save('failed')
        raise
    finally:
        for client in clients.values():client.close()
        if jump_client:jump_client.close()


if __name__=='__main__':main()
