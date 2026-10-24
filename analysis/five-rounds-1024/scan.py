"""只读复扫授权柜；凭据仅在内存，未知主机密钥拒绝，结果不含地址/凭据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True)
    p.add_argument('--host-keys',required=True)
    p.add_argument('--dependencies',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--racks',nargs='+',choices=tuple(f'{n:02d}' for n in range(1,9)),default=['01','05','06','07'])
    p.add_argument('--jump-rack',choices=tuple(f'{n:02d}' for n in range(1,9)))
    p.add_argument('--jump-node',default='CPU1')
    p.add_argument('--additional-host-keys',help='Public known_hosts copied from an authenticated trusted host')
    a=p.parse_args()
    sys.path.insert(0,a.dependencies)
    import openpyxl
    import paramiko
    book=openpyxl.load_workbook(a.inventory,read_only=True,data_only=True)
    nodes=[(s.title[-2:],*r[:4]) for s in book if s.title[-2:] in a.racks
           for r in s.iter_rows(min_row=2,values_only=True) if r[0] and r[1]]
    jump=None
    if a.jump_rack:
        sheet=next(s for s in book if s.title.endswith(a.jump_rack))
        _,address,user,password=next(r[:4] for r in sheet.iter_rows(min_row=2,values_only=True) if r[0]==a.jump_node)
        jump=paramiko.SSHClient();jump.load_host_keys(a.host_keys)
        jump.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            jump.connect(address,username=user,password=password,timeout=12,auth_timeout=12,
                banner_timeout=12,look_for_keys=False,allow_agent=False)
        except Exception as error:
            jump.close()
            raise RuntimeError('Jump connection failed: '+type(error).__name__) from None
    command=r'''date -Is
npu-smi info | grep -c 'No running processes found in NPU'
ps -eo pid=,comm= | awk '$2 ~ /^(python[0-9.]*)$/{print $1}' | while read -r p; do
 if grep -qE '/(libtorch_npu|libascendcl)[^/]*\.so' "/proc/$p/maps" 2>/dev/null; then echo "NPU_CONTEXT $p"; fi
done
for item in 'ZYX /root/miniconda3/envs/zyx/bin/python' 'HKQ /root/miniconda3/envs/hkq/bin/python' 'CANN /home/pkg/b080/cann-9.2.0/set_env.sh' 'SHMEM /home/codex-deepep/ep32-r31-ds-1021/shmem/lib/libshmem.so'; do
 read -r name path <<< "$item"
 if test -f "$path"; then echo "RUNTIME_$name=1"; else echo "RUNTIME_$name=0"; fi
done
if test -f /home/codex-deepep/ep32-r31-ds-1021/launch.lock; then
 if (flock -n -s 9) 9< /home/codex-deepep/ep32-r31-ds-1021/launch.lock; then echo SLOT_AVAILABLE=1; else echo SLOT_AVAILABLE=0; fi
else echo SLOT_AVAILABLE=1
fi
'''
    def scan(node):
        rack,name,address,user,password=node
        result=dict(rack=rack,node=name,time_utc=datetime.now(timezone.utc).isoformat())
        if jump:result['via']=dict(rack=a.jump_rack,node=a.jump_node)
        client=paramiko.SSHClient();client.load_host_keys(a.host_keys)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            # 只补目标主机已保存的公钥；已知密钥冲突必须失败，绝不覆盖或接受新抓取密钥。
            if a.additional_host_keys:
                extra=paramiko.HostKeys(a.additional_host_keys).lookup(address)
                current=client.get_host_keys().lookup(address)
                for kind,key in (extra or {}).items():
                    if current and kind in current and current[kind]!=key:
                        raise ValueError('Conflicting trusted host keys')
                    client.get_host_keys().add(address,kind,key)
            tunnel=jump.get_transport().open_channel('direct-tcpip',(address,22),('127.0.0.1',0),timeout=12) if jump else None
            client.connect(address,username=user,password=password,timeout=12,auth_timeout=12,
                           banner_timeout=12,look_for_keys=False,allow_agent=False,sock=tunnel)
            _,out,err=client.exec_command(command,timeout=20)
            lines=out.read().decode(errors='replace').splitlines();error=err.read().decode(errors='replace')
            code=out.channel.recv_exit_status()
            if code or len(lines)<2 or not lines[1].isdigit():
                raise RuntimeError('Incomplete idle probe')
            result.update(idle_cards=int(lines[1]),npu_contexts=len([s for s in lines if s.startswith('NPU_CONTEXT')]))
            result['slot_available']='SLOT_AVAILABLE=1' in lines
            result['available']=result['idle_cards']==8 and result['npu_contexts']==0 and result['slot_available']
            result['runtime']={s.split('=',1)[0][8:].lower():s.endswith('=1') for s in lines if s.startswith('RUNTIME_')}
        except Exception as error:
            result.update(available=False,error=type(error).__name__+': '+str(error).replace(str(password),'[redacted]'))
        finally:
            client.close()
            if 'tunnel' in locals() and tunnel:tunnel.close()
        return result
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with a.output.open('x',encoding='utf-8') as saved,ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(scan,nodes):
                line=json.dumps(result);print(line,flush=True);saved.write(line+'\n');saved.flush()
    finally:
        if jump:jump.close()


if __name__=='__main__':main()
