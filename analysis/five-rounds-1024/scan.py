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
    a=p.parse_args()
    sys.path.insert(0,a.dependencies)
    import openpyxl
    import paramiko
    book=openpyxl.load_workbook(a.inventory,read_only=True,data_only=True)
    nodes=[(s.title[-2:],*r[:4]) for s in book if s.title[-2:] in ('01','05','06','07')
           for r in s.iter_rows(min_row=2,values_only=True) if r[0] and r[1]]
    command=r'''date -Is
npu-smi info | grep -c 'No running processes found in NPU'
ps -eo pid=,comm= | awk '$2 ~ /^(python[0-9.]*)$/{print $1}' | while read -r p; do
 if grep -qE '/(libtorch_npu|libascendcl)[^/]*\.so' "/proc/$p/maps" 2>/dev/null; then echo "NPU_CONTEXT $p"; fi
done
'''
    def scan(node):
        rack,name,address,user,password=node
        result=dict(rack=rack,node=name,time_utc=datetime.now(timezone.utc).isoformat())
        client=paramiko.SSHClient();client.load_host_keys(a.host_keys)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(address,username=user,password=password,timeout=12,auth_timeout=12,
                           banner_timeout=12,look_for_keys=False,allow_agent=False)
            _,out,err=client.exec_command(command,timeout=20)
            lines=out.read().decode(errors='replace').splitlines();error=err.read().decode(errors='replace')
            code=out.channel.recv_exit_status()
            if code or len(lines)<2 or not lines[1].isdigit():
                raise RuntimeError('Incomplete idle probe')
            result.update(idle_cards=int(lines[1]),npu_contexts=len([s for s in lines if s.startswith('NPU_CONTEXT')]))
            result['available']=result['idle_cards']==8 and result['npu_contexts']==0
        except Exception as error:
            result.update(available=False,error=type(error).__name__+': '+str(error).replace(str(password),'[redacted]'))
        finally:
            client.close()
        return result
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as saved,ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(scan,nodes):
            line=json.dumps(result);print(line,flush=True);saved.write(line+'\n');saved.flush()


if __name__=='__main__':main()
