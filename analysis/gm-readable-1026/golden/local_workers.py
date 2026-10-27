"""Small test launcher: real child exits, bounded failure cleanup, no elastic Store/barrier."""
import argparse
import os
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--node',type=int,required=True);p.add_argument('--master',required=True)
p.add_argument('--ep',type=int,choices=(8,32),required=True);p.add_argument('--port',type=int,required=True)
p.add_argument('worker');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
if not 0<=a.node<a.ep//8:raise ValueError('Invalid node index')
children=[]
def interrupted(signum,frame):raise KeyboardInterrupt(f'launcher signal {signum}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    for local in range(8):
        env=dict(os.environ,RANK=str(a.node*8+local),LOCAL_RANK=str(local),WORLD_SIZE=str(a.ep),
            LOCAL_WORLD_SIZE='8',MASTER_ADDR=a.master,MASTER_PORT=str(a.port))
        env.pop('TORCHELASTIC_USE_AGENT_STORE',None)
        children.append(subprocess.Popen([sys.executable,'-I','-u',a.worker,*a.arguments],env=env,start_new_session=True))
    while True:
        codes=[child.poll() for child in children]
        if any(code is not None and code!=0 for code in codes):raise RuntimeError(f'Worker exit codes: {codes}')
        if all(code==0 for code in codes):break
        time.sleep(0.1)
finally:
    # 只处理本启动器创建且仍存活的进程组，不枚举/终止外部任务。
    for child in children:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+10
    for child in children:
        if child.poll() is None:
            try:child.wait(timeout=max(0.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                child.wait()
