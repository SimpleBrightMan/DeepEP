"""Accuracy-only authenticated stdlib RPC; no Torch distributed collective or device transport."""
import os
from multiprocessing.managers import BaseManager
import threading
import time


CONTROL_KIND='test_manager_tcp'


class Coordinator:
    def __init__(self,world):
        self.world=world;self.condition=threading.Condition();self.rounds={};self.sequences=[0]*world
        self.error=None;self.closed=set()

    def abort(self,message):
        with self.condition:
            self.error=str(message)[:4096];self.condition.notify_all()

    def gather(self,rank,sequence,stage,value):
        with self.condition:
            try:
                if not 0<=rank<self.world or sequence!=self.sequences[rank]:raise ValueError('rank/sequence mismatch')
                self.sequences[rank]+=1
                item=self.rounds.setdefault(sequence,dict(stage=stage,values={},readers=0))
                if item['stage']!=stage or rank in item['values']:raise ValueError('collective stage/duplicate rank mismatch')
                item['values'][rank]=value;self.condition.notify_all()
                deadline=time.monotonic()+120
                while len(item['values'])!=self.world and self.error is None:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('host collective missing ranks')
                    self.condition.wait(remaining)
                if self.error is not None:raise RuntimeError(self.error)
                result=[item['values'][r] for r in range(self.world)]
                item['readers']+=1
                if item['readers']==self.world:del self.rounds[sequence]
                return result
            except Exception as error:
                self.error=str(error);self.condition.notify_all();raise

    def mark_closed(self,rank):
        with self.condition:self.closed.add(rank);self.condition.notify_all()

    def wait_closed(self):
        with self.condition:
            if not self.condition.wait_for(lambda:len(self.closed)==self.world-1,timeout=120):
                raise TimeoutError('host close acknowledgements missing')


def open_control_group(api,rank,world,label):
    auth=bytes.fromhex(os.environ['MEGAMOE_CONTROL_TOKEN'])
    if len(auth)!=32:raise ValueError('Missing one-run host-control authentication')
    class Manager(BaseManager):pass
    state=Coordinator(world) if rank==0 else None
    Manager.register('coordinator',callable=lambda:state,
        exposed=('gather','abort','mark_closed','wait_closed'))
    address=(os.environ['MASTER_ADDR'],int(os.environ['MASTER_PORT'])+2)
    server=None
    if rank==0:
        server=Manager(address=address,authkey=auth).get_server()
        threading.Thread(target=server.serve_forever,daemon=True).start()
    manager=Manager(address=address,authkey=auth);deadline=time.monotonic()+120
    while True:
        try:manager.connect();break
        except (ConnectionRefusedError,OSError):
            if time.monotonic()>=deadline:raise
            time.sleep(0.1)
    remote=manager.coordinator()
    class Group(api.StoreGroup):
        def __init__(self):self._rank,self._size,self._sequence=rank,world,0
        def all_gather(self,value,stage):
            sequence=self._sequence;self._sequence+=1
            return remote.gather(rank,sequence,stage,value)
        def barrier(self,stage='barrier'):self.all_gather(None,'barrier:'+stage)
        def abort(self,message):remote.abort(message)
        def close(self):
            self.barrier('close')
            if rank!=0:remote.mark_closed(rank)
            else:
                remote.wait_closed()
                # CPU-only teardown grace; not part of any kernel/performance measurement.
                time.sleep(1)
            # Server只属于本次rank0进程；避免解释器退出时再发decref RPC，晚退出的peer可能等已退出server。
            # 数值/计数检查和全体关闭确认均已完成，不省略任何设备同步或失败检查。
            remote._close.cancel()
            connection=getattr(remote._tls,'connection',None)
            if connection is not None:connection.close()
    group=Group()
    group._test_manager=manager;group._test_server=server
    group.barrier('control_connect')
    print(f'HOST_CONTROL_READY rank={rank}/{world} backend={CONTROL_KIND}',flush=True)
    return group
