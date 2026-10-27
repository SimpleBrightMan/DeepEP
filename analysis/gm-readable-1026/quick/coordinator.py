"""隔离测试的CPU Gloo控制层；设备Dispatch/Combine仍为SHMEM/MTE。"""
from datetime import timedelta
import os


def open_control_group(api, rank, world, label):
    import torch.distributed as dist

    # 连续profiler下的Store RPC有空包/超时历史；仅测试控制改用CPU Gloo。
    # 不初始化HCCL，不改变或替代设备通信。两版必须使用同一控制实现。
    port=int(os.environ['MASTER_PORT'])+2
    if not 1024<=port<=65535: raise ValueError('Invalid control-store port')
    if not dist.is_gloo_available():raise RuntimeError('CPU Gloo unavailable; do not fall back silently')
    os.environ['GLOO_SOCKET_IFNAME']='enp35s0f2'
    # torchrun可设置TORCHELASTIC_USE_AGENT_STORE；自定义端口不能走其env rendezvous，
    # 否则所有rank都当客户端，没人监听。显式建立仅用于Gloo初始化的Store。
    store=dist.TCPStore(os.environ['MASTER_ADDR'],port,world,rank==0,timedelta(seconds=120),
        wait_for_workers=False,use_libuv=True)
    dist.init_process_group('gloo',store=store,rank=rank,world_size=world,timeout=timedelta(seconds=120))
    pg=dist.distributed_c10d._get_default_group()
    class CpuControlGroup(api.StoreGroup):
        # 保持生产API识别的StoreGroup类型；只覆盖测试需要的CPU集体操作。
        def __init__(self):
            self._rank,self._size,self._sequence=rank,world,0
            self._control_store=store
        def all_gather(self,value,stage):
            packet=(self._sequence,stage,value)
            peers=[None]*world
            dist.all_gather_object(peers,packet,group=pg)
            if any(p[:2]!=packet[:2] for p in peers):
                raise RuntimeError(f'CPU control collective order mismatch: rank={rank} stage={stage}')
            self._sequence+=1
            return [p[2] for p in peers]
        def barrier(self,stage='barrier'):
            self.all_gather(None,'barrier:'+stage)
        def abort(self,error):
            print(f'CPU_CONTROL_ABORT rank={rank}: {error}',flush=True)
            if dist.is_initialized():dist.destroy_process_group(pg)
        def close(self):
            self.barrier('close')
            dist.destroy_process_group(pg)
            self._control_store=None
    group=CpuControlGroup()
    group.barrier('control_connect')
    print(f'HOST_CONTROL_READY rank={rank}/{world} backend=cpu_gloo',flush=True)
    return group
