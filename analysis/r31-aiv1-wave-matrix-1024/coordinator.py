"""隔离启动器Store的主机协调；不替代任何设备通信或计时内工作。"""
from datetime import timedelta
import os


def open_control_group(api, rank, world, label):
    from torch.distributed import TCPStore

    # Store由rank0持有到全体close之后；不同case使用新进程和独立namespace。
    # +2与torchrun端口及SHMEM端口均不同，实际端口记录在placement中。
    port=int(os.environ['MASTER_PORT'])+2
    if not 1024<=port<=65535: raise ValueError('Invalid control-store port')
    store=TCPStore(os.environ['MASTER_ADDR'],port,None,rank==0,timedelta(seconds=300),
                   wait_for_workers=False,use_libuv=False)
    group=api.StoreGroup(store,rank,world,prefix='ep32-regression/'+label,timeout=300)
    group.barrier('control_connect')
    print(f'HOST_CONTROL_READY rank={rank}/{world}',flush=True)
    return group
