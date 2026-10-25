"""CPU Gloo测试：不导入NPU扩展、不创建设备上下文，模拟profiler的fork。"""
import argparse
import importlib.util
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace


def pause():
    time.sleep(0.003)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds',type=int,default=64)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    store=load('host_store',root/'deep_ep/_store.py')
    controller=load('host_control',Path(__file__).resolve().parents[1]/'five-rounds-1024/coordinator.py')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    group=controller.open_control_group(SimpleNamespace(StoreGroup=store.StoreGroup),rank,world,'cpu-stress')
    for step in range(a.rounds):
        if step%4==0:
            child=multiprocessing.get_context('fork').Process(target=pause)
            child.start();child.join(10)
            if child.exitcode!=0:raise RuntimeError('Fork child failed')
        peers=group.all_gather((rank,step),f'round{step}')
        assert peers==[(r,step) for r in range(world)]
        group.barrier(f'round{step}/done')
    group.close()
    print(f'CPU_CONTROL_PASS rank={rank} rounds={a.rounds}',flush=True)


if __name__=='__main__':main()
