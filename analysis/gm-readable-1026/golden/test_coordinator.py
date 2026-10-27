from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import secrets
import socket


def test_actual_tcp_group(monkeypatch):
    spec=importlib.util.spec_from_file_location('golden_coordinator',Path(__file__).with_name('coordinator.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    monkeypatch.setenv('MASTER_ADDR','127.0.0.1');monkeypatch.setenv('MASTER_PORT',str(port-2))
    monkeypatch.setenv('MEGAMOE_CONTROL_TOKEN',secrets.token_hex(32))
    class Api:
        class StoreGroup:pass
    def worker(rank):
        group=module.open_control_group(Api,rank,8,'cpu-test')
        for i in range(8):
            values=group.all_gather(dict(rank=rank,iteration=i),f'round/{i}')
            assert values==[dict(rank=r,iteration=i) for r in range(8)]
            group.barrier(f'round/{i}/checked')
        group.close()
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
