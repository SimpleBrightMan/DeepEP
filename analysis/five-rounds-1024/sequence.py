"""在同一节点组顺序补R2–R5；每个case仍重新过空闲门禁，失败即停。"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True)
    p.add_argument('--host-keys',required=True)
    p.add_argument('--dependencies',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rack',required=True)
    p.add_argument('--nodes',required=True)
    p.add_argument('--placement-tag',required=True)
    a=p.parse_args()
    scripts=Path(__file__).resolve().parent
    for round_id,port in ((2,27120),(3,27320),(4,27520),(5,27720)):
        output=a.root/f'r{round_id}-ep32-{a.placement_tag}'
        command=[sys.executable,str(scripts/'campaign.py'),'--inventory',a.inventory,'--host-keys',a.host_keys,
            '--dependencies',a.dependencies,'--packages',str(a.root/f'r{round_id}-packages.tar.gz'),
            '--output',str(output),'--rack',a.rack,'--nodes',a.nodes,'--ep','32','--bs-values','4096',
            '--remote-root',f'/home/codex-deepep/five-rounds-r{round_id}-ep32-1024-seq-a',
            '--port-base',str(port)]
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,str(scripts/'report.py'),str(output)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable,str(scripts/'summarize.py'),str(a.root)],check=True)


if __name__=='__main__':main()
