"""冻结R31 ABBA与采集合同，复用已验证的四节点门禁/采集器；失败即停。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--packages', type=Path, required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--host-keys', required=True)
    p.add_argument('--dependencies', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--nodes', default='CPU7,CPU3,CPU4,CPU1')
    p.add_argument('--rack', default='05', choices=('01','05','06','07'))
    p.add_argument('--remote-root', required=True)
    a = p.parse_args()
    with tarfile.open(a.packages) as bundle:
        registry = json.loads(bundle.extractfile('packages.json').read())
    if set(registry) != {'paired', 'paired2'} or any(r['protocol'] != 8 for r in registry.values()):
        raise ValueError('Expected the two matching R31 protocol8 packages')
    name = 'deep_ep_experimental/megamoe/libmegamoe_kernel.so'
    if registry['paired']['native'][name] != registry['paired2']['native'][name]:
        raise ValueError('Only host policy may differ')
    if [registry[m]['wave_m_groups'] for m in ('paired','paired2')] != [6,3]:
        raise ValueError('Wrong wave policy')
    a.output.mkdir(parents=True, exist_ok=False)
    jobs = 'paired-a,paired2-a,paired2-b,paired-b'
    manifest = dict(status='preparing', ep=32, bs=4096, jobs=jobs, nodes=a.nodes.split(','), rack=a.rack,
        methods={'paired':'R31 control', 'paired2':'R31 two-round wave'}, packages=registry,
        package_archive_sha256=hashlib.sha256(a.packages.read_bytes()).hexdigest(),
        harness_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.harness.iterdir() if f.is_file()},
        accuracy='not_run_user_requested',timer='off performance + separate rank0 full passive on pass',
        remote_root=a.remote_root)
    def save(status, **fields):
        manifest.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        (a.output/'experiment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(dict(status=status, **fields)), flush=True)
    common = ['--inventory',a.inventory,'--host-keys',a.host_keys,'--dependencies',a.dependencies,
        '--packages',str(a.packages),'--output',str(a.output),'--nodes',a.nodes,'--rack',a.rack,
        '--remote-root',a.remote_root]
    phases = [('deploy', []), ('probe', ['--probe-label','r31-wave-a','--probe-method','paired']),
        ('trace', ['--preflight-label','r31-wave-a','--bs','4096','--jobs',jobs,
                   '--trace-packages',str(a.packages),'--trace-package-dir','native-r31-waves-v1','--trace-ranks','0'])]
    save('preparing')
    for phase, extra in phases:
        save(phase)
        with (a.output/('controller-'+phase+'.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable,str(a.harness/'campaign.py'),phase,*common,*extra],
                                    stdout=log,stderr=subprocess.STDOUT,timeout=4500)
        if result.returncode:
            save('failed',failed_phase=phase,exit_code=result.returncode)
            raise SystemExit(result.returncode)
    save('collected_not_yet_audited')


if __name__ == '__main__':
    main()
