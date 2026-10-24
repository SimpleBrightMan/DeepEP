"""隔离构建两轮wave+AIV1完成等待；控制包复用已冻结两轮wave版本。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--archive-sha256', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    archive = root/'source.tar'
    if sha(archive) != a.archive_sha256:
        raise ValueError('Source archive mismatch')
    source = root/'source'
    source.mkdir(exist_ok=False)
    with tarfile.open(archive) as t:
        t.extractall(source, filter='data')
    tests = source/'experiments/megamoe/tests'
    checks = ['test_paired_wave_budget_cpu.py', 'test_gmm2_aiv_completion.py',
              'test_aiv1_fair_scheduler.py', 'test_workspace_layout.py']
    with (root/'cpu-contract.log').open('x') as log:
        subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *[str(tests/n) for n in checks]],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    print('CPU_CONTRACT_PASS', flush=True)
    old = Path('/home/codex-deepep/r31-two-round-build-1023')
    prior = json.loads((old/'packages.json').read_text())['paired2']
    packages = root/'packages'
    packages.mkdir()
    for method in ('control', 'combined'):
        shutil.copytree(old/'packages/paired2', packages/method)
    package = packages/'combined'
    shutil.copytree(source/'experiments/megamoe/python/deep_ep_experimental',
                    package/'deep_ep_experimental', dirs_exist_ok=True)
    catlass = old/'catlass'
    cmake = str(Path(sys.executable).with_name('cmake'))
    pybind = subprocess.check_output([sys.executable, '-m', 'pybind11', '--cmakedir'], text=True).strip()
    build = root/'build'
    command = [cmake, '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
               '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
               '-DDEEPEP_BUILD_MEGAMOE=ON', '-DDEEPEP_MEGAMOE_TIMER=ON',
               '-DASCEND_HOME_PATH=/home/pkg/b080/cann-9.2.0',
               '-DSHMEM_ROOT=/home/codex-deepep/20261012-d8-sdk/DeepEP/third-party/shmem/install/shmem',
               '-DCATLASS_ROOT='+str(catlass), '-DPython_EXECUTABLE='+sys.executable, '-Dpybind11_DIR='+pybind]
    (root/'configure.json').write_text(json.dumps(command, indent=2))
    print('NATIVE_BUILD_START', flush=True)
    with (root/'build.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        subprocess.run([cmake, '--build', str(build), '--target', 'megamoe_ext', '--parallel', '2'],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    native = package/'deep_ep_experimental/megamoe'
    for item in native.glob('_C*.so'):
        item.unlink()  # 仅覆盖本次新建的隔离包副本。
    extensions = list((build/'experiments/megamoe').glob('_C*.so'))
    if len(extensions) != 1:
        raise ValueError('Ambiguous host extension')
    for item in [extensions[0], build/'experiments/megamoe/libmegamoe_kernel.so']:
        shutil.copy2(item, native/item.name)
    registry = {}
    for method in ('control', 'combined'):
        pkg = packages/method
        probe = ('import sys,json,torch,torch_npu;sys.path.insert(0,'+repr(str(pkg))+');'
                 'from deep_ep_experimental.megamoe import api;from deep_ep_experimental.megamoe.timer import protocol as p;'
                 'e=api._extension();print(json.dumps(dict(protocol=e.api_protocol,groups=p.DYNAMIC_TRACE_GROUPS,'
                 'aiv0=p.DYNAMIC_TRACE_AIV0,timer_numel=e.timer_numel,expected_numel=p.TOTAL_BUFFER_SIZE)))')
        policy = json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', probe], text=True))
        if policy['protocol'] != 8 or policy['groups'] != 32 or policy['aiv0'] != 1 or policy['timer_numel'] != policy['expected_numel']:
            raise ValueError('Wrong native ABI/timer contract')
        registry[method] = dict(revision=a.revision if method=='combined' else prior['revision'],
            protocol=8, wave_m_groups_paired=3, archive_sha256=a.archive_sha256 if method=='combined' else prior['archive_sha256'],
            native={str(f.relative_to(pkg)):sha(f) for f in (pkg/'deep_ep_experimental/megamoe').glob('*.so')},
            trace_policy=policy, trace_only_changes={})
    for name, expected in prior['native'].items():
        if registry['control']['native'][name] != expected:
            raise ValueError('Frozen control package changed')
    (root/'packages.json').write_text(json.dumps(registry, indent=2))
    with tarfile.open(root/'packages.tar.gz', 'w:gz') as t:
        t.add(packages, arcname='packages')
        t.add(root/'packages.json', arcname='packages.json')
    print('PACKAGES_READY', sha(root/'packages.tar.gz'), flush=True)


if __name__=='__main__':
    main()
