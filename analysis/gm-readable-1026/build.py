"""隔离构建并封装对照；不安装到共享Python，不覆盖既有源码/构建目录。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dependencies', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--baseline-revision', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--source-sha256', required=True)
    a = p.parse_args()
    root, deps, baseline = a.root.resolve(), a.dependencies.resolve(), a.baseline.resolve()
    archive = root / 'source.tar'
    if sha(archive) != a.source_sha256:
        raise ValueError('Source archive identity mismatch')
    prior = json.loads((baseline / 'packages.json').read_text())['combined']
    if prior['revision'] != a.baseline_revision:
        raise ValueError('Unexpected baseline source identity')
    for relative, expected in prior['native'].items():
        if sha(baseline / 'packages/combined' / relative) != expected:
            raise ValueError('Baseline native identity mismatch: ' + relative)
    source = root / 'source'
    source.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        tar.extractall(source, filter='data')
    tests = source / 'experiments/megamoe/tests'
    names = ['test_device_schedule.py', 'test_public_api.py', 'test_paired_wave_budget_cpu.py',
             'test_paired_launch_cpu.py', 'test_paired_supply_cpu.py', 'test_gmm2_aiv_completion.py',
             'test_aiv1_fair_scheduler.py', 'test_workspace_layout.py', 'test_wave_gm_policy.py',
             'test_aiv1_resident_ub.py', 'test_packed_weight_tiles.py']
    if any(not (tests / name).is_file() for name in names):
        raise ValueError('Required CPU test missing')
    with (root / 'cpu-contract.log').open('x') as log:
        subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        *[str(tests / name) for name in names]],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    print('CPU_CONTRACT_PASS', flush=True)
    packages = root / 'packages'
    packages.mkdir()
    for method in ('control', 'combined'):
        shutil.copytree(baseline / 'packages/combined', packages / method)
    package = packages / 'combined'
    shutil.copytree(source / 'experiments/megamoe/python/deep_ep_experimental',
                    package / 'deep_ep_experimental', dirs_exist_ok=True)
    cmake = str(Path(sys.executable).with_name('cmake'))
    pybind = subprocess.check_output([sys.executable, '-m', 'pybind11', '--cmakedir'], text=True).strip()
    build = root / 'build'
    command = [cmake, '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
               '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
               '-DDEEPEP_BUILD_MEGAMOE=ON', '-DDEEPEP_MEGAMOE_TIMER=ON', '-DDEEPEP_DEVICE_JOBS=4',
               '-DASCEND_HOME_PATH=/home/pkg/b080/cann-9.2.0', '-DSHMEM_ROOT=' + str(deps / 'shmem'),
               '-DCATLASS_ROOT=' + str(deps / 'catlass'), '-DPython_EXECUTABLE=' + sys.executable,
               '-Dpybind11_DIR=' + pybind]
    (root / 'configure.json').write_text(json.dumps(command, indent=2))
    print('NATIVE_BUILD_START', flush=True)
    with (root / 'build.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        subprocess.run([cmake, '--build', str(build), '--target', 'megamoe_ext', '--parallel', '2'],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    native = package / 'deep_ep_experimental/megamoe'
    # 删除范围仅为本脚本刚创建的包副本；旧冻结包从不写入。
    if native.resolve().parent != (package / 'deep_ep_experimental').resolve():
        raise ValueError('Unexpected package target')
    for item in native.glob('_C*.so'):
        item.unlink()
    extensions = list((build / 'experiments/megamoe').glob('_C*.so'))
    if len(extensions) != 1:
        raise ValueError('Ambiguous host extension')
    for item in [extensions[0], build / 'experiments/megamoe/libmegamoe_kernel.so']:
        shutil.copy2(item, native / item.name)
    registry = {}
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = '/home/codex-deepep/ep32-r31-ds-1021/shmem/lib:' + env.get('LD_LIBRARY_PATH', '')
    for method in ('control', 'combined'):
        pkg = packages / method
        probe = ('import sys,json,torch,torch_npu;sys.path.insert(0,' + repr(str(pkg)) + ');'
                 'from deep_ep_experimental.megamoe import api;from deep_ep_experimental.megamoe.timer import protocol as p;'
                 'e=api._extension();print(json.dumps(dict(protocol=e.api_protocol,groups=p.DYNAMIC_TRACE_GROUPS,'
                 'aiv0=p.DYNAMIC_TRACE_AIV0,timer_numel=e.timer_numel,expected_numel=p.TOTAL_BUFFER_SIZE)))')
        policy = json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', probe], text=True, env=env))
        if policy['protocol'] != 8 or policy['groups'] != 32 or policy['aiv0'] != 1 or policy['timer_numel'] != policy['expected_numel']:
            raise ValueError('Native/parser timer ABI mismatch')
        registry[method] = dict(revision=a.revision if method == 'combined' else prior['revision'],
            protocol=8, archive_sha256=a.source_sha256 if method == 'combined' else prior['archive_sha256'],
            native={str(f.relative_to(pkg)): sha(f) for f in (pkg / 'deep_ep_experimental/megamoe').glob('*.so')},
            trace_policy=policy, trace_only_changes={})
    if registry['control']['native'] != prior['native']:
        raise ValueError('Frozen control changed')
    (root / 'packages.json').write_text(json.dumps(registry, indent=2))
    with tarfile.open(root / 'packages.tar.gz', 'w:gz') as tar:
        tar.add(packages, arcname='packages')
        tar.add(root / 'packages.json', arcname='packages.json')
    print('PACKAGES_READY', sha(root / 'packages.tar.gz'), flush=True)


if __name__ == '__main__':
    main()
