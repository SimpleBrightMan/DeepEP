"""从冻结合并包隔离构建单变量候选；不安装到共享环境。"""
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
    p.add_argument('--baseline', type=Path, required=True, help='Frozen 4a0de857 build root; read-only')
    p.add_argument('--revision', required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--host-only', action='store_true')
    p.add_argument('--resume-build', action='store_true', help='Preserve failed log; verify source before build-only retry')
    a = p.parse_args()
    root, deps = a.root.resolve(), a.dependencies.resolve()
    archive = root/'source.tar'
    if sha(archive) != a.source_sha256:
        raise ValueError('Source archive identity mismatch')
    source = root/'source'
    if a.resume_build:
        if (root/'packages.json').exists():
            raise ValueError('Already packaged; do not overwrite')
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.isfile() and sha(source/member.name) != hashlib.sha256(tar.extractfile(member).read()).hexdigest():
                    raise ValueError('Frozen source changed: '+member.name)
    else:
        source.mkdir(exist_ok=False)
        with tarfile.open(archive) as tar:
            tar.extractall(source, filter='data')
    tests = source/'experiments/megamoe/tests'
    checks = ['test_device_schedule.py', 'test_paired_wave_budget_cpu.py', 'test_gmm2_aiv_completion.py',
              'test_aiv1_fair_scheduler.py', 'test_workspace_layout.py', 'test_wave_gm_policy.py',
              'test_aiv1_resident_ub.py', 'test_gmm1_cv_plan.py']
    if (tests/'test_combine_quota_cpu.py').exists():
        checks.append('test_combine_quota_cpu.py')
    with (root/('cpu-contract-resume.log' if a.resume_build else 'cpu-contract.log')).open('x') as log:
        subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *[str(tests/n) for n in checks]],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    print('CPU_CONTRACT_PASS', flush=True)
    baseline = a.baseline.resolve()
    prior = json.loads((baseline/'packages.json').read_text())['combined']
    if prior['native']['deep_ep_experimental/megamoe/libmegamoe_kernel.so'] != (
        '9525043992ad15a446840bdca3b3ca333a9de2c76b31b41623fe6f8f0199ce94'):
        raise ValueError('Not the frozen current merged baseline')
    packages = root/'packages'
    if not a.resume_build:
        packages.mkdir()
        for method in ('control', 'combined'):
            shutil.copytree(baseline/'packages/combined', packages/method)
    package = packages/'combined'
    shutil.copytree(source/'experiments/megamoe/python/deep_ep_experimental',
                    package/'deep_ep_experimental', dirs_exist_ok=True)
    if a.host_only:
        # 不凭开关假定设备相同：逐文件比较冻结源码后，才允许复用设备二进制。
        with tarfile.open(deps/'baseline-source.tar') as tar:
            for member in tar:
                if member.isfile() and member.name.startswith('experiments/megamoe/csrc/kernels/'):
                    if sha(source/member.name) != hashlib.sha256(tar.extractfile(member).read()).hexdigest():
                        raise ValueError('Host-only candidate changed device source: '+member.name)
    cmake = str(Path(sys.executable).with_name('cmake'))
    pybind = subprocess.check_output([sys.executable, '-m', 'pybind11', '--cmakedir'], text=True).strip()
    build = root/'build'
    command = [cmake, '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
               '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
               '-DDEEPEP_BUILD_MEGAMOE=ON', '-DDEEPEP_MEGAMOE_TIMER=ON',
               '-DASCEND_HOME_PATH=/home/pkg/b080/cann-9.2.0', '-DSHMEM_ROOT='+str(deps/'shmem'),
               '-DCATLASS_ROOT='+str(deps/'catlass'), '-DPython_EXECUTABLE='+sys.executable,
               '-Dpybind11_DIR='+pybind]
    (root/'configure.json').write_text(json.dumps(command, indent=2))
    print('NATIVE_BUILD_START', flush=True)
    with (root/('build-resume.log' if a.resume_build else 'build.log')).open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        if a.host_only:
            shutil.copy2(package/'deep_ep_experimental/megamoe/libmegamoe_kernel.so',
                         build/'experiments/megamoe/libmegamoe_kernel.so')
            os.utime(build/'experiments/megamoe/libmegamoe_kernel.so', None)
        subprocess.run([cmake, '--build', str(build), '--target', 'megamoe_ext', '--parallel', '2'],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    native = package/'deep_ep_experimental/megamoe'
    for item in native.glob('_C*.so'):
        item.unlink()  # 仅本次新建的包副本；原冻结包不动。
    extensions = list((build/'experiments/megamoe').glob('_C*.so'))
    if len(extensions) != 1:
        raise ValueError('Ambiguous host extension')
    for item in [extensions[0], build/'experiments/megamoe/libmegamoe_kernel.so']:
        shutil.copy2(item, native/item.name)
    registry = {}
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = '/home/codex-deepep/ep32-r31-ds-1021/shmem/lib:'+env.get('LD_LIBRARY_PATH', '')
    for method in ('control', 'combined'):
        pkg = packages/method
        probe = ('import sys,json,torch,torch_npu;sys.path.insert(0,'+repr(str(pkg))+');'
                 'from deep_ep_experimental.megamoe import api;from deep_ep_experimental.megamoe.timer import protocol as p;'
                 'e=api._extension();print(json.dumps(dict(protocol=e.api_protocol,groups=p.DYNAMIC_TRACE_GROUPS,'
                 'aiv0=p.DYNAMIC_TRACE_AIV0,timer_numel=e.timer_numel,expected_numel=p.TOTAL_BUFFER_SIZE)))')
        policy = json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', probe], text=True, env=env))
        if policy['protocol'] != 8 or policy['groups'] != 32 or policy['aiv0'] != 1 or policy['timer_numel'] != policy['expected_numel']:
            raise ValueError('Native/parser timer ABI mismatch')
        registry[method] = dict(revision=a.revision if method == 'combined' else prior['revision'],
            protocol=8, archive_sha256=a.source_sha256 if method == 'combined' else prior['archive_sha256'],
            native={str(f.relative_to(pkg)):sha(f) for f in (pkg/'deep_ep_experimental/megamoe').glob('*.so')},
            trace_policy=policy, trace_only_changes={})
    if registry['control']['native'] != prior['native']:
        raise ValueError('Frozen control changed')
    if a.host_only and registry['control']['native']['deep_ep_experimental/megamoe/libmegamoe_kernel.so'] != registry['combined']['native']['deep_ep_experimental/megamoe/libmegamoe_kernel.so']:
        raise ValueError('Host-only device binary changed')
    (root/'packages.json').write_text(json.dumps(registry, indent=2))
    with tarfile.open(root/'packages.tar.gz', 'w:gz') as tar:
        tar.add(packages, arcname='packages')
        tar.add(root/'packages.json', arcname='packages.json')
    print('PACKAGES_READY', sha(root/'packages.tar.gz'), flush=True)


if __name__ == '__main__':
    main()
