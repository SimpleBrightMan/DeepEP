"""隔离构建R31控制/候选：共享同一设备库，只改变host预算；不安装到Conda。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--resume-package', action='store_true', help='Reuse verified completed candidate build after import-only failure')
    a = p.parse_args()
    root = a.root.resolve()
    archive = root / 'source.tar'
    if sha(archive) != a.archive_sha256:
        raise ValueError('Source archive mismatch')
    source = root / 'source'
    if a.resume_package:
        if (root/'packages.json').exists():
            raise ValueError('Already packaged; do not resume a completed build')
        with tarfile.open(archive) as bundle:
            for member in bundle:
                if member.isfile() and sha(source/member.name) != hashlib.sha256(bundle.extractfile(member).read()).hexdigest():
                    raise ValueError('Source changed since initial build')
    else:
        source.mkdir(exist_ok=False)
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter='data')
    cpu_log = f'cpu-contract-resume-{time.time_ns()}.log' if a.resume_package else 'cpu-contract.log'
    with (root / cpu_log).open('x') as log:
        for test in ('test_paired_wave_budget_cpu.py',):
            subprocess.run([sys.executable, '-I', '-B', str(source/'experiments/megamoe/tests'/test), '-v'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    if 'skipped' in (root / cpu_log).read_text():
        raise ValueError('Required CPU checks skipped')
    print('CPU_CONTRACT_PASS', flush=True)
    control_source = root / 'control-source'
    if not a.resume_package:
        shutil.copytree(source, control_source)
    with tarfile.open(root / 'parent-host.tar') as bundle:
        members = [m for m in bundle if m.isfile()]
        if len(members) != 1 or members[0].name != 'experiments/megamoe/csrc/forward.cpp':
            raise ValueError('Unexpected parent host archive')
        parent_host = bundle.extractfile(members[0]).read()
        if a.resume_package:
            if (control_source/members[0].name).read_bytes() != parent_host:
                raise ValueError('Control host changed since initial build')
        else:
            (control_source / members[0].name).write_bytes(parent_host)
    # 控制和候选除forward.cpp外所有生产源码一致；测试/说明不是设备输入。
    production = ('experiments/megamoe/csrc', 'experiments/megamoe/python', 'deep_ep', 'csrc')
    for folder in production:
        for path in (source / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.relative_to(source).as_posix() != 'experiments/megamoe/csrc/forward.cpp':
                if sha(path) != sha(control_source / path.relative_to(source)):
                    raise ValueError('Unexpected production difference: ' + str(path))
    base = Path('/home/codex-deepep/ep32-r31-ds-1021/r31/package')
    frozen = json.loads(base.parent.joinpath('identity.json').read_text())
    if not base.joinpath('deep_ep').is_dir():
        raise ValueError('Frozen R31 base package unavailable')
    catlass_archive = Path('/home/codex-deepep/ep32-main-cause-1022/catlass-for-trace.tar.gz')
    expected_catlass = 'd7cd129b3ad5daba1bb1466ba28081b5d912c7f991c4fc1450bf8e9c4c0de1c2'
    if sha(catlass_archive) != expected_catlass:
        raise ValueError('CATLASS mismatch')
    catlass = root / 'catlass'
    if not a.resume_package:
        catlass.mkdir()
        with tarfile.open(catlass_archive) as bundle:
            bundle.extractall(catlass, filter='data')
    cmake = str(Path(sys.executable).with_name('cmake'))
    pybind = subprocess.check_output([sys.executable, '-m', 'pybind11', '--cmakedir'], text=True).strip()
    packages = root / 'packages'
    packages.mkdir(exist_ok=a.resume_package)
    registry = {}
    shared_kernel = None
    for method, src, groups, revision in (
            ('paired2', source, 3, a.revision),
            ('paired', control_source, 6, 'd986ac9dbe576af5e3d966617099cc9a0f302baf')):
        package = packages / method
        reuse_candidate = a.resume_package and method == 'paired2'
        if not reuse_candidate:
            shutil.copytree(base, package)
        # 只在新副本中用精确源码覆盖Python层，保留未改动的DeepEP核心扩展ABI。
        shutil.copytree(src/'experiments/megamoe/python/deep_ep_experimental', package/'deep_ep_experimental', dirs_exist_ok=True)
        build = root / ('build-' + method)
        command = [cmake, '-S', str(src), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                   '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
                   '-DDEEPEP_BUILD_MEGAMOE=ON', '-DDEEPEP_MEGAMOE_TIMER=ON',
                   '-DASCEND_HOME_PATH=/home/pkg/b080/cann-9.2.0',
                   '-DSHMEM_ROOT=/home/codex-deepep/20261012-d8-sdk/DeepEP/third-party/shmem/install/shmem',
                   '-DCATLASS_ROOT=' + str(catlass), '-DPython_EXECUTABLE=' + sys.executable, '-Dpybind11_DIR=' + pybind]
        (root/('configure-' + method + '.json')).write_text(json.dumps(command, indent=2))
        if reuse_candidate:
            if '[100%] Built target megamoe_ext' not in (root/'build-paired2.log').read_text():
                raise ValueError('Candidate native build did not complete')
            print('REUSE_COMPLETED_CANDIDATE_BUILD', flush=True)
        else:
            print('BUILD_START', method, flush=True)
            with (root/('build-' + method + '.log')).open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
                if shared_kernel is not None:
                    # 两版device源相同，使用同一已编译库，排除编译路径差异这一变量。
                    shutil.copy2(shared_kernel, build/'experiments/megamoe/libmegamoe_kernel.so')
                subprocess.run([cmake, '--build', str(build), '--target', 'megamoe_ext', '--parallel', '2', '--verbose'],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        kernel = build/'experiments/megamoe/libmegamoe_kernel.so'
        if shared_kernel is None:
            shared_kernel = kernel
        elif sha(kernel) != sha(shared_kernel):
            raise ValueError('Device kernel differs between control and candidate')
        extensions = list((build/'experiments/megamoe').glob('_C*.so'))
        if len(extensions) != 1:
            raise ValueError('Ambiguous host extension')
        native = package/'deep_ep_experimental/megamoe'
        for old in native.glob('_C*.so'):
            old.unlink()  # 仅替换本次新建副本。
        shutil.copy2(extensions[0], native/extensions[0].name)
        shutil.copy2(kernel, native/kernel.name)
        probe = ('import sys,json,torch,torch_npu;sys.path.insert(0,' + repr(str(package)) + ');'
                 'from deep_ep_experimental.megamoe import api;from deep_ep_experimental.megamoe.timer import protocol as p;'
                 'e=api._extension();print(json.dumps(dict(protocol=e.api_protocol,paired_gate_up_max_tokens=e.paired_gate_up_max_tokens,'
                 'timer_numel=e.timer_numel,expected_numel=p.TOTAL_BUFFER_SIZE,groups=p.DYNAMIC_TRACE_GROUPS,aiv0=p.DYNAMIC_TRACE_AIV0)))')
        policy = json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', probe], text=True))
        if policy['protocol'] != 8 or policy['groups'] != 32 or policy['aiv0'] != 1 or policy['timer_numel'] != policy['expected_numel']:
            raise ValueError('Wrong protocol/timer policy')
        registry[method] = dict(revision=revision, protocol=8, wave_m_groups=groups,
            archive_sha256=a.archive_sha256, parent_host_archive_sha256=sha(root/'parent-host.tar'),
            host_source_sha256=sha(src/'experiments/megamoe/csrc/forward.cpp'),
            native={str(path.relative_to(package)):sha(path) for path in native.glob('*.so')},
            trace_policy=policy, trace_only_changes={}, frozen_core_identity=frozen.get('revision'),
            trace_catlass_archive_sha256=expected_catlass)
        print('BUILD_PASS', method, flush=True)
    (root/'packages.json').write_text(json.dumps(registry, indent=2))
    with tarfile.open(root/'packages.tar.gz', 'w:gz') as bundle:
        bundle.add(packages, arcname='packages')
        bundle.add(root/'packages.json', arcname='packages.json')
    print('PACKAGES_READY', sha(root/'packages.tar.gz'), flush=True)


if __name__ == '__main__':
    main()
