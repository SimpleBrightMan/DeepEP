"""Build a fresh cumulative node from an immutable archive, never the live checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('revision')
a = p.parse_args()
if not a.name.replace('-', '').isalnum():
    raise ValueError('Invalid node name')
campaign = Path(__file__).resolve().parent
root = campaign.parent
out = campaign / a.name
out.mkdir(exist_ok=False)
source, package, build = out/'source', out/'package', out/'native-build'
source.mkdir()
archive = campaign/f'{a.name}.tar'
with tarfile.open(archive) as bundle:
    archive_revision = bundle.pax_headers.get('comment', '')
    if not archive_revision.startswith(a.revision) or len(a.revision) < 7:
        raise ValueError(f'Archive/source revision mismatch: requested={a.revision}, archive={archive_revision}')
    for member in bundle.getmembers():
        rel = PurePosixPath(member.name)
        if rel.is_absolute() or '..' in rel.parts or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe source archive')
    bundle.extractall(source, filter='data')
shutil.copytree(root/'cv-adaptive-1020-v1/cv-package', package)
shutil.copytree(source/'experiments/megamoe/python/deep_ep_experimental/megamoe',
                package/'deep_ep_experimental/megamoe', dirs_exist_ok=True)
tests = source/'experiments/megamoe/tests'
with (out/'cpu-tests.log').open('x') as log:
    selected = [
        str(tests/name) for name in ('test_aiv1_resident_ub.py', 'test_aiv1_fair_scheduler.py',
        'test_dispatch.py', 'test_aiv1_async_events.py', 'test_gmm1_cv_plan.py', 'test_timer.py', 'test_breakdown.py')]
    if (tests/'test_small_linear_task_schedule.py').exists(): selected.append(str(tests/'test_device_schedule.py'))
    if (tests/'test_small_linear_task_schedule.py').exists(): selected.append(str(tests/'test_small_linear_task_schedule.py'))
    if (tests/'test_shared_epilogue_credit.py').exists(): selected.append(str(tests/'test_shared_epilogue_credit.py'))
    subprocess.run([sys.executable, '-I', '-m', 'pytest', '-q', *selected],
        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
cann = Path(os.environ['ASCEND_HOME_PATH'])
commands = [
    ['cmake', '-S', str(campaign/'native'), '-B', str(build), '-G', 'Ninja',
     f'-DPython_EXECUTABLE={sys.executable}', f'-DMEGAMOE_SOURCE={source}', f'-DASCEND_HOME_PATH={cann}',
     f'-DSHMEM_ROOT={root.parent}/20261012-d8-sdk/DeepEP/third-party/shmem/install/shmem',
     f'-DCATLASS_ROOT={root}/bs512-focus-1020/catlass',
     f'-DASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR={cann}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
    ['cmake', '--build', str(build), '--target', 'megamoe_ext', '--parallel', '2']]
(out/'commands.json').write_text(json.dumps(commands, indent=2))
with (out/'compile.log').open('x') as log:
    for command in commands:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
libraries = [build/'megamoe/libmegamoe_kernel.so', *sorted((build/'megamoe').glob('_C*.so'))]
if len(libraries) != 2:
    raise ValueError('Expected both host extension and kernel')
for lib in libraries:
    shutil.copyfile(lib, package/'deep_ep_experimental/megamoe'/lib.name)
identity = dict(revision=archive_revision, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    libraries={lib.name: hashlib.sha256(lib.read_bytes()).hexdigest() for lib in libraries},
    source_hashes={str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted((source/'experiments/megamoe').rglob('*')) if f.is_file()},
    catlass_revision='2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1', package=str(package))
(out/'identity.json').write_text(json.dumps(identity, indent=2))
(out/'build.rc').write_text('0\n')
print('BUILD_COMPLETE', a.name, identity['libraries'], flush=True)
