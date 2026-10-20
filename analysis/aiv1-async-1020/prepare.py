"""Use the production CMake target; rebuild host and device for the new timer ABI."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

out = Path(__file__).resolve().parent
root = out.parent
source = out / 'async-source'
package = out / 'async-package'
resume = sys.argv[1:] == ['--resume-build']
if sys.argv[1:] and not resume:
    raise ValueError('Only --resume-build is accepted, after a recorded configure/link failure')
if resume:
    if (out / 'identity.json').exists() or not (out / 'compile.log').exists():
        raise ValueError('Resume requires a failed build, never a completed candidate')
else:
    source.mkdir(exist_ok=False)
    with tarfile.open(root / 'aiv1-async-source-1020.tar') as bundle:
        bundle.extractall(source, filter='data')
    shutil.copytree(root / 'bs512-focus-1020/baseline-package', package)
    python_package = source / 'experiments/megamoe/python/deep_ep_experimental/megamoe'
    shutil.copytree(python_package, package / 'deep_ep_experimental/megamoe', dirs_exist_ok=True)
build = out / 'native-build'
cann = Path(os.environ['ASCEND_HOME_PATH'])
shmem = root.parent / '20261012-d8-sdk/DeepEP/third-party/shmem/install/shmem'
catlass = root / 'bs512-focus-1020/catlass'
commands = [
    ['cmake', '-S', str(out / 'native'), '-B', str(build), '-G', 'Ninja',
     f'-DPython_EXECUTABLE={sys.executable}', f'-DMEGAMOE_SOURCE={source}',
     f'-DASCEND_HOME_PATH={cann}',
     f'-DSHMEM_ROOT={shmem}', f'-DCATLASS_ROOT={catlass}',
     f'-DASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR={cann}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
    ['cmake', '--build', str(build), '--target', 'megamoe_ext', '--parallel', '2'],
]
(out / 'build-commands.json').write_text(json.dumps(commands, indent=2))
with (out / ('compile-resume.log' if resume else 'compile.log')).open('x') as log:
    for command in commands:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
libraries = [build / 'megamoe/libmegamoe_kernel.so', *sorted((build / 'megamoe').glob('_C*.so'))]
if len(libraries) != 2:
    raise RuntimeError(f'Expected one kernel and one rebuilt extension, got {libraries}')
for path in libraries:
    shutil.copyfile(path, package / 'deep_ep_experimental/megamoe' / path.name)
identity = dict(package=str(package), parent='a772852 resident + fair',
    wrapper_sha256=hashlib.sha256((out / 'native/CMakeLists.txt').read_bytes()).hexdigest(),
    libraries={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
    source_hashes={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((source / 'experiments/megamoe').rglob('*')) if p.is_file()},
    catlass_revision='2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1')
identity['kernel_sha256'] = identity['libraries']['libmegamoe_kernel.so']
(out / 'identity.json').write_text(json.dumps(identity, indent=2))
print('BUILD_COMPLETE', identity['libraries'], flush=True)
