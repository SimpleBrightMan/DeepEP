"""Build the fair scheduler separately; retain the resident-only binary as its control."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

out = Path(__file__).resolve().parent
root = out.parent
control = root / 'bs512-focus-1020'
source = out / 'fair-source'
source.mkdir(exist_ok=False)
with tarfile.open(root / 'aiv1-fair-source-1020.tar') as bundle:
    bundle.extractall(source, filter='data')
package = out / 'fair-package'
shutil.copytree(control / 'baseline-package', package)
kernel = package / 'deep_ep_experimental/megamoe/libmegamoe_kernel.so'
command = json.loads((control / 'baseline-compile-command.json').read_text())
command = [x.replace(str(control / 'baseline-source'), str(source)) for x in command]
command[-1] = str(kernel)
(out / 'compile-command.json').write_text(json.dumps(command, indent=2))
with (out / 'compile.log').open('x') as log:
    subprocess.run(command, cwd=source / 'experiments/megamoe', stdout=log,
                   stderr=subprocess.STDOUT, check=True, timeout=300)
identity = dict(package=str(package), kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
    source_hashes={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((source / 'experiments/megamoe').rglob('*')) if p.is_file()},
    parent='4d69066 resident UB', catlass_revision='2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1')
(out / 'identity.json').write_text(json.dumps(identity, indent=2))
print('BUILD_COMPLETE', identity['kernel_sha256'], flush=True)
