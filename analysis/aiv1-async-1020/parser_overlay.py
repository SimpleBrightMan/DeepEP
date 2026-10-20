"""Version the Python-only exporter fix without rewriting the compiled source snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

out = Path(__file__).resolve().parent
build = out.parent / 'aiv1-async-1020-v2'
check = out / 'parser-check'
check.mkdir(exist_ok=False)
with tarfile.open(out.parent / 'aiv1-async-parser-1020.tar') as archive:
    archive.extractall(check, filter='data')
source = check / 'experiments/megamoe'
(source / 'csrc').symlink_to(build / 'async-source/experiments/megamoe/csrc', target_is_directory=True)
destination = build / 'async-package/deep_ep_experimental/megamoe'
shutil.copytree(source / 'python/deep_ep_experimental/megamoe', destination, dirs_exist_ok=True)
identity = json.loads((out / 'identity.json').read_text())
identity['python_overlay'] = {str(p.relative_to(source / 'python')): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((source / 'python').rglob('*.py'))}
(out / 'identity.json').write_text(json.dumps(identity, indent=2))
print('PYTHON_EXPORT_OVERLAY', len(identity['python_overlay']), flush=True)
