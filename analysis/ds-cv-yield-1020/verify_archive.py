"""Extract the completed evidence bundle and verify every archived byte locally."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive, destination = (Path(p).resolve() for p in sys.argv[1:])
if destination.exists():
    raise ValueError('Use a fresh destination; do not overlay old experimental evidence')
if os.name == 'nt':
    destination = Path('\\\\?\\' + str(destination))
destination.mkdir(parents=True)
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        name = PurePosixPath(member.name)
        if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f'Unsafe archive entry: {member.name}')
    bundle.extractall(destination, filter='data')
manifest = json.loads((destination / 'manifest.json').read_text())
for item in manifest:
    name = PurePosixPath(item['path'])
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Unsafe manifest path')
    path = destination / item['path']
    if path.stat().st_size != item['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError(f'Corrupt local evidence: {item["path"]}')
print('LOCAL_MANIFEST_PASS', len(manifest), 'files',
      sum(item['path'].endswith('/kernel_details.csv') for item in manifest), 'kernel CSVs')
