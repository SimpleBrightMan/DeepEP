"""Freeze a fresh reference node from git archive; source and native hashes are explicit."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile

p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('name');p.add_argument('revision')
a=p.parse_args()
if not a.name.replace('-','').isalnum():raise ValueError('Invalid node name')
out=a.campaign/a.name;out.mkdir(exist_ok=False)
source=out/'source';source.mkdir()
with tarfile.open(a.campaign/f'{a.name}.tar') as bundle:
    revision=bundle.pax_headers.get('comment','')
    if len(a.revision)<7 or not revision.startswith(a.revision):raise ValueError('Archive revision mismatch')
    for m in bundle.getmembers():
        rel=PurePosixPath(m.name)
        if rel.is_absolute() or '..' in rel.parts or not (m.isfile() or m.isdir()):raise ValueError('Unsafe source archive')
    bundle.extractall(source,filter='data')
package=out/'package';package.mkdir()
shutil.copytree(source/'analysis/ds-bisect-1021/reference/deep_gemm',package/'deep_gemm')
host=a.campaign.parent/'candidate-dashboard-1020/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so'
expected='896ff49f1c71ab6cc5eb403699074469dbf7c314cd33c41a4f29807644f0aab4'
if hashlib.sha256(host.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen host extension changed')
shutil.copyfile(host,package/'deep_gemm'/host.name)
identity=dict(node=a.name,revision=revision,host_source='frozen DS host extension; same ABI; not rebuilt',
              archive_sha256=hashlib.sha256((a.campaign/f'{a.name}.tar').read_bytes()).hexdigest(),
              files={str(f.relative_to(package)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in sorted(package.rglob('*')) if f.is_file()})
variant=source/'analysis/ds-bisect-1021/reference/VARIANT.json'
identity['variant']=json.loads(variant.read_text()) if variant.exists() else dict(weight_sf_layout='pair_major',activation_sf_layout='pair_major',numeric_contract='DS')
(out/'identity.json').write_text(json.dumps(identity,indent=2));(out/'build.rc').write_text('0\n')
print('DS_SOURCE_PREPARED',a.name,revision,flush=True)
