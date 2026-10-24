"""安装冻结SHMEM运行库；已有文件必须同SHA，不覆盖不同版本。"""
import hashlib
import json
from pathlib import Path
import tarfile

stage=Path('/home/codex-deepep/r31-ep32-continuous-1024-runtime')
archive=stage/'shmem-libs.tar.gz'
expected='5233b9dfff194b66295477d2cb0cca3f5e65459628774f5593216ffad6d00234'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected,'SHMEM archive identity mismatch'
target=Path('/home/codex-deepep/ep32-r31-ds-1021/shmem/lib')
assert target.resolve()==target,'Unexpected SHMEM target symlink'
names={'libshmem.so','libshmem_utils.so','aclshmem_bootstrap_uid.so','aclshmem_bootstrap_config_store.so'}
with tarfile.open(archive) as source:
    members=[m for m in source.getmembers() if m.isfile()]
    assert {m.name for m in members}=={'lib/'+name for name in names},'Unexpected runtime files'
    payloads={Path(m.name).name:source.extractfile(m).read() for m in members}
target.mkdir(parents=True,exist_ok=True)
for name,content in payloads.items():
    if (target/name).exists():
        assert hashlib.sha256((target/name).read_bytes()).digest()==hashlib.sha256(content).digest(),f'Existing SHMEM differs: {name}'
for name,content in payloads.items():
    if not (target/name).exists():
        (target/name).write_bytes(content)
        (target/name).chmod(0o755)
print(json.dumps(dict(shmem_archive_sha256=expected,files={n:hashlib.sha256(v).hexdigest() for n,v in payloads.items()})))
