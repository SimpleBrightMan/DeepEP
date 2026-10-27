"""Frozen candidate identity and full-output Golden case contract."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ARCHIVE_SHA='32dbbb93e008f6fa327298760fb87cfe4fcca1fc00de6475d0c7d0b7591097ac'
REVISION='8118a7647f43db5e1fe86116a1d473514bcb6134'
BATCHES=(32,96,256,512,1024,2048,4096)
CASES=[('balanced',bs) for bs in BATCHES]+[('random',96),('skewed',96)]
CONFIG=dict(schema_version=1,model=dict(hidden=7168,intermediate_hidden=3072,num_experts=384,
    num_topk=6,num_shared_experts=0),bs_values=list(BATCHES),warmup=1,iters=1,seed=20260922)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root):
    root=Path(root).resolve()
    if sha(root/'packages.tar.gz')!=ARCHIVE_SHA:raise ValueError('Frozen package archive mismatch')
    registry=json.loads((root/'packages.json').read_text())['combined']
    if registry['revision']!=REVISION:raise ValueError('Wrong candidate revision')
    with tarfile.open(root/'packages.tar.gz') as tar:
        # Python oracle/API must match the same immutable archive, not just the shared libraries.
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith('packages/combined/') and member.name.endswith(('.py','.so','.json')):
                actual=root/member.name
                if sha(actual)!=hashlib.sha256(tar.extractfile(member).read()).hexdigest():
                    raise ValueError('Extracted package changed: '+member.name)
    base=root/'packages/combined'
    sys.path.insert(0,str(base))
    return registry


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
