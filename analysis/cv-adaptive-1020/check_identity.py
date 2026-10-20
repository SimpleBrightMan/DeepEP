"""Audit current production source/Python and archived unified native libraries."""
import argparse
import hashlib
import json
from pathlib import Path


def check(source,evidence):
    data=json.loads((evidence/'identity.json').read_text(encoding='utf-8'))
    counts={'native':0,'python':0}
    for name,expected in data['source_hashes'].items():
        role='native' if name.startswith('experiments/megamoe/csrc/') else 'python' if name.startswith('experiments/megamoe/python/') and name.endswith('.py') else None
        if role is None:continue
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Source mismatch: {name}')
        counts[role]+=1
    for name,expected in data['libraries'].items():
        if hashlib.sha256((evidence/'native-adaptive'/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Binary mismatch: {name}')
    print('SOURCE_IDENTITY_PASS',counts,'2 native libraries')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('evidence',type=Path)
    a=p.parse_args()
    check(a.source,a.evidence)
