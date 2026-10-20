"""Verify committed production bytes against the archived native build and exporter overlay."""
import argparse
import hashlib
import json
from pathlib import Path


def check(source, evidence):
    identity = json.loads((evidence / 'identity.json').read_text(encoding='utf-8'))
    verified = 0
    for name, expected in identity['source_hashes'].items():
        if not name.startswith('experiments/megamoe/csrc/'):
            continue
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Native build source mismatch: {name}')
        verified += 1
    overlay = identity['python_overlay']
    for name, expected in overlay.items():
        path = source / 'experiments/megamoe/python' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Exporter overlay mismatch: {name}')
    for name, expected in identity['libraries'].items():
        if hashlib.sha256((evidence / 'native-async' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Native binary mismatch: {name}')
    print('SOURCE_IDENTITY_PASS', verified, 'native source files;', len(overlay), 'Python files; 2 native libraries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    check(args.source, args.evidence)
