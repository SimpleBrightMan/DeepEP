"""只读核对包身份及设备入口代码段；不执行.so，不把差异直接归因于性能。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import tarfile


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def elf(blob):
    if blob[:6] != b'\x7fELF\x02\x01':
        return None
    shoff = struct.unpack_from('<Q', blob, 40)[0]
    size, count, string_index = struct.unpack_from('<HHH', blob, 58)
    if size != 64 or not 0 < count < 4096 or shoff + size * count > len(blob) or string_index >= count:
        return None
    sections = [struct.unpack_from('<IIQQQQIIQQ', blob, shoff + i * size) for i in range(count)]

    def string(data, offset):
        end = data.find(b'\0', offset)
        if end < 0:
            raise ValueError('Unterminated ELF string')
        return data[offset:end].decode(errors='replace')

    entries = []
    for section in sections:
        if section[1] not in (2, 11) or section[9] != 24 or section[6] >= count:
            continue
        strings = sections[section[6]]
        strings = blob[strings[4]:strings[4] + strings[5]]
        for position in range(section[4], section[4] + section[5], 24):
            name, info, _, index, value, length = struct.unpack_from('<IBBHQQ', blob, position)
            name = string(strings, name)
            active = (name.startswith('_Z15mega_moe_kernelILb1ELb1ELb0EE') or
                      name.startswith('_Z15mega_moe_kernelILb1ELb1EE'))
            if not active or info & 15 != 2 or not length or index >= count:
                continue
            target = sections[index]
            start = target[4] + value - target[3]
            if start < 0 or start + length > len(blob):
                raise ValueError('Kernel symbol outside ELF')
            entries.append(dict(name=name, size=length, sha256=sha(blob[start:start + length])))
    return dict(machine=struct.unpack_from('<H', blob, 18)[0], entries=entries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packages', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(package_sha256=sha(a.packages.read_bytes()), methods={})
    with tarfile.open(a.packages) as tar:
        registry = json.loads(tar.extractfile('packages.json').read())
        for method in ('control', 'combined'):
            relative = 'deep_ep_experimental/megamoe/libmegamoe_kernel.so'
            data = tar.extractfile('packages/' + method + '/' + relative).read()
            if sha(data) != registry[method]['native'][relative]:
                raise ValueError('Native identity mismatch: ' + method)
            objects = []
            for match in re.finditer(b'\x7fELF', data):
                try:
                    obj = elf(data[match.start():])
                except (struct.error, ValueError):
                    obj = None
                if obj:
                    objects.append(dict(offset=match.start(), **obj))
            result['methods'][method] = dict(revision=registry[method]['revision'],
                                             native=registry[method]['native'], elfs=objects)
    comparison = []
    before = [e for e in result['methods']['control']['elfs'] if e['machine'] == 4137]
    after = [e for e in result['methods']['combined']['elfs'] if e['machine'] == 4137]
    if len(before) != len(after) or not before:
        raise ValueError('Device ELF domain mismatch')
    for index, (left, right) in enumerate(zip(before, after)):
        for core in ('aic', 'aiv'):
            old = [e for e in left['entries'] if e['name'].endswith('_mix_' + core)]
            new = [e for e in right['entries'] if e['name'].endswith('_mix_' + core)]
            if len(old) != 1 or len(new) != 1:
                raise ValueError('Ambiguous paired A8W4 entry: ' + core)
            old, new = old[0], new[0]
            comparison.append(dict(device_elf_index=index, core=core, before_bytes=old['size'], after_bytes=new['size'],
                                   identical_entry_bytes=old['sha256'] == new['sha256']))
    result['comparison'] = comparison
    result['interpretation'] = 'Raw entry bytes only; changed bytes/symbol names are not proof of faster or slower execution.'
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(comparison=comparison, revisions={m: v['revision'] for m, v in result['methods'].items()})))


if __name__ == '__main__':
    main()
