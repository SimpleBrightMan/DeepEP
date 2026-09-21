"""Record a prebuilt, administrator-provided SDK; never download/build dependencies.

Header checks establish the expected public API, not binary provenance. The SDK
provider must attest that the library was built from the pinned commit.
"""

import argparse
import hashlib
import json
from pathlib import Path


def record(sdk, revision):
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())
    expected = lock["shmem"]
    if revision != expected["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    for relative, digest in expected["header_sha256_lf"].items():
        header = sdk / "include" / relative
        normalized = header.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != digest:
            raise ValueError(f"SDK public header mismatch: {relative}")
    files = sorted((sdk / "include").rglob("*.h")) + [sdk / "lib/libshmem.so"]
    hashes = {}
    for path in files:
        if not path.resolve().is_relative_to(sdk) or not path.is_file():
            raise ValueError("SDK files must stay inside the selected root")
        hashes[path.relative_to(sdk).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = sdk / "deepep-sdk.json"
    if manifest.exists():
        raise FileExistsError("SDK manifest already exists; do not overwrite an existing SDK identity")
    manifest.write_text(json.dumps({"revision": revision, "sha256": hashes}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True)
    parser.add_argument("--revision", required=True, help="SDK provider's source revision attestation")
    args = parser.parse_args()
    print(record(args.sdk, args.revision))
