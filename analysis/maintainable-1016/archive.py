"""Archive reproducible measurements, not multi-GB output arrays or credentials."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    files = set()
    case_files = {"manifest.json", "profiling.json", "output_regression.json", "resources.json", "kernel_samples.json"}
    files.update(p for p in (root / "runs").rglob("*") if p.is_file() and
                 (p.name in case_files or (p.name.startswith("kernel_details") and p.suffix == ".csv")))
    files.update(p for p in (root / "a8w8").rglob("*.json") if p.is_file())
    files.update(p for p in root.iterdir() if p.is_file() and p.suffix in (".log", ".csv", ".json", ".sh", ".py"))
    kernel = root / "source/experiments/megamoe/csrc"
    identities = {str(p.relative_to(kernel)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(kernel.rglob("*")) if p.is_file() and p.suffix in (".cpp", ".hpp")}
    payload = json.dumps(identities, indent=2).encode()
    with tarfile.open(root / "compact-evidence.tar.gz", "x:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        entry = tarfile.TarInfo("compiled-kernel-files.json")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))


if __name__ == "__main__":
    main()
