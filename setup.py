"""Metadata hooks do not import frameworks, probe devices or configure CMake."""

from pathlib import Path
import sys

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.resolve()))
# PEP 517 setup execution must add the project root before this local import.
from build_support import CMakeBuild  # noqa: E402

setup(
    name="ascend-deepep",
    version=(ROOT / "VERSION").read_text(encoding="utf-8").strip(),
    description="DeepEP for Ascend: native runtime foundation",
    license_files=["third-party/deepseek-deepep.LICENSE"],
    python_requires=">=3.11",
    packages=find_packages(include=["deep_ep", "deep_ep.*"]),
    ext_modules=[Extension("deep_ep._C", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
