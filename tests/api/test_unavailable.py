"""An API-only checkout must never report successful communication or completion."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import deep_ep


ROOT = Path(__file__).resolve().parents[2]
BASELINE = json.loads(Path(__file__).with_name("upstream_api.json").read_text(encoding="utf-8"))
METHODS = [(name, method) for name, spec in BASELINE["objects"].items() for method in spec["methods"]]


class Uninspectable:
    """Fail if a stub touches a tensor, process group, handle, or event."""

    def __getattr__(self, name):
        raise AssertionError(f"Stub unexpectedly inspected input attribute {name}")


@pytest.mark.parametrize("name,method", METHODS, ids=[f"{n}.{m}" for n, m in METHODS])
def test_every_declared_operation_fails_before_touching_inputs(name, method):
    operation = getattr(getattr(deep_ep, name), method)
    args = [
        Uninspectable()
        for p in inspect.signature(operation).parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    with pytest.raises(NotImplementedError, match=rf"{name}\.{method} is not implemented"):
        operation(*args)


@pytest.mark.parametrize("name", BASELINE["exports"])
def test_constructors_cannot_create_usable_objects(name):
    cls = getattr(deep_ep, name)
    args = [Uninspectable() for p in inspect.signature(cls).parameters.values() if p.default is inspect.Parameter.empty]
    with pytest.raises(NotImplementedError, match=f"{name}.__init__"):
        cls(*args)


def run_isolated(code):
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1")
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_imports_and_failure_paths_do_not_load_framework_or_native_runtime():
    run_isolated(
        """
import importlib.abc, sys
class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'numpy', 'mpi4py'} or fullname == 'deep_ep._C':
            raise AssertionError('Unexpected optional dependency import: ' + fullname)
sys.meta_path.insert(0, BlockFramework())
import deep_ep
from deep_ep.buffers.legacy import Buffer
from deep_ep.buffers.elastic import ElasticBuffer, EPHandle
from deep_ep.utils.event import EventHandle, EventOverlap
assert all(getattr(deep_ep, name) is not None for name in deep_ep.__all__)
for operation in (lambda: Buffer(None), lambda: ElasticBuffer(None), Buffer.capture,
                  ElasticBuffer.capture, EventHandle, EventOverlap):
    try:
        operation()
    except NotImplementedError:
        continue
    raise AssertionError('Stub unexpectedly succeeded')
assert 'deep_ep._C' not in sys.modules
assert 'torch' not in sys.modules
"""
    )


def test_fake_event_does_not_gain_completion_or_context_manager_success():
    event = object.__new__(deep_ep.EventOverlap)
    with pytest.raises(NotImplementedError):
        event.current_stream_wait(release_handle=True)
    with pytest.raises(NotImplementedError):
        with event:
            pytest.fail("Entered an unimplemented completion context")
