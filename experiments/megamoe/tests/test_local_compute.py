"""Exercise the CPU launcher contract without importing torch or starting devices."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/local_compute.py"
    spec = importlib.util.spec_from_file_location("local_compute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("args", [
    ["--bs", "0"], ["--bs", "9"], ["--bs", "65544"],
    ["--iters", "1"], ["--timeout", "0"],
    ["--devices", "0", "0"], ["--devices", "8"],
    ["--warmup", "0"], ["--profile-iters", "1"], ["--profile-rounds", "0"],
])
def test_invalid_config_does_not_launch(launcher, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["local_compute.py", *args])
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched invalid config"))
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2


def test_collects_only_requested_devices(launcher, monkeypatch, tmp_path):
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["local_compute.py", "--devices", "1", "3", "--output", str(output)])
    calls = []

    class Completed:
        returncode = 0

        def __init__(self, command, **kwargs):
            device = int(command[command.index("--worker") + 1])
            calls.append(device)
            (output / f"device_{device}.json").write_text(json.dumps({"device": device}))

        def poll(self):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Completed)
    launcher.main()
    assert calls == [1, 3]
    assert json.loads((output / "summary.json").read_text()) == [{"device": 1}, {"device": 3}]


def test_worker_failure_cleans_up_only_launched_children(launcher, monkeypatch, tmp_path):
    output = tmp_path / "failed"
    monkeypatch.setattr(sys, "argv", ["local_compute.py", "--devices", "0", "1", "--output", str(output)])
    children = []

    class Child:
        def __init__(self, command, **kwargs):
            self.returncode = 1 if not children else None
            self.terminated = False
            children.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="Worker failed"):
        launcher.main()
    assert not children[0].terminated
    assert children[1].terminated
    assert not (output / "summary.json").exists()
