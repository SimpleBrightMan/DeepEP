"""The EP16 PUT profile owns independent queues without changing MTE owners."""

from types import SimpleNamespace

import pytest

from deep_ep import _runtime
from deep_ep._store import StoreGroup


class UrmaRuntime(_runtime.Runtime):
    _megamoe = True
    _megamoe_urma = True


@pytest.mark.parametrize("case", ["normal", "large", "world", "different-host", "mixed-profile", "missing-factory"])
def test_urma_runtime_profile(monkeypatch, case):
    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://127.0.0.1:19091")
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "host")
    monkeypatch.setenv("DEEPEP_SHMEM_DENSE_QPS", "0")
    created = []

    def create(*args):
        created.append(args)
        return SimpleNamespace(close=lambda: None)

    def wrong(*args):
        pytest.fail("URMA must not use the Elastic or original MegaMoE factory")

    native = SimpleNamespace(runtime_busy=lambda: False, Runtime=wrong, create_megamoe_runtime=wrong)
    if case != "missing-factory":
        native.create_megamoe_urma_runtime = create
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0, Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    group = StoreGroup(None, 0, 8 if case == "world" else 16, prefix="test")

    def exchange(local, stage):
        assert local[-1] == 33  # MTE|UDMA plus the dedicated MegaMoE profile.
        peers = []
        for rank in range(group.size()):
            peer = list(local)
            peer[6] = rank
            peers.append(peer)
        if case == "different-host":
            peers[-1][4] += 1
        elif case == "mixed-profile":
            peers[-1][-1] = 0
        return peers

    monkeypatch.setattr(group, "all_gather", exchange)
    errors = {
        "world": "EP16",
        "different-host": "one Linux host",
        "mixed-profile": "runtime engines",
        "missing-factory": "missing create_megamoe_urma_runtime",
    }
    if case in errors:
        with pytest.raises((ValueError, RuntimeError), match=errors[case]):
            UrmaRuntime(group, 2 << 20)
        assert not created
    else:
        capacity = 4 << 30 if case == "large" else 2 << 20
        runtime = UrmaRuntime(group, capacity)
        try:
            assert created == [(0, 16, capacity, "tcp://127.0.0.1:19091", 120)]
        finally:
            runtime.close()


def test_urma_owner_requires_megamoe_class():
    class InvalidRuntime(_runtime.Runtime):
        _megamoe_urma = True

    with pytest.raises(ValueError, match="MegaMoE runtime owner"):
        InvalidRuntime(None, 2 << 20)
