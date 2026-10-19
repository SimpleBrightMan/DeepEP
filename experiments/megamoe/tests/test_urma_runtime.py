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
        return SimpleNamespace(close=lambda: None, megamoe_urma_topology_error="")

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
        if stage == "URMA MTE domain readiness":
            assert local == ""
            return [""] * group.size()
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


@pytest.mark.parametrize(
    "case", ["opt-in", "default-denied", "mixed-opt-in", "duplicate-device", "different-visible", "bad-mte-domain"]
)
def test_two_hosts_eight_ranks_each_require_consistent_opt_in_and_local_device_identity(monkeypatch, case):
    from deep_ep_experimental.megamoe.urma import _UrmaRuntime

    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://192.0.2.1:19091")
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "host-a")
    monkeypatch.setenv("DEEPEP_SHMEM_DENSE_QPS", "0")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    created = []
    closed = []
    aborts = []

    def create(*args):
        created.append(args)
        return SimpleNamespace(close=lambda: closed.append(True), megamoe_urma_topology_error="")

    native = SimpleNamespace(runtime_busy=lambda: False, Runtime=None, create_megamoe_urma_runtime=create)
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0, Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    group = StoreGroup(None, 0, 16, prefix="two-host-test")
    monkeypatch.setattr(group, "abort", aborts.append)
    allow = case != "default-denied"

    def exchange(local, stage):
        if stage == "URMA MTE domain readiness":
            assert local == ""
            return [""] * 15 + (["peer 0 has no MTE topology bit"] if case == "bad-mte-domain" else [""])
        assert local[-1] == 33 + 64 * int(allow)
        peers = []
        for rank in range(16):
            peer = list(local)
            peer[4] += rank // 8  # Different boot identities for the two hosts.
            peer[6] = rank % 8  # The same local device ids are valid on different hosts.
            peers.append(peer)
        if case == "mixed-opt-in":
            peers[-1][-1] ^= 64
        elif case == "duplicate-device":
            peers[9][6] = 0
        elif case == "different-visible":
            peers[9][5] += 1
        return peers

    monkeypatch.setattr(group, "all_gather", exchange)
    failures = {
        "default-denied": "one Linux host",
        "mixed-opt-in": "multi-host opt-in",
        "duplicate-device": "distinct local NPU",
        "different-visible": "common visible-device mapping",
        "bad-mte-domain": "URMA MTE domain is not ready.*rank=15",
    }
    if case in failures:
        with pytest.raises((ValueError, RuntimeError), match=failures[case]):
            _UrmaRuntime(group, 2 << 20, allow_multi_host=allow)
        if case == "bad-mte-domain":
            assert len(created) == 1 and _runtime._OWNER is not None
            assert closed == [] and len(aborts) == 1
        else:
            assert created == []
    else:
        runtime = _UrmaRuntime(group, 2 << 20, allow_multi_host=True)
        try:
            assert runtime._megamoe_urma_multi_host is True
            assert created == [(0, 16, 2 << 20, "tcp://192.0.2.1:19091", 120)]
        finally:
            runtime.close()


def test_multi_host_flag_is_not_available_to_unrelated_runtime_profiles():
    class InvalidRuntime(_runtime.Runtime):
        _megamoe_urma_multi_host = True

    with pytest.raises(ValueError, match="dedicated MegaMoE URMA"):
        InvalidRuntime(None, 2 << 20)


@pytest.mark.parametrize("value", [1, "true", None])
def test_multi_host_opt_in_requires_boolean(value):
    from deep_ep_experimental.megamoe.urma import _UrmaRuntime

    with pytest.raises(TypeError, match="must be bool"):
        _UrmaRuntime(None, 2 << 20, allow_multi_host=value)
