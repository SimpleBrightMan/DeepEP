"""Bounded diagnostic file collection, including nonshared rank filesystems."""

from pathlib import Path

import pytest

from deep_ep._store import StoreGroup, _CHUNK_BYTES
from test_store import MemoryStore, groups, together


class BoundedFileStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.max_payload_bytes = 0
        self.max_payload_keys = 0
        self.payloads = 0

    def set(self, key, value):
        with self.changed:
            if "/file/" in key and key.endswith("/data"):
                assert len(value) <= _CHUNK_BYTES
                self.payloads += 1
            super().set(key, value)
            pending = [v for k, v in self.values.items() if "/file/" in k and k.endswith("/data")]
            self.max_payload_bytes = max(self.max_payload_bytes, sum(map(len, pending)))
            self.max_payload_keys = max(self.max_payload_keys, len(pending))


def source_files(tmp_path, lengths):
    files, expected = [], []
    for rank, size in enumerate(lengths):
        # Every source has its own private directory and unrelated basename.
        directory = tmp_path / f"host-{rank}"
        directory.mkdir()
        path = directory / "source.json"
        content = (bytes([rank + 1]) * size) if size else b""
        path.write_bytes(content)
        files.append(path)
        expected.append(content)
    return files, expected


@pytest.mark.parametrize("dst", [0, 2])
def test_gather_files_are_rank_ordered_and_transport_is_bounded(tmp_path, dst):
    files, expected = source_files(tmp_path, [0, 17, 5 * _CHUNK_BYTES + 19, 2 * _CHUNK_BYTES])
    store = BoundedFileStore()
    peers = groups(4, store)
    destination = tmp_path / "receiver-only" / "collected"
    results = together(
        peers,
        lambda group: group.gather_file(
            files[group.rank()], destination if group.rank() == dst else None, "allrank trace", dst=dst
        ),
    )
    assert [path.name for path in results[dst]] == [f"rank_{rank}.bin" for rank in range(4)]
    assert [path.read_bytes() for path in results[dst]] == expected
    assert all(result is None for rank, result in enumerate(results) if rank != dst)
    assert [path.read_bytes() for path in files] == expected
    assert store.payloads > 0
    assert store.max_payload_keys == 1
    assert store.max_payload_bytes <= _CHUNK_BYTES
    assert not any("/file/" in key or "/collective/" in key for key in store.values)
    assert not list(destination.glob("*.part"))


def test_gather_files_repeated_operations_have_no_stale_data(tmp_path):
    files, expected = source_files(tmp_path, [11, _CHUNK_BYTES + 1, 0])
    peers = groups(3)

    def work(group):
        for iteration in range(3):
            dst = iteration
            result = group.gather_file(
                files[group.rank()],
                tmp_path / f"round-{iteration}" if group.rank() == dst else None,
                f"round {iteration}",
                dst=dst,
            )
            if group.rank() == dst:
                assert [path.read_bytes() for path in result] == expected
        assert group.all_gather(group.rank(), "after exports") == [0, 1, 2]

    together(peers, work)
    assert not any("/file/" in key or "/collective/" in key for key in peers[0]._store.values)


@pytest.mark.parametrize("size", [0, _CHUNK_BYTES + 3])
def test_single_rank_local_copy_needs_no_store_payload(tmp_path, size):
    files, expected = source_files(tmp_path, [size])
    store = BoundedFileStore()
    group = groups(1, store)[0]
    result = group.gather_file(files[0], tmp_path / "collected", "one rank")
    assert [path.read_bytes() for path in result] == expected
    assert store.payloads == 0


def test_missing_source_aborts_other_ranks(tmp_path):
    files, _ = source_files(tmp_path, [3, 5, 7])
    files[1].unlink()

    def work(group):
        with pytest.raises(RuntimeError, match="source.json"):
            group.gather_file(files[group.rank()], tmp_path / "collected" if group.rank() == 0 else None, "missing")

    together(groups(3), work)


def test_receiver_open_failure_aborts_other_ranks(tmp_path, monkeypatch):
    files, _ = source_files(tmp_path, [3, _CHUNK_BYTES + 3, 7])
    original = Path.open

    def fail_output(path, mode="r", *args, **kwargs):
        if path.name == "rank_1.part" and mode == "wb":
            raise OSError("receiver filesystem unavailable")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_output)

    def work(group):
        with pytest.raises(RuntimeError, match="receiver filesystem unavailable"):
            group.gather_file(files[group.rank()], tmp_path / "collected" if group.rank() == 0 else None, "write")

    together(groups(3), work)


class FailingFile:
    def __init__(self, file, *, write=False, short_read=False):
        self.file, self.write_failure, self.short_read = file, write, short_read

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def fileno(self):
        return self.file.fileno()

    def read(self, count):
        if self.short_read:
            return self.file.read(max(count - 1, 0))
        raise OSError("source read unavailable")

    def write(self, payload):
        assert self.write_failure
        raise OSError("receiver disk full")


@pytest.mark.parametrize("failure", ["read", "write", "short_read"])
def test_chunk_io_failure_aborts_without_waiting_for_timeout(tmp_path, monkeypatch, failure):
    files, _ = source_files(tmp_path, [3, 2 * _CHUNK_BYTES + 3, 7])
    original = Path.open

    def fail_io(path, mode="r", *args, **kwargs):
        file = original(path, mode, *args, **kwargs)
        if failure == "write" and path.name == "rank_1.part" and mode == "wb":
            return FailingFile(file, write=True)
        if failure != "write" and path == files[1] and mode == "rb":
            return FailingFile(file, short_read=failure == "short_read")
        return file

    monkeypatch.setattr(Path, "open", fail_io)
    message = {"read": "source read unavailable", "write": "receiver disk full", "short_read": "source shrank"}[failure]

    def work(group):
        with pytest.raises(RuntimeError, match=message):
            group.gather_file(files[group.rank()], tmp_path / "collected" if group.rank() == 0 else None, failure)

    together(groups(3), work)


def test_destination_mismatch_is_collective_error(tmp_path):
    files, _ = source_files(tmp_path, [3, 7])

    def work(group):
        with pytest.raises(RuntimeError, match="metadata mismatch"):
            group.gather_file(files[group.rank()], tmp_path / f"dst-{group.rank()}", "mismatch", dst=group.rank())

    together(groups(2), work)


def test_stage_mismatch_is_collective_error(tmp_path):
    files, _ = source_files(tmp_path, [3, 7])

    def work(group):
        with pytest.raises(RuntimeError, match="collective order mismatch"):
            group.gather_file(
                files[group.rank()], tmp_path / "collected" if group.rank() == 0 else None, f"stage-{group.rank()}"
            )

    together(groups(2), work)


def test_source_is_not_overwritten(tmp_path):
    source = tmp_path / "rank_0.bin"
    source.write_bytes(b"original")
    group = groups(1)[0]
    with pytest.raises(RuntimeError, match="must not overwrite"):
        group.gather_file(source, tmp_path, "overlap")
    assert source.read_bytes() == b"original"


def test_real_tcpstore_file_transfer(tmp_path):
    torch = pytest.importorskip("torch", reason="Real TCPStore test needs CPU PyTorch")
    from datetime import timedelta

    server = torch.distributed.TCPStore(
        "127.0.0.1", 0, None, True, timedelta(seconds=5), wait_for_workers=False, use_libuv=False
    )
    clients = [
        torch.distributed.TCPStore("127.0.0.1", server.port, None, False, timedelta(seconds=5), use_libuv=False)
        for _ in range(3)
    ]
    peers = [StoreGroup(client, rank, 3, prefix="actual-file", timeout=5) for rank, client in enumerate(clients)]
    files, expected = source_files(tmp_path, [0, 5 * _CHUNK_BYTES + 19, 73])
    result = together(
        peers,
        lambda group: group.gather_file(
            files[group.rank()], tmp_path / "collected" if group.rank() == 0 else None, "actual socket"
        ),
    )
    assert [path.read_bytes() for path in result[0]] == expected
    assert not torch.distributed.is_initialized()
