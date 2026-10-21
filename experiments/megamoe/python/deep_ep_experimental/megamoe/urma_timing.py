"""Device launch alignment for the URMA benchmark, outside operator timing.

The guard follows mega_test/benchmark/timing.py: one HCCL barrier and one
finite-input exp sequence before a continuous warmup + measurement block.
It reduces host launch skew; it cannot remove physical rank speed differences.
"""

from datetime import timedelta
import math


class UrmaLaunchGuard:
    """Own a benchmark-only HCCL domain and reusable FP32 exp buffers.

    StoreGroup remains responsible for control/reporting. Its unique session
    namespaces the HCCL rendezvous without creating a second TCP listener or
    reusing another job's communication domain. This standalone benchmark must
    not already have a torch.distributed default process group.
    """

    def __init__(self, group, shape=(100000, 20000), repeat=50, timeout=300):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        self.group = group
        self.shape, self.repeat = tuple(shape), repeat
        self.bytes = math.prod(self.shape) * 4 * 2
        self.input = self.output = self.process_group = None
        error = None
        try:
            if dist.is_initialized():
                raise RuntimeError("URMA standalone benchmark requires no pre-existing ProcessGroup")
            if not self.shape or any(type(n) is not int or n < 1 for n in self.shape):
                raise ValueError("exp shape dimensions must be positive integers")
            if type(repeat) is not int or repeat < 0 or timeout <= 0:
                raise ValueError("exp repeat must be nonnegative and HCCL timeout positive")
            if repeat:
                self.input = torch.zeros(self.shape, dtype=torch.float32, device="npu")
                self.output = torch.empty_like(self.input)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        # Propagate allocation/preflight failures before peers enter HCCL init.
        checks = group.all_gather(error, "URMA launch guard allocation")
        if any(checks):
            self.input = self.output = None
            raise RuntimeError(
                f"URMA launch guard needs {self.bytes / 2**30:.2f} GiB/rank for two FP32 buffers; "
                "adjust --exp-shape/--exp-repeat. "
                + "; ".join(f"rank={rank}: {item}" for rank, item in enumerate(checks) if item)
            )
        self.store = dist.PrefixStore(group._prefix + "urma-benchmark-hccl/", group._store)
        dist.init_process_group(
            backend="hccl",
            store=self.store,
            rank=group.rank(),
            world_size=group.size(),
            timeout=timedelta(seconds=timeout),
        )
        self.process_group = dist.group.WORLD

    def begin(self):
        """Enqueue the guard on the caller stream; never synchronize after exp.

        HCCL barrier is a device collective, unlike StoreGroup.barrier. URMA's
        _enqueue_forward subsequently inserts owner_stream.wait_stream(caller),
        so the dedicated native stream cannot overtake this exp sequence.
        """
        self.dist.barrier(group=self.process_group)
        for _ in range(self.repeat):
            self.torch.exp(self.input, out=self.output)

    def end(self):
        # forward_prepared orders caller after owner. HCCL therefore sees the
        # completed block; neither this barrier nor synchronization is timed.
        self.dist.barrier(group=self.process_group)
        self.torch.npu.synchronize()

    def close(self):
        if self.process_group is not None:
            self.dist.destroy_process_group(self.process_group)
            self.process_group = None
        self.input = self.output = None

    def describe(self):
        return dict(
            method="HCCL barrier + finite-input exp before continuous block; HCCL barrier + synchronize after",
            exp_shape=list(self.shape),
            exp_repeat=self.repeat,
            allocated_bytes_per_rank=self.bytes if self.repeat else 0,
            scope="CACHED_FORWARD and each separate timer/trace diagnostic pass",
            stream_order="caller barrier/exp -> owner.wait_stream(caller) -> native -> caller.wait_stream(owner)",
            limitations="reduces host launch skew; does not equalize physical device speeds or synchronize every iteration",
        )


def measure_cached_forward(forward, *, warmup, iters, guard):
    """Enqueue warmup and samples together; inspect Events only after one drain."""
    torch = guard.torch
    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(warmup + iters)]
    guard.begin()
    for begin, end in events:
        begin.record()
        forward()
        end.record()
    guard.end()
    # Warmup execution shares the same continuous block but not statistics.
    return [begin.elapsed_time(end) * 1000 for begin, end in events[warmup:]]
