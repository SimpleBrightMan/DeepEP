"""EP regression through the public A8W8 API; compare reports across builds.

This records complete BF16 output hashes, not independent Golden accuracy.
Launch with torchrun and a matching SHMEM endpoint/environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

from deep_ep_experimental.megamoe.api import StoreGroup, SymmBuffer, mega_moe, transform_weights_for_mega_moe
from deep_ep_experimental.megamoe.reference import quantize_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env()
    rank, world = group.rank(), group.size()
    if world != 8:
        raise ValueError("This regression fixture is EP8, E8, TopK2")
    h, intermediate = 1024, 256
    rng = np.random.default_rng(20261016 + rank)

    def weights(n, k):
        data = torch.from_numpy(rng.normal(0, 0.025, (1, n, k)).astype(np.float32)).npu()
        data = data.to(torch.float8_e4m3fn)
        scales = torch.full((1, n, k // 64, 2), 127, dtype=torch.uint8, device="npu")
        return data, scales.view(torch.float8_e8m0fnu)

    w1, w2 = transform_weights_for_mega_moe(weights(2 * intermediate, h), weights(h, intermediate))
    buffer = SymmBuffer(group, 8, 4096, 2, h, intermediate, mma_type="fp8xfp8", ranks_per_node=8)
    reports = []
    try:
        # Return to previous BS as well: workspace/flag lifecycle must survive plan replacement.
        for bs in (1, 17, 96, 257, 4096, 17):
            source = rng.normal(0, 0.25, (bs, h)).astype(np.float32)
            ids = (np.arange(bs, dtype=np.int32)[:, None] + rank + np.array([0, 3], dtype=np.int32)) % world
            buffer.topk_idx[:bs].copy_(torch.from_numpy(ids).npu())
            buffer.topk_weights[:bs].fill_(0.5)
            buffer.prepare(bs)
            buffer.validate_inputs(bs)
            for input_kind in ("bf16", "fp8"):
                if input_kind == "bf16":
                    x = torch.from_numpy(source).npu().to(torch.bfloat16)
                else:
                    data, scales = quantize_input(source)
                    x = (torch.from_numpy(data).npu().view(torch.float8_e4m3fn),
                         torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu))
                for _ in range(5):
                    output, counts = mega_moe(x, buffer.topk_idx[:bs], buffer.topk_weights[:bs], w1, w2, buffer)
                torch.npu.synchronize()
                cpu = output.cpu()
                if not bool(torch.isfinite(cpu.float()).all()):
                    raise RuntimeError(f"rank={rank} BS={bs} input={input_kind}: non-finite output")
                reports.append(dict(bs=bs, input=input_kind,
                    sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    counts=counts.cpu().tolist()))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"rank_{rank}.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(rank=rank, ep=world, mma="fp8xfp8", golden=False, cases=reports), stream, indent=2)
        group.barrier("A8W8 regression complete")
    except BaseException as error:
        group.abort(f"A8W8 regression rank={rank}: {error}")
        raise
    else:
        buffer.destroy()
        group.close()


if __name__ == "__main__":
    main()
