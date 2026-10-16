"""Run the installed full-MoE benchmark and archive untimed differential outputs.

This is a baseline/candidate regression check, not an independent numerical Golden.
Output copying happens after collect_profile returns, never inside measured kernels.
"""
import hashlib
import json
import os

import numpy as np
import torch

from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe import benchmark

original_call = api.fp8_fp4_mega_moe
original_profile = benchmark.collect_profile
last_output = None


def remember_output(y, *args, **kwargs):
    global last_output
    original_call(y, *args, **kwargs)
    last_output = y


def profile_and_archive(call, group, framework, config, prefix, directory, bs):
    samples = original_profile(call, group, framework, config, prefix, directory, bs)
    framework.npu.synchronize()
    result = last_output.cpu()
    bits = result.view(torch.uint16).numpy()
    finite = bool(torch.isfinite(result).all())
    report = dict(
        kind="baseline_candidate_output_regression_not_golden",
        rank=int(os.environ["RANK"]), bs=bs, shape=list(result.shape), finite=finite,
        sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
    )
    np.save(directory.parent / "output_bits.npy", bits, allow_pickle=False)
    (directory.parent / "output_regression.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not finite:
        raise RuntimeError(f"rank={report['rank']} BS={bs}: output contains NaN/Inf")
    return samples


if __name__ == "__main__":
    api.fp8_fp4_mega_moe = remember_output
    benchmark.collect_profile = profile_and_archive
    benchmark.main()
