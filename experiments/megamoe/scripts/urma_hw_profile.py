"""EP4/8/16 external msopprof entry; uses the installed MegaMoE URMA kernel.

No torch_npu profiler, Events, internal timer/trace, or kernel-only replay.
Run through run_urma_hw_profile.sh, not directly.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import sys


def capture_forward(forward, guard):
    """One MegaMoE call per rank per application replay; no warmup or MSTX."""
    guard.begin()
    guard.before_forward()
    forward()
    guard.end()


def run(args, group):
    import torch
    import deep_ep_experimental.megamoe as package
    from deep_ep_experimental.megamoe.api import transform_weights_for_mega_moe
    from deep_ep_experimental.megamoe.benchmark import inputs, load_config
    from deep_ep_experimental.megamoe.reference import make_weights, quantize_input
    from deep_ep_experimental.megamoe.urma import UrmaBuffer, _extension
    from deep_ep_experimental.megamoe.urma_timing import UrmaLaunchGuard
    from deep_ep_experimental.megamoe.validation import collective_check

    world = group.size()
    if world not in (4, 8, 16) or int(os.environ["LOCAL_WORLD_SIZE"]) != world:
        raise ValueError("One host with EP4/8/16 allocated devices is required")
    config = load_config("urma", str(args.bs))
    model = config["model"]
    expected = dict(hidden=3584, intermediate_hidden=1280, num_experts=384,
                    num_topk=8, num_shared_experts=0)
    if model != expected:
        raise ValueError(f"Unexpected installed urma config: {model}")

    rank = group.rank()
    directory = args.output_dir / f"rank_{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    collective_check(group, "hardware profiling options", None,
                     (config, args.exp_repeat, args.exp_shape))
    owner = UrmaBuffer(group, **model, num_max_tokens_per_rank=args.bs,
                       cv_rows=args.cv_rows, gmm1_ahead_mgroups=12)

    def device_pair(pair):
        data, scale = pair
        return torch.from_numpy(data).npu(), torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu)

    raw = make_weights(model["num_experts"] // world, model["hidden"],
                       model["intermediate_hidden"], config["seed"] + rank)
    weights = transform_weights_for_mega_moe(*map(device_pair, raw))
    del raw
    guard = UrmaLaunchGuard(group, args.exp_shape, args.exp_repeat, args.timeout)
    host_x, host_ids, host_weights = inputs(config, rank, args.bs, args.routing, world=world)
    codes, scales = quantize_input(host_x)
    x = (torch.from_numpy(codes).npu().view(torch.float8_e4m3fn),
         torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu))
    ids = torch.from_numpy(host_ids).npu()
    topk_weights = torch.from_numpy(host_weights).npu()
    output = torch.empty((args.bs, model["hidden"]), dtype=torch.bfloat16, device="npu")
    prepared = owner.prepare_routes(ids)
    forward = partial(owner.forward_prepared, x, topk_weights, *weights, prepared,
                      out=output, timer=None, trace=None)
    guard.prepare_per_call_alignment()
    manifest = dict(
        status="READY", rank=rank, local_rank=int(os.environ["LOCAL_RANK"]),
        pid=os.getpid(), python=sys.executable, package=str(package.__file__),
        created_utc=datetime.now(timezone.utc).isoformat(), config=config,
        bs=args.bs, ep=world, cv_rows=args.cv_rows, gmm1_ahead_mgroups=12,
        warmup=0, total_forward_calls=1, selected_forward_calls=1,
        internal_timer=False, internal_trace=False, nested_profiler=False,
        launch_guard=guard.describe(), mstx_enabled=False,
        native_schedule=getattr(_extension(), "urma_compute_schedule", None),
        native_tail=getattr(_extension(), "urma_tail_schedule", None),
        shmem_endpoint=os.environ["DEEPEP_SHMEM_ENDPOINT"],
        numerical_validation="not performed; capture entry only",
    )
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.npu.synchronize()
    group.barrier("hardware profile ready")
    if rank == 0:
        print(f"[HW PROFILE] EP{world} BS{args.bs} H3584 I1280 E384 K8; "
              "warmup=0; total=1; selected=1; MSTX off; timer/trace off; "
              "HCCL barrier + exp retained", flush=True)
    capture_forward(forward, guard)
    group.barrier("hardware profile forward completed")
    # Do not report accuracy or a hardware trace as successful based on execution
    # alone. The external tool owns the capture/export status and output files.
    manifest["status"] = "FORWARD_COMPLETED"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del forward, prepared
    owner.destroy()
    guard.close()
    group.close()
    print(f"[HW PROFILE] rank={rank} forward/teardown completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--cv-rows", type=int, default=256)
    parser.add_argument("--routing", choices=("balanced", "random-balanced"), default="random-balanced")
    parser.add_argument("--exp-repeat", type=int, default=50)
    parser.add_argument("--exp-shape", nargs=2, type=int, default=[100000, 20000])
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.exp_repeat < 0 or min(args.exp_shape) < 1 or args.timeout < 1:
        parser.error("invalid exp/timeout")
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise RuntimeError("Launch through run_urma_hw_profile.sh (TASK_QUEUE_ENABLE=0 required)")
    if not os.environ.get("MSOPPROF_EXE_PATH"):
        raise RuntimeError("Missing msopprof injection environment; use the shell launcher")
    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend
    from deep_ep._store import StoreGroup

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env(timeout=args.timeout)
    try:
        run(args, group)
    except BaseException as error:
        group.abort(f"hardware profile failed: {type(error).__name__}: {error}")
        # Avoid a second collective teardown while another worker is failing.
        raise


if __name__ == "__main__":
    main()
