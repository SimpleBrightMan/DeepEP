# Copyright (c) 2026, SimpleBright_Man

"""Full-output regression against the unchanged production schedule."""

import json
from pathlib import Path


def signature(config, rank, world, bs, input_kind, routing):
    return dict(
        model=config["model"],
        seed=config["seed"],
        rank=rank,
        world=world,
        bs=bs,
        input=input_kind,
        routing=routing,
        numerical_contract="ascend_post_gmm2_route_weight_v1",
    )


def expected_counts(spec):
    import numpy as np

    if spec["routing"] != "balanced":
        raise ValueError("Core sweep baseline verification requires balanced routing")
    model, world = spec["model"], spec["world"]
    local = model["num_experts"] // world
    routes = np.arange(world * spec["bs"] * model["num_topk"], dtype=np.int64)
    selected = routes[routes % world == spec["rank"]]
    return np.bincount((selected // world) % local, minlength=local)


def verify(actual, counts, spec, reference=None, *, rtol=0.0, atol=0.0):
    import numpy as np
    from .reference import error_metrics

    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Output comparison tolerances must be finite and nonnegative")
    if actual.shape != (spec["bs"], spec["model"]["hidden"]):
        raise ValueError("Unexpected output shape")
    counts_equal = bool(np.array_equal(counts, expected_counts(spec)))
    report = dict(
        rank=spec["rank"],
        finite=bool(np.isfinite(actual).all()),
        counts_equal=counts_equal,
        checked_tokens=spec["bs"],
        rtol=rtol,
        atol=atol,
        reference="unchanged_production_schedule" if reference else "finite_and_route_counts_only",
    )
    report["passed"] = report["finite"] and counts_equal
    if reference is not None:
        path = Path(reference) / f"rank_{spec['rank']}" / f"bs_{spec['bs']}" / "output_reference.npz"
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["signature"])) != spec:
                raise ValueError(f"Baseline signature mismatch: {path}")
            expected, baseline_counts = saved["output"], saved["counts"]
        report.update(error_metrics(actual, expected))
        report["baseline_counts_equal"] = bool(np.array_equal(counts, baseline_counts))
        report["output_equal"] = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
        report["passed"] &= report["finite"] and report["baseline_counts_equal"] and report["output_equal"]
    return report


def save(path, actual, counts, spec):
    import numpy as np

    np.savez(path, output=actual, counts=counts, signature=json.dumps(spec, sort_keys=True))
