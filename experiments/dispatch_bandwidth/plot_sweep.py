# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Redraw complete CROSS sweep JSON as one bandwidth-versus-core-count figure per H."""

import argparse
import math
from pathlib import Path
import sys

if __package__:
    from . import report
    from .sweep import DEFAULT_CORES, DEFAULT_HIDDEN, SweepError, _same, csv_integers, load_sweep, validate_case
else:
    import report
    from sweep import DEFAULT_CORES, DEFAULT_HIDDEN, SweepError, _same, csv_integers, load_sweep, validate_case


def verify_raw_reports(path, data):
    base = Path(path).resolve().parent
    expected = data["expected"]
    for case in data["cases"]:
        directory = (base / case["case_dir"]).resolve()
        if not directory.is_relative_to(base):
            raise SweepError("Case artifacts are outside the sweep directory")
        reports = report.load_reports(directory, expected["world"])
        rebuilt = report.summarize_reports(
            reports, expected_world=expected["world"], expected_iterations=expected["iterations"]
        )
        validate_case(dict(case, summary=rebuilt), expected)
        for field in ("configuration", "classes", "per_rank", "per_round", "checked_rows", "timing_validation"):
            _same(case["summary"].get(field), rebuilt[field], f"raw report reconstruction.{field}")


def plot_sweep(
    path,
    *,
    output_dir=None,
    hidden_values=DEFAULT_HIDDEN,
    core_values=DEFAULT_CORES,
    svg=False,
    target_gbps=134.0,
    verify_raw=False,
):
    # All input validation precedes importing a plotting backend or writing a figure.
    data = load_sweep(path, hidden_values=hidden_values, core_values=core_values)
    if not math.isfinite(target_gbps) or target_gbps <= 0:
        raise SweepError("Target line must be finite and positive")
    if verify_raw:
        verify_raw_reports(path, data)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected = data["expected"]
    by_shape = {(case["hidden"], case["cores"]): case["summary"]["classes"]["cross"] for case in data["cases"]}
    x_values = sorted(expected["core_values"])
    upper = max(target_gbps, max(value["rank_mean_gbps_per_card"]["max"] for value in by_shape.values())) * 1.1
    output_dir = Path(output_dir) if output_dir is not None else Path(path).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for hidden in expected["hidden_values"]:
        values = [by_shape[hidden, cores]["rank_mean_gbps_per_card"] for cores in x_values]
        means, lower, higher = ([value[key] for value in values] for key in ("mean", "min", "max"))
        figure, axis = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
        try:
            axis.fill_between(x_values, lower, higher, alpha=0.2, color="#2864b4", label="Rank min-max")
            axis.plot(x_values, means, marker="o", color="#2864b4", linewidth=2, label="Mean of rank means")
            axis.axhline(
                target_gbps,
                color="#777777",
                linestyle="--",
                linewidth=1.2,
                label=f"Target {target_gbps:g} GB/s/card (reference)",
            )
            row_bytes = (hidden + 255) // 256 * 256 + (hidden // 32 + 31) // 32 * 32
            title = (
                f"CROSS Dispatch bandwidth | H={hidden}  record={row_bytes} B\n"
                f"BS={expected['batch']}  EP={expected['world']}  group={expected['group_size']}  "
                f"K={expected['topk']}  E={expected['experts']}  I={expected['intermediate']}  "
                f"metadata={expected['metadata_mode']}  slots={expected['buffer_count']}"
            )
            if data["data_kind"] == "synthetic_test":
                title = "SYNTHETIC TEST DATA\n" + title
            axis.set_title(title, fontsize=12)
            axis.set_xlabel("Active CROSS workers (AIV1)")
            axis.set_ylabel("Effective CROSS GB/s per card")
            axis.set_xticks(x_values)
            axis.set_ylim(0, upper)
            if len(x_values) > 1:
                axis.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
            axis.grid(axis="both", alpha=0.2)
            axis.legend(loc="best", fontsize=9)
            figure.supxlabel(
                "Points: per-rank mean bandwidth, averaged over ranks; shaded band: rank min-max.\n"
                "Each rank uses its own class timing envelope; no cross-NPU clock subtraction.",
                fontsize=9,
            )
            for extension in ("png", "svg") if svg else ("png",):
                target = output_dir / f"cross_bandwidth_h{hidden}.{extension}"
                figure.savefig(target, dpi=160)
                if target.stat().st_size < 1000 or (
                    extension == "png" and target.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
                ):
                    raise SweepError(f"Plot export is empty or invalid: {target}")
                outputs.append(target)
        finally:
            plt.close(figure)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--hidden-values",
        type=csv_integers,
        default=DEFAULT_HIDDEN,
        help="Expected H matrix; override explicitly for a smaller experiment",
    )
    parser.add_argument("--core-values", type=csv_integers, default=DEFAULT_CORES)
    parser.add_argument("--target-gbps", type=float, default=134.0)
    parser.add_argument("--svg", action="store_true")
    parser.add_argument(
        "--verify-raw", action="store_true", help="Recompute each summary from its native rank JSON files"
    )
    args = parser.parse_args(argv)
    try:
        outputs = plot_sweep(
            args.sweep_json,
            output_dir=args.output_dir,
            hidden_values=args.hidden_values,
            core_values=args.core_values,
            svg=args.svg,
            target_gbps=args.target_gbps,
            verify_raw=args.verify_raw,
        )
    except (ImportError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"No successful sweep plot: {error}", file=sys.stderr)
        return 1
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
