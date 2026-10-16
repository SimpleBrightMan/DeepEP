# Copyright (c) 2026, SimpleBright_Man

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "hierarchical_bench", ROOT / "experiments/megamoe/scripts/hierarchical_bench.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_plan_no_device_access_or_output(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan must not execute anything")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    target = tmp_path / "absent"
    launcher.main(["--plan", "--output-dir", str(target)])
    document = json.loads(capsys.readouterr().out)
    assert document["bs_values"] == [2048, 4096, 8192]
    assert len(document["commands"]) == 3 and not target.exists()
    for command in document["commands"]:
        assert "--nproc-per-node=16" in command
        assert "deep_ep_experimental.megamoe.hierarchy_benchmark" in command


@pytest.mark.parametrize(
    "option,value",
    [
        ("--bs-values", "256"),
        ("--bs-values", "4097"),
        ("--bs-values", "2048,2048"),
        ("--bs-values", "8193"),
        ("--bs-values", "x"),
        ("--hidden", "3712"),
        ("--intermediate", "1290"),
        ("--experts", "385"),
        ("--topk", "33"),
        ("--warmup", "-1"),
        ("--iters", "0"),
        ("--relative-rmse", "nan"),
        ("--max-abs", "-0.01"),
    ],
)
def test_reject_invalid_shape_or_tolerance(option, value):
    with pytest.raises(SystemExit):
        launcher.parse_args([option, value])


def test_command_forwards_full_model_and_accuracy_configuration(tmp_path):
    args = launcher.parse_args(
        [
            "--bs-values",
            "384",
            "--hidden",
            "4096",
            "--intermediate",
            "1536",
            "--experts",
            "256",
            "--topk",
            "6",
            "--routing",
            "random",
            "--input",
            "bf16",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--relative-rmse",
            "0.005",
            "--max-abs",
            "0.02",
            "--output-dir",
            str(tmp_path),
        ]
    )
    command = launcher.command(args, 384)
    expected = {
        "--hidden": "4096",
        "--intermediate": "1536",
        "--experts": "256",
        "--topk": "6",
        "--routing": "random",
        "--input": "bf16",
        "--warmup": "1",
        "--iters": "2",
        "--relative-rmse": "0.005",
        "--max-abs": "0.02",
        "--bs": "384",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert Path(command[command.index("--output") + 1]) == tmp_path / "bs_384"
    assert "--timer" not in command


def test_does_not_inherit_other_branch_core_configuration():
    original = {name: "12" for name in launcher.CORE_ENV}
    original["SHMEM_ROOT"] = "/sdk"
    original["PYTHON_EXEC"] = "/old/python"
    original["PYTHONPATH"] = "/cann/site-packages"
    cleaned = launcher.environment(original)
    assert all(name not in cleaned for name in launcher.CORE_ENV)
    assert cleaned["SHMEM_ROOT"] == "/sdk" and cleaned["OMP_NUM_THREADS"] == "1"
    assert all(name in original for name in launcher.CORE_ENV)
    assert cleaned["PYTHON_EXEC"] == sys.executable
    assert cleaned["PYTHONPATH"] == original["PYTHONPATH"]


def test_install_check_matches_worker_environment_and_reports_missing_module(tmp_path, monkeypatch, capsys):
    def run(argv, **kwargs):
        assert argv[:2] == [sys.executable, "-c"]
        assert kwargs["cwd"] == tmp_path and kwargs["env"] == launcher.environment()
        assert "hierarchy_benchmark" in argv[2] and "configure_hierarchy" in argv[2]
        return SimpleNamespace(returncode=1, stdout='{"passed":false,"error":"missing hierarchy_benchmark"}', stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="before starting any ranks"):
        launcher.check_installation(tmp_path)
    assert "missing hierarchy_benchmark" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["runner", "native", None])
def test_install_probe_checks_actual_imports_without_npu_calls(tmp_path, missing):
    modules = {
        "deep_ep/__init__.py": "",
        "deep_ep/_C.py": "",
        "deep_ep_experimental/__init__.py": "",
        "deep_ep_experimental/megamoe/__init__.py": "",
        "torch.py": "",
        "torch_npu.py": "",
        "deep_ep_experimental/megamoe/hierarchy_benchmark.py": "",
        "deep_ep_experimental/megamoe/_C.py": (
            "class Plan:\n    configure_hierarchy = None\nhierarchy_buffer_size = None\n"
            if missing != "native"
            else "class Plan: pass\n"
        ),
    }
    if missing == "runner":
        modules.pop("deep_ep_experimental/megamoe/hierarchy_benchmark.py")
    for relative, source in modules.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    result = subprocess.run(
        [sys.executable, "-c", launcher.INSTALL_PROBE], cwd=tmp_path, capture_output=True, text=True
    )
    report = json.loads(result.stdout)
    assert report["passed"] is (missing is None)
    assert result.returncode == (0 if missing is None else 1)
    if missing:
        assert "hierarchy" in report["error"] or "hierarchical" in report["error"]


def make_case(tmp_path):
    args = launcher.parse_args(["--bs-values", "384", "--iters", "2", "--output-dir", str(tmp_path)])
    directory = tmp_path / "bs_384"
    directory.mkdir()
    timing = dict(ranks=16, rounds=2, acl_span_us=dict(rank_max_mean=100.0), wall_us=dict(rank_max_mean=110.0))
    report = dict(
        schema_version=1,
        prototype="staged_hierarchical_dedup_fp32_partial",
        config={
            name: getattr(args, name)
            for name in (
                "hidden",
                "intermediate",
                "experts",
                "topk",
                "routing",
                "input",
                "warmup",
                "iters",
                "seed",
                "relative_rmse",
                "max_abs",
            )
        },
        accuracy=[dict(rank=rank, passed=True, counts_equal=True) for rank in range(16)],
        repeat_accuracy=[dict(passed=True) for _ in range(16)],
        dynamic_accuracy=[dict(passed=True) for _ in range(16)],
        timer=False,
        plan_wall_ms=[1.0] * 16,
        direct=timing,
        hierarchy=timing,
        stage_payload_gbps={stage: dict(mean=0.1, minimum=0.0, maximum=0.2) for stage in launcher.STAGES},
        rebuild_every_call_estimate_us=1110.0,
    )
    report["config"]["bs"] = 384
    for rank in range(16):
        rank_dir = directory / f"rank_{rank}"
        rank_dir.mkdir()
        samples = [dict(acl_span_us=100.0, wall_us=110.0, stages=dict(dispatch_cross_bytes=3712)) for _ in range(2)]
        local = dict(
            rank=rank, direct_samples=samples, hierarchy_samples=samples, oracle=dict(dispatch_cross_bytes=3712)
        )
        (rank_dir / "results.json").write_text(json.dumps(local))
    (directory / "summary.json").write_text(json.dumps(report))
    return args, report


def test_case_validation_and_csv_export(tmp_path):
    args, report = make_case(tmp_path)
    result = launcher.read_case(args, 384)
    assert result == report
    launcher.write_summary(tmp_path, launcher.plan(args), [result])
    csv = (tmp_path / "summary.csv").read_text()
    assert "hierarchy_acl_span_us" in csv and "combine_cross_stage_GBps" in csv
    assert json.loads((tmp_path / "summary.json").read_text())["cases"][0]["config"]["bs"] == 384


@pytest.mark.parametrize(
    "corruption", ["missing_rank", "failed_count", "repeat", "dynamic", "shape", "timer", "payload", "time"]
)
def test_reject_incomplete_or_failed_results(tmp_path, corruption):
    args, report = make_case(tmp_path)
    if corruption == "missing_rank":
        report["accuracy"].pop()
    elif corruption == "failed_count":
        report["accuracy"][0]["counts_equal"] = False
    elif corruption == "repeat":
        report["repeat_accuracy"][0]["passed"] = False
    elif corruption == "dynamic":
        report["dynamic_accuracy"].pop()
    elif corruption == "shape":
        report["config"]["hidden"] = 4096
    elif corruption == "timer":
        report["timer"] = True
    elif corruption == "time":
        report["hierarchy"]["acl_span_us"]["rank_max_mean"] = float("nan")
    else:
        path = tmp_path / "bs_384" / "rank_15" / "results.json"
        local = json.loads(path.read_text())
        local["hierarchy_samples"][1]["stages"]["dispatch_cross_bytes"] = 0
        path.write_text(json.dumps(local))
    (tmp_path / "bs_384" / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        launcher.read_case(args, 384)
