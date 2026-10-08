# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Validate launch plans and build subprocess environments without SSH or devices."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/megamoe/scripts"
spec = importlib.util.spec_from_file_location("megamoe_batch_launch", SCRIPTS / "batch_launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    for key in ("SHARED_CODE", "BS_VALUES", "MASTER_PORT", "SHMEM_PORT", "SSH_PORT", "SSH_AUTH", "REMOTE_USER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "CONDA_BASE": "/opt/conda",
        "CONDA_ENV": "test env",
        "ASCEND_ENV": "/opt/cann/set_env.sh",
    }.items():
        monkeypatch.setenv(key, value)
    hosts = [f"192.0.2.{i}" for i in range(1, 5)]
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({"hosts": hosts, "master": hosts[2]}))
    return ["--hosts-file", str(path), "--ep-size", "32"]


def test_explicit_plan_preserves_rank_order_and_has_no_side_effects(configuration, monkeypatch, capsys):
    with monkeypatch.context() as guard:
        guard.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("SSH during plan"))
        guard.setattr(Path, "mkdir", lambda *a, **kw: pytest.fail("write during plan"))
        launcher.main([*configuration, "--plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["hosts"] == ["192.0.2.3", "192.0.2.1", "192.0.2.2", "192.0.2.4"]
    assert plan["root"] == str(ROOT)
    script = launcher.remote_script(plan, Path("/tmp/run"), 0, "prepare")
    assert "experiments/megamoe/scripts/batch_node.py" in script
    assert "conda activate 'test env'" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "hosts", [["192.0.2.1"] * 4, ["192.0.2.1"], ["invalid", "192.0.2.2", "192.0.2.3", "192.0.2.4"]]
)
def test_invalid_hosts_fail_before_launch(hosts):
    with pytest.raises(ValueError):
        launcher.select_hosts({"hosts": hosts}, 32)


def test_missing_environment_is_explicit(configuration, monkeypatch):
    monkeypatch.delenv("ASCEND_ENV")
    with pytest.raises(ValueError, match="ASCEND_ENV"):
        launcher.make_plan(launcher.parse_args(configuration))


def test_master_override_and_default(configuration):
    hosts = [f"192.0.2.{i}" for i in range(1, 5)]
    assert launcher.select_hosts({"hosts": hosts}, 32) == hosts
    plan = launcher.make_plan(launcher.parse_args([*configuration, "--master-ip", hosts[-1]]))
    assert plan["hosts"][0] == hosts[-1]
    with pytest.raises(ValueError, match="Master"):
        launcher.select_hosts({"hosts": hosts}, 32, "192.0.2.99")


@pytest.mark.parametrize("entry", ["build.py", "batch_node.py", "batch_launch.py"])
def test_relocated_entrypoints_work_in_isolated_python(entry):
    result = subprocess.run([sys.executable, "-I", str(SCRIPTS / entry), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_build_isolates_git_libraries_and_preserves_framework_environment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("megamoe_build", SCRIPTS / "build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    for name in ("SHMEM_ROOT", "CATLASS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "DEEPEP_NPU_ARCH": lock["target"],
        "DEEPEP_BUILD_MEGAMOE": "ON",
        "DEEPEP_MEGAMOE_TIMER": "ON",
        "LD_LIBRARY_PATH": "/opt/conda/lib:/opt/cann/lib64",
        "LD_PRELOAD": "/opt/conda/lib/libstdc++.so.6",
        "HTTPS_PROXY": "http://proxy.example:8080",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(builder.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(builder, "cann_version", lambda version: None)
    original_environment = dict(os.environ)
    clones, queries, framework_checks = [], [], []

    class DependenciesPrepared(Exception):
        pass

    def check_environment(command, env):
        assert env["HTTPS_PROXY"] == original_environment["HTTPS_PROXY"]
        if command[0] == "git":
            assert "LD_LIBRARY_PATH" not in env
            assert "LD_PRELOAD" not in env
        else:
            for name in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
                assert env[name] == original_environment[name]

    def run(command, *, cwd, env, check):
        check_environment(command, env)
        assert check
        if command[0] == "git":
            recursive = [] if command[-1] == "third-party/catlass" else ["--recursive"]
            assert command[1:-1] == ["submodule", "update", "--init", *recursive, "--"]
            clones.append(command[-1])
        elif command[1:4] == ["-m", "pip", "install"]:
            # Stop before package installation or compilation; only dependency preparation is exercised.
            raise DependenciesPrepared
        else:
            assert command[:2] == [sys.executable, "-c"]
            framework_checks.append(command)

    def check_output(command, *, env, text):
        check_environment(command, env)
        assert command[:2] == ["git", "-C"] and text
        queries.append(command)
        if command[3:] == ["rev-parse", "HEAD"]:
            return (
                lock["shmem"]["revision"] if Path(command[2]).name == "shmem" else lock["megamoe"]["catlass_revision"]
            )
        assert command[3:] == ["status", "--porcelain", "--untracked-files=normal"]
        return ""

    monkeypatch.setattr(builder.subprocess, "run", run)
    monkeypatch.setattr(builder.subprocess, "check_output", check_output)
    with pytest.raises(DependenciesPrepared):
        builder.build(tmp_path)
    assert clones == ["third-party/shmem", "third-party/catlass"]
    assert len(queries) == 4
    assert len(framework_checks) == 1
    assert dict(os.environ) == original_environment


def test_local_plan_needs_no_remote_environment(monkeypatch, capsys):
    for key in ("CONDA_BASE", "CONDA_ENV", "ASCEND_ENV", "SHARED_CODE", "BS_VALUES", "MASTER_PORT", "SHMEM_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SSH_AUTH", "invalid-unused")
    monkeypatch.setenv("SSH_PORT", "invalid-unused")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("process during plan"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: pytest.fail("write during plan"))
    launcher.main(["--ep-size", "8", "--bs-values", "96", "--plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["local"] and plan["hosts"] == ["127.0.0.1"]
    assert plan["timer"] and plan["parse_timer"]
    assert plan["config"]["bs_values"] == [96]
    script = launcher.remote_script(plan, Path("/tmp/run"), 0, "run", 96)
    assert "conda activate" not in script
    assert "cd /tmp" in script and "unset PYTHONHOME PYTHONPATH" in script


@pytest.mark.parametrize("extra", [["--hosts-file", "unused.json"], ["--master-ip", "192.0.2.1"]])
def test_local_rejects_remote_targets(extra):
    with pytest.raises(ValueError, match="EP8 runs on this machine"):
        launcher.make_plan(launcher.parse_args(["--ep-size", "8", *extra]))


def test_multinode_still_requires_hosts():
    with pytest.raises(ValueError, match="requires --hosts-file"):
        launcher.make_plan(launcher.parse_args([]))


def test_compute_replay_rejects_accuracy_and_shared_experts(tmp_path):
    with pytest.raises(ValueError, match="Compute-replay"):
        launcher.make_plan(launcher.parse_args(["--ep-size", "8", "--mode", "accuracy"]))
    config = json.loads((ROOT / "experiments/megamoe/python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro.json").read_text())
    config["model"]["num_shared_experts"] = 1
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="num_shared_experts=0"):
        launcher.make_plan(launcher.parse_args(["--ep-size", "8", "--config", str(path)]))


def test_compute_replay_summary_rejects_full_moe_results(tmp_path, capsys):
    from deep_ep_experimental.megamoe.performance import summarize_rounds

    report = summarize_rounds([[4000.0, 3900.0, 3800.0, 3700.0]] * 8, 8)
    path = tmp_path / "bs_4096/rank_0/bs_4096/profiling.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="performance report"):
        launcher.summarize(tmp_path, [4096], 8)
    report["metric"] = "megamoe_compute_replay_duration_us"
    path.write_text(json.dumps(report))
    launcher.summarize(tmp_path, [4096], 8)
    assert "MegaMoe compute replay=" in capsys.readouterr().out
    assert "E2E" not in (tmp_path / "summary.txt").read_text()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_local_stage_uses_bash_and_propagates_failure(tmp_path, monkeypatch, exit_code):
    plan = launcher.make_plan(launcher.parse_args(["--ep-size", "8", "--bs-values", "96"]))
    (tmp_path / "logs").mkdir()
    import io

    commands = []

    class Process:
        stdin = io.BytesIO()

        def poll(self):
            return exit_code

    def start(command, **kwargs):
        commands.append(command)
        assert kwargs["env"].get("SSHPASS") is None
        return Process()

    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    if exit_code:
        with pytest.raises(RuntimeError, match="run node=0 exit=7"):
            launcher.run_stage(plan, tmp_path, "run", bs=96)
    else:
        launcher.run_stage(plan, tmp_path, "run", bs=96)
    assert commands == [["bash", "-s"]]


def test_local_stage_executes_isolated_worker(tmp_path, monkeypatch):
    # 真正经过 bash/子进程，确认不是仅生成了一个正确外观的命令。
    script = tmp_path / "experiments/megamoe/scripts/batch_node.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "p = Path(sys.argv[sys.argv.index('--run-dir') + 1])\n"
        "(p / 'probe.json').write_text(json.dumps(dict(args=sys.argv[1:], "
        "isolated=sys.flags.isolated, cwd=os.getcwd(), pythonpath=os.getenv('PYTHONPATH'))))\n"
    )
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    plan = launcher.make_plan(launcher.parse_args(["--ep-size", "8", "--bs-values", "96"]))
    plan["root"] = str(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow"))
    launcher.run_stage(plan, run_dir, "run", bs=96)
    probe = json.loads((run_dir / "probe.json").read_text())
    assert probe["isolated"] == 1 and probe["pythonpath"] is None
    assert Path(probe["cwd"]).resolve() != tmp_path.resolve()
    assert probe["args"] == ["run", "--run-dir", str(run_dir), "--node-rank", "0", "--bs", "96"]


def test_local_commands_keep_eight_ranks_and_timer(tmp_path):
    spec = importlib.util.spec_from_file_location("megamoe_batch_node", SCRIPTS / "batch_node.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    plan = launcher.make_plan(launcher.parse_args(["--ep-size", "8", "--bs-values", "96"]))
    command = node.benchmark_command(plan, tmp_path, 0, 96)
    for option, value in (("--nnodes", "1"), ("--nproc-per-node", "8"),
                          ("--master-addr", "127.0.0.1"), ("--timer-ranks", "0-7")):
        assert command[command.index(option) + 1] == value
    assert "--timer" in command
    golden = node.golden_command(plan, tmp_path, 0, 96)
    assert golden[golden.index("--ep-size") + 1] == "8"
    assert golden[golden.index("--ranks") + 1] == "0-7"
    plan["timer"] = False
    assert "--timer" not in node.benchmark_command(plan, tmp_path, 0, 96)


@pytest.mark.parametrize("git_exit", [0, 7])
def test_shmem_nested_git_preserves_compiler_environment_and_cleans_up(tmp_path, monkeypatch, git_exit):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("shmem_build", ROOT / "scripts/build_shmem.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").touch()
    tools = tmp_path / "tools with spaces"
    tools.mkdir()
    q = shlex.quote
    probe = (
        "import json, os, pathlib, sys; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(env=dict(os.environ), args=sys.argv[2:])))"
    )
    git = tools / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = "-C" ]; then\n'
        '  if [ "$3" = "rev-parse" ]; then\n'
        f'    if [ "$4" = "HEAD" ]; then echo {q(lock["shmem"]["revision"])}; else echo {q(str(source))}; fi\n'
        "  fi\n  exit 0\nfi\n"
        f'{q(sys.executable)} -c {q(probe)} git-env.json "$@"\nexit {git_exit}\n'
    )
    git.chmod(0o700)
    (source / "scripts/build.sh").write_text(
        'git clone --branch v3.11.3 --depth 1 https://example.invalid/json.git "json target"\n'
        f"{q(sys.executable)} -c {q(probe)} compiler-env.json\n"
    )
    for name, value in {
        "PATH": str(tools) + os.pathsep + os.defpath,
        "LD_LIBRARY_PATH": "/fixture/conda/lib:/fixture/cann/lib64",
        "LD_PRELOAD": "libm.so.6",
        "https_proxy": "http://proxy.example:8080",
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(builder, "cann_version", lambda expected: {"Version": expected})
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    recorded = []

    def record(sdk, revision, **kwargs):
        recorded.append((sdk, revision))
        return sdk / "deepep-sdk.json"

    monkeypatch.setattr(builder, "record", record)
    original_environment = dict(os.environ)
    if git_exit:
        with pytest.raises(subprocess.CalledProcessError) as error:
            builder.prepare(tmp_path)
        assert error.value.returncode == git_exit
        assert not recorded
        assert not (source / "compiler-env.json").exists()
    else:
        builder.prepare(tmp_path)
        assert recorded == [(source / "install/shmem", lock["shmem"]["revision"])]
        compiler_env = json.loads((source / "compiler-env.json").read_text())["env"]
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "https_proxy", "CMAKE_BUILD_PARALLEL_LEVEL"):
            assert compiler_env[name] == original_environment[name]
    report = json.loads((source / "git-env.json").read_text())
    git_env = report["env"]
    assert "LD_LIBRARY_PATH" not in git_env and "LD_PRELOAD" not in git_env
    assert git_env["https_proxy"] == original_environment["https_proxy"]
    assert report["args"] == [
        "clone",
        "--branch",
        "v3.11.3",
        "--depth",
        "1",
        "https://example.invalid/json.git",
        "json target",
    ]
    assert not Path(git_env["PATH"].split(os.pathsep)[0]).exists()
    assert dict(os.environ) == original_environment
