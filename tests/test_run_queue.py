"""scripts/run_queue.sh: runs the queued experiments in order, re-reads the queue, never re-runs a finished
experiment, carries on after a failed one, stops without marking an interrupted one finished, and always shuts
down at the end.

Experiments go through the real scripts/run_exp.sh with the fake interpreter of test_run_exp.py.
"""

import os
import subprocess
from pathlib import Path

from conftest import REPO_ROOT
from test_run_exp import FAKE_PYTHON


def config(name: str, extra: str = "") -> str:
    return f"""source "$REPO/configs/exp/common.sh"
EXP={name}
N_TASKS=1
ARGS=("${{COMMON_ARGS[@]}}" --epochs 1)
{extra}"""


def run_queue(tmp_path: Path, queue: str, configs: dict[str, str], log: str = "", returncode: int = 0) -> list[str]:
    """Runs the queue; returns the experiments whose training was started, in order."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "queue.txt").write_text(queue)
    (runs / "queue.log").write_text(log)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name, text in configs.items():
        (config_dir / f"{name}.sh").write_text(text)
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "HOME": str(tmp_path), "PYTHON": str(fake), "CALLS": str(calls), "RUNS": str(runs),
           "DATA": str(tmp_path / "data"), "CONFIG_DIR": str(config_dir),
           "SHUTDOWN_CMD": f"touch {tmp_path / 'shutdown'}"}
    result = subprocess.run(["bash", str(REPO_ROOT / "scripts/run_queue.sh")], env=env, capture_output=True,
                            text=True)
    assert result.returncode == returncode, result.stderr
    assert (tmp_path / "shutdown").exists()
    started = [line.split("--output_dir ")[1].split()[0] for line in calls.read_text().splitlines()
               if line.startswith("main.py")] if calls.exists() else []
    return [Path(d).name for d in started]


def log_lines(tmp_path: Path) -> list[str]:
    return [" ".join(line.split()[:2]) for line in (tmp_path / "runs/queue.log").read_text().splitlines()]


def test_runs_in_order_rereads_the_queue_and_skips_finished(tmp_path):
    appends_third = 'grep -q third "$RUNS/queue.txt" || echo third >> "$RUNS/queue.txt"'
    queue = "# pilot\nold\nfirst\n\nsecond  words after the name are ignored\n"
    configs = {"old": config("old"), "first": config("first", appends_third), "second": config("second"),
               "third": config("third")}
    assert run_queue(tmp_path, queue, configs, log="old exit=0 2026-09-28 10:00:00\n") == ["first", "second", "third"]
    assert log_lines(tmp_path) == ["old exit=0", "first exit=0", "second exit=0", "third exit=0",
                                   "queue empty", "queue stopped"]
    assert (tmp_path / "runs/first.log").read_text().startswith("== first: 1 tasks")


def test_failed_experiment_is_logged_and_the_queue_goes_on(tmp_path):
    configs = {"bad": config("bad", "export FAIL_TASK=1"), "good": config("good")}
    assert run_queue(tmp_path, "bad\ngood\n", configs) == ["bad", "good"]
    lines = log_lines(tmp_path)
    assert lines[0].startswith("bad exit=") and lines[0] != "bad exit=0"
    assert lines[1:] == ["good exit=0", "queue empty", "queue stopped"]


def test_empty_queue_just_shuts_down(tmp_path):
    assert run_queue(tmp_path, "", {}) == []
    assert log_lines(tmp_path) == ["queue empty", "queue stopped"]


def test_interrupted_experiment_stops_the_queue_and_is_not_marked_finished(tmp_path):
    configs = {"cut": config("cut", "kill -TERM $$"), "next": config("next")}  # run_exp.sh killed by SIGTERM
    assert run_queue(tmp_path, "cut\nnext\n", configs, returncode=143) == []
    assert log_lines(tmp_path) == ["cut interrupted", "queue stopped"]
