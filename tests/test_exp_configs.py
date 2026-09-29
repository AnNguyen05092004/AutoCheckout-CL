"""Every experiment config sources cleanly and its ARGS parse with pdp/main.py's parser."""

import subprocess

import pytest
from conftest import REPO_ROOT

from pdp_helpers import main_args

# Pilot configs keep the effective batch they were run with on 28/09; everything else uses 4.
PILOT_EFF_BATCH_32 = {"P1", "P2", "P3", "FSA_pilot"}
CONFIGS = sorted(p for p in (REPO_ROOT / "configs/exp").glob("*.sh") if p.name != "common.sh")


def sourced(config):
    script = f'REPO="{REPO_ROOT}"; DATA=/data/rpc; RUNS=/data/runs; source "{config}"; ' \
             'printf "%s\\0" "$EXP" "${N_TASKS}" "${START_TASK:-1}" "${REUSE_TASK1:-}" "${ARGS[@]}"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    exp, n_tasks, start, reuse, *args = out.split("\0")[:-1]
    return exp, int(n_tasks), int(start), reuse, args


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_config_parses(config):
    exp, n_tasks, start, reuse, args = sourced(config)
    assert exp == config.stem and 1 <= start <= n_tasks <= 5
    parsed = main_args(args)
    assert parsed.n_gpus == 1 and parsed.accelerator == "gpu" and parsed.lr == 1e-4
    assert parsed.eff_batch_size == (32 if exp in PILOT_EFF_BATCH_32 else 4)
    if parsed.use_prompts:
        assert parsed.n_classes == 225 and parsed.freeze == "backbone,encoder,decoder"
    else:
        assert parsed.optim_groups == "detr" and parsed.lr_backbone_names == ["backbone"]
    if reuse:
        assert reuse == "E4" and parsed.repo_name.endswith("FSA/task_1/hf_model")


def test_expected_experiments_exist():
    names = {p.stem for p in CONFIGS}
    assert {"P1", "P2", "P3", "FSA_pilot", "E0", "FSA", "DET", "E1", "E2", "E3", "E3_coco", "E4"} <= names
    assert {f"A{i}" for i in range(1, 10)} <= names


def test_method_comparison_starts_from_fsa_and_e3_coco_from_the_paper_checkpoint():
    for name in ("E1", "E2", "E3", "E4"):
        args = sourced(REPO_ROOT / f"configs/exp/{name}.sh")[4]
        assert main_args(args).repo_name == "/data/runs/FSA/task_1/hf_model", name
    args = sourced(REPO_ROOT / "configs/exp/E3_coco.sh")[4]
    assert main_args(args).repo_name == "SenseTime/deformable-detr"
