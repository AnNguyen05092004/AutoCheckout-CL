"""B1: full fine-tuning without prompts — joint training (E0) and a Hugging Face export for FSA (I1)."""

import pytest
import torch

from autocheckout.predictions import load_predictions

from pdp_helpers import make_trainer, pdp_args, run_main, use_tiny_detr

pytestmark = pytest.mark.slow
FINETUNE = ("--use_prompts", "0", "--local_query", "0", "--pseudo", "none", "--freeze", "",
            "--optim_groups", "detr", "--lr_backbone_names", "backbone", "--lr", "1e-4")


def test_detr_learning_rate_groups(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path, *FINETUNE), task_id=1)
    groups = trainer.configure_optimizers()["optimizer"].param_groups
    names = {id(p): n for n, p in trainer.named_parameters()}
    lr_of = {names[id(p)]: g["lr"] for g in groups for p in g["params"]}
    assert all(lr == 1e-5 for n, lr in lr_of.items() if "backbone" in n)
    assert all(lr == pytest.approx(1e-5) for n, lr in lr_of.items() if "sampling_offsets" in n)
    assert lr_of["model.class_embed.0.weight"] == 1e-4
    # everything is trained except the backbone stem and layer1, frozen by Deformable DETR itself
    untrained = {n for n, p in trainer.named_parameters() if n not in lr_of}
    assert untrained and all("backbone" in n and ("conv1" in n or "layer1" in n) for n in untrained)


def test_joint_training_is_one_task_over_every_class(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, *FINETUNE, "--joint", "1")
    run = tmp_path / "run"
    assert not (run / "task_1").exists()
    preds = load_predictions(run / "task_2" / "pred_test.npz")
    assert preds.meta["seen_classes"] == 5 and preds.label.max() < 5


def test_hf_export_loads_as_the_start_of_pdp_task_1(tmp_path, monkeypatch):
    run_main(tmp_path / "fsa", monkeypatch, *FINETUNE, "--n_tasks", "1", "--save_hf", "1")
    hf_dir = tmp_path / "fsa" / "run" / "task_1" / "hf_model"
    assert (hf_dir / "config.json").exists() and (hf_dir / "preprocessor_config.json").exists()

    (tmp_path / "pdp").mkdir()
    args = pdp_args(tmp_path / "pdp", "--repo_name", str(hf_dir))
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(args, task_id=1)
    saved = torch.load(tmp_path / "fsa" / "run" / "task_1" / "task_final.pth", map_location="cpu")["model"]
    for key in ("model.input_proj.0.0.weight", "class_embed.0.weight", "model.encoder.layers.0.fc1.weight"):
        torch.testing.assert_close(trainer.model.state_dict()[key], saved[key])
    assert hasattr(trainer.model.model, "prompts")  # PDP adds its prompts on top of the FSA weights
