"""F10: the task log reports the trained parameters per group (full-size model, as in the plan)."""

import pytest

import engine
from models.configuration_deformable_detr import DeformableDetrConfig
from pdp_helpers import RPC_RESERVED, RPC_SIZES, make_trainer, pdp_args

pytestmark = pytest.mark.slow


def test_full_size_model_counts(tmp_path, monkeypatch):
    # default Deformable DETR (ResNet-50, 6+6 layers) without downloading ImageNet weights
    monkeypatch.setattr(engine, "DeformableDetrConfig", lambda: DeformableDetrConfig(use_pretrained_backbone=False))
    args = pdp_args(tmp_path, "--num_prompts", "100", "--prompt_len", "10", sizes=RPC_SIZES, reserved=RPC_RESERVED)
    trainer = make_trainer(args, task_id=1)  # resume() applies --freeze and logs the summary

    total, trainable, groups = trainer.log_trainable_parameters()
    assert round(total / 1e6, 2) == 69.12
    assert round(trainable / 1e6, 2) == 35.00
    by_group = {group for group, _ in groups}
    assert {"prompts", "class_embed", "input_proj", "bbox_embed", "query_position_embeddings",
            "reference_points", "level_embed"} <= by_group
    assert not by_group & {"backbone", "encoder", "decoder"}
    args.log_file.flush()
    log = (tmp_path / "log.txt").read_text()
    assert "Trainable parameters: 35.00M of 69.12M" in log
