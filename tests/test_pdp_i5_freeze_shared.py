"""I5: from task 2 on, the parameters shared by all tasks can be frozen; prompts and classifier still train."""

import pytest

from pdp_helpers import make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow
SHARED = {"input_proj", "query_position_embeddings", "reference_points", "level_embed", "bbox_embed"}


@pytest.mark.parametrize("flag, task_id, shared_trained", [(0, 2, True), (1, 1, True), (1, 2, False)])
def test_shared_parameters_frozen_only_after_task_1(tmp_path, monkeypatch, flag, task_id, shared_trained):
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path, "--freeze_shared_after_task1", str(flag))
    trainer = make_trainer(args, task_id=task_id)
    _, _, groups = trainer.log_trainable_parameters()
    trained = {group for group, _ in groups}
    assert {"prompts", "class_embed"} <= trained
    assert (SHARED <= trained) == shared_trained and (not SHARED & trained) == (not shared_trained)
    query_tf = trainer.model.model.prompts.query_tf[0].weight
    assert query_tf.requires_grad == shared_trained
