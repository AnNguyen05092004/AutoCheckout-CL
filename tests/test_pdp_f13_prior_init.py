"""F13: the classifier starts at the focal-loss prior unless it is loaded from the checkpoint."""

import math

import pytest
import torch

from models.configuration_deformable_detr import DeformableDetrConfig
from models.image_processing_deformable_detr import DeformableDetrImageProcessor
from models.modeling_deformable_detr import DeformableDetrForObjectDetection
from pdp_helpers import TINY_DETR, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow
PRIOR = -math.log(0.99 / 0.01)


def saved_model(path, num_labels):
    config = DeformableDetrConfig(**TINY_DETR)
    config.num_labels, config.use_prompts = num_labels, 0
    model = DeformableDetrForObjectDetection(config)
    with torch.no_grad():
        model.class_embed[0].bias.fill_(0.3)  # a "trained" value
    model.save_pretrained(path)
    DeformableDetrImageProcessor().save_pretrained(path)
    return str(path)


def class_bias(trainer):
    return trainer.model.class_embed[0].bias.detach()


@pytest.mark.parametrize("flag", [1, 0])
def test_new_classifier_gets_the_prior(tmp_path, monkeypatch, flag):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path, "--prior_init_classifier", str(flag)), task_id=1)
    expected = PRIOR if flag else 0.0  # without the fix Hugging Face's init leaves 0 (p = 0.5)
    torch.testing.assert_close(class_bias(trainer), torch.full_like(class_bias(trainer), expected))


def test_checkpoint_with_another_class_count_gets_the_prior(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    repo = saved_model(tmp_path / "coco_like", num_labels=4)  # differs from the 6 outputs below
    trainer = make_trainer(pdp_args(tmp_path, "--repo_name", repo), task_id=1)
    assert torch.allclose(class_bias(trainer), torch.tensor(PRIOR))


def test_classifier_of_the_same_size_is_kept(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    repo = saved_model(tmp_path / "fsa_like", num_labels=6)  # sizes (3, 2) -> 5 slots + 1
    trainer = make_trainer(pdp_args(tmp_path, "--repo_name", repo), task_id=1)
    assert torch.allclose(class_bias(trainer), torch.tensor(0.3))
