"""F7: prototypes are built only from queries that classify their matched object correctly."""

import pytest
import torch

from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow


def fixed_matcher(outputs, targets):
    """Query 5 <-> first object, query 7 <-> second object of every image."""
    return [(torch.tensor([5, 7][:len(t["class_labels"])]), torch.arange(len(t["class_labels"])))
            for t in targets]


@pytest.mark.parametrize("correct_only, expected_cached", [(1, {1}), (0, {0, 1})])
def test_only_correctly_classified_queries_enter_memory(tmp_path, monkeypatch, correct_only, expected_cached):
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path, "--proto_correct_only", str(correct_only), "--epochs", "1", "--ddl_lambda", "0")
    trainer = make_trainer(args, task_id=1)
    trainer.train()
    monkeypatch.setattr(trainer.model.matcher, "forward", fixed_matcher)
    with torch.no_grad():  # every query predicts class 1
        head = trainer.model.class_embed[0]
        head.weight.zero_()
        head.bias.fill_(-10.0)
        head.bias[1] = 10.0

    batch = make_batch([[0, 1]])
    for step in range(2):  # the memory is updated every second batch of the last epoch
        trainer.common_step(batch, step)

    cached = {c for c, n in trainer.class_cache_count.items() if n > 0}
    assert cached == expected_cached
    assert trainer.class_prototypes[1].norm() > 0
    assert trainer.missing_prototypes() == sorted({0, 1, 2} - expected_cached)
