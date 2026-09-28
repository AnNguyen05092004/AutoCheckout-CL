"""F4: the query loss L_Q back-propagates into query_tf (it was computed under no_grad)."""

import pytest

from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("query_loss_grad", [1, 0])
def test_query_loss_trains_query_tf(tmp_path, monkeypatch, query_loss_grad):
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path, "--query_loss_grad", str(query_loss_grad), "--ddl_lambda", "0")
    trainer = make_trainer(args, task_id=1)
    trainer.train()

    _, loss_dict = trainer.common_step(make_batch([[0, 1], [2]]), 0)
    query_loss = loss_dict["query_loss"]
    assert query_loss.requires_grad == bool(query_loss_grad)

    if query_loss_grad:
        query_loss.backward()  # L_Q alone
        weight = trainer.model.model.prompts.query_tf[0].weight
        assert weight.grad is not None and weight.grad.abs().sum() > 0
