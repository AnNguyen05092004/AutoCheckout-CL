"""B2: either prompt pool can be switched off (E1, ablations A2/A3)."""

from types import SimpleNamespace

import pytest
import torch

from models.prompt import Prompt


def make_prompt(shared, private):
    torch.manual_seed(0)
    args = SimpleNamespace(local_query=0, use_shared_pool=shared, use_private_pool=private)
    return Prompt(emb_d=16, n_tasks=2, prompt_param=[4, 2, 0], key_dim=16, args=args, task_num_classes=[3, 2])


@pytest.mark.parametrize("shared, private", [(1, 1), (1, 0), (0, 1)])
def test_only_enabled_pools_are_used_and_trained(shared, private):
    prompt = make_prompt(shared, private)
    (ek, ev), _, _ = prompt(torch.randn(2, 16), 0, None, train=True)
    (ek.sum() + ev.pow(2).sum()).backward()
    assert ek.shape == (2, 1, 16)
    shared_grad, private_grad = prompt.shared_p_0.grad, prompt.private_p_0.grad
    assert (shared_grad is not None and shared_grad.abs().sum() > 0) == bool(shared)
    assert (private_grad is not None and private_grad.abs().sum() > 0) == bool(private)


def test_at_least_one_pool_is_required():
    with pytest.raises(ValueError, match="at least one"):
        make_prompt(0, 0)
