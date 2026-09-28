"""F3: the directional decoupled loss L_DDL (paper eq. 9-10) is computed and has gradients."""

import math
from types import SimpleNamespace

import torch

from models.prompt import Prompt


def make_prompt():
    torch.manual_seed(0)
    return Prompt(emb_d=16, n_tasks=3, prompt_param=[4, 2, 0], key_dim=16,
                  args=SimpleNamespace(local_query=0), task_num_classes=[3, 2, 2])


def reference_ddl(shared, private, threshold=math.pi / 2):
    """Direct transcription of eq. 9-10 without lambda: 2/(Ns*Np) * sum max(0, theta_ddl - theta_ij)."""
    total = 0.0
    for ps in shared.reshape(len(shared), -1):
        for pp in private.reshape(len(private), -1):
            cos = torch.dot(ps, pp) / (ps.norm() * pp.norm())
            total += max(0.0, threshold - math.acos(float(cos)))
    return 2 * total / (len(shared) * len(private))


def test_matches_paper_formula_averaged_over_layers():
    prompt = make_prompt()
    prompt.set_task_id(1)
    prompt.init_task_prompts()
    with torch.no_grad():  # make some pairs have angles below 90 degrees
        for e in prompt.e_layers:
            getattr(prompt, f"private_p_{e}")[3] += getattr(prompt, f"shared_p_{e}")[0]
    expected = sum(
        reference_ddl(getattr(prompt, f"shared_p_{e}").detach(),
                      getattr(prompt, f"private_p_{e}")[:5].detach())
        for e in prompt.e_layers) / len(prompt.e_layers)
    assert expected > 0
    assert abs(prompt.ddl_loss_all_layers().item() - expected) < 1e-5


def test_gradients_reach_shared_pool_and_current_private_prompts_only():
    prompt = make_prompt()
    prompt.set_task_id(1)
    prompt.init_task_prompts()
    with torch.no_grad():
        for e in prompt.e_layers:
            getattr(prompt, f"private_p_{e}")[:5] += getattr(prompt, f"shared_p_{e}")[0]
    prompt.ddl_loss_all_layers().backward()
    shared_grad = prompt.shared_p_0.grad
    private_grad = prompt.private_p_0.grad
    assert shared_grad is not None and shared_grad.abs().sum() > 0
    assert private_grad[:3].abs().sum() == 0  # task 1 prompts are frozen (detached)
    assert private_grad[3:5].abs().sum() > 0  # current task (task 2) prompts
    assert private_grad[5:].abs().sum() == 0  # future task
