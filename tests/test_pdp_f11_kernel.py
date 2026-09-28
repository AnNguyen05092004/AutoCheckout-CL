"""F11: the CUDA kernel of multi-scale deformable attention loads and matches the PyTorch version.

Runs only on a machine with CUDA (the VM); skipped on the Mac.
"""

import time
from pathlib import Path

import pytest
import torch

import models.modeling_deformable_detr as detr_module
from models.load_custom import load_cuda_kernels  # noqa: F401  (import must succeed everywhere)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_kernel_sources_ship_with_transformers():
    import transformers

    root = Path(transformers.__file__).resolve().parent / "kernels" / "deformable_detr"
    for name in ("vision.cpp", "cpu/ms_deform_attn_cpu.cpp", "cuda/ms_deform_attn_cuda.cu"):
        assert (root / name).is_file(), root / name


def random_inputs(device, batch=2, queries=300, heads=8, head_dim=32, levels=4, points=4):
    shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]], device=device)[:levels]
    start_index = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
    length = int((shapes[:, 0] * shapes[:, 1]).sum())
    value = torch.randn(batch, length, heads, head_dim, device=device)
    locations = torch.rand(batch, queries, heads, levels, points, 2, device=device)
    weights = torch.rand(batch, queries, heads, levels, points, device=device)
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True)
    return value, shapes, start_index, locations, weights


@cuda
def test_kernel_is_loaded():
    assert detr_module.MultiScaleDeformableAttention is not None, detr_module.KERNEL_LOAD_ERROR


@cuda
def test_kernel_matches_pytorch_forward_and_backward():
    torch.manual_seed(0)
    value, shapes, start, locations, weights = random_inputs("cuda")
    tensors = [t.clone().requires_grad_(True) for t in (value, locations, weights)]
    kernel_out = detr_module.MultiScaleDeformableAttentionFunction.apply(
        tensors[0], shapes, start, tensors[1], tensors[2], 64)
    reference = [t.clone().requires_grad_(True) for t in (value, locations, weights)]
    torch_out = detr_module.multi_scale_deformable_attention(reference[0], shapes, reference[1], reference[2])
    assert (kernel_out - torch_out).abs().max().item() <= 1e-4

    grad = torch.randn_like(kernel_out)
    kernel_out.backward(grad)
    torch_out.backward(grad)
    for ours, ref in zip(tensors, reference, strict=True):
        assert torch.allclose(ours.grad, ref.grad, atol=1e-3, rtol=1e-3)


@cuda
def test_kernel_speed_is_reported(capsys):
    value, shapes, start, locations, weights = random_inputs("cuda", batch=4)
    timings = {}
    for name, fn in {
        "kernel": lambda: detr_module.MultiScaleDeformableAttentionFunction.apply(
            value, shapes, start, locations, weights, 64),
        "pytorch": lambda: detr_module.multi_scale_deformable_attention(value, shapes, locations, weights),
    }.items():
        fn()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - begin) / 20
    with capsys.disabled():
        print(f"\nMSDA forward, batch 4: kernel {timings['kernel'] * 1e3:.2f} ms, "
              f"pytorch {timings['pytorch'] * 1e3:.2f} ms")
