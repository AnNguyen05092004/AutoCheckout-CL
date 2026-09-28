"""R4/V5: the benchmark runs a full PDP training step (teacher + PPG + student) and two-pass inference."""

import pytest
import torch

import benchmark
import main as pdp_main
from pdp_helpers import IMAGE_SIZE, use_tiny_detr, write_task_config

pytestmark = pytest.mark.slow


def bench_args(tmp_path, *extra):
    config = write_task_config(tmp_path / "tasks.json", sizes=(100, 25), reserved=0)
    args = benchmark.parse_args([
        "--task_config", str(config), "--n_classes", "126", "--repo_name", "", "--n_tasks", "2",
        "--use_prompts", "1", "--local_query", "1", "--num_prompts", "4", "--prompt_len", "2",
        "--freeze", "backbone,encoder,decoder", "--new_params", "class_embed,prompts", "--lr", "1e-4",
        "--batch_size", "2", "--bench_steps", "3", "--bench_train_size", str(IMAGE_SIZE),
        "--bench_sizes", str(IMAGE_SIZE), "--accelerator", "cpu", *extra])
    pdp_main.setup_task_info(args)
    args.log_file = open(tmp_path / "log.txt", "w")  # noqa: SIM115
    return args


def test_train_and_infer_modes(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    args = bench_args(tmp_path)
    train = benchmark.bench_train(args, torch.device("cpu"))
    assert train["seconds_per_image"] > 0 and train["batch_size"] == 2
    infer = benchmark.bench_infer(args, torch.device("cpu"))
    assert infer[f"ms_per_image_{IMAGE_SIZE}"] > 0
    storage = benchmark.storage_per_class(args)
    assert storage["prototype_memory_bytes_per_class"] == 100 * 1024  # 100 KiB, as in the plan (V5)
