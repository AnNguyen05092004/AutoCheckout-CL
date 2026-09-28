"""Helpers for tests of the PDP code: toy task configs and a small, fast Deformable DETR on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RPC_SIZES = (100, 25, 25, 25, 25)
RPC_RESERVED = 24


def task_config_dict(sizes=RPC_SIZES, reserved=RPC_RESERVED, name="toy") -> dict:
    """A task config in the docs/formats.md layout with fake SKU names."""
    tasks, offset, rpc_id = [], 0, 1
    for task_id, size in enumerate(sizes, start=1):
        classes = [{"label": offset + i, "name": f"sku_{rpc_id + i}", "rpc_category_id": rpc_id + i,
                    "supercategory": "drink"} for i in range(size)]
        tasks.append({"task_id": task_id, "offset": offset, "reserved": False, "classes": classes})
        offset += size
        rpc_id += size
    if reserved:
        tasks.append({"task_id": len(sizes) + 1, "offset": offset, "reserved": True, "classes": [
            {"label": offset + i, "name": f"reserved_{offset + i}", "rpc_category_id": None}
            for i in range(reserved)]})
    return {"name": name, "seed": 0, "tasks": tasks}


def write_task_config(path: Path, **kwargs) -> Path:
    path.write_text(json.dumps(task_config_dict(**kwargs)))
    return path


def main_args(argv: list[str]) -> argparse.Namespace:
    """Parse PDP main.py arguments exactly as the command line would."""
    import main as pdp_main

    parser = argparse.ArgumentParser(parents=[pdp_main.get_args_parser()])
    return parser.parse_args(argv)


# --- a small Deformable DETR for fast CPU tests ---------------------------------------------------
# d_model and num_queries keep the PDP values (query_tf is hard-coded to 300 x 256); the rest shrinks.
TINY_DETR = dict(use_pretrained_backbone=False, backbone="resnet18", encoder_layers=1, decoder_layers=2,
                 encoder_ffn_dim=64, decoder_ffn_dim=64)
IMAGE_SIZE = 64


def use_tiny_detr(monkeypatch) -> None:
    """Make engine.local_trainer build the tiny model with random weights (no download)."""
    import engine
    from models.configuration_deformable_detr import DeformableDetrConfig

    monkeypatch.setattr(engine, "DeformableDetrConfig", lambda: DeformableDetrConfig(**TINY_DETR))


def pdp_args(tmp_path: Path, *extra: str, sizes=(3, 2), reserved=0) -> argparse.Namespace:
    """Arguments for a small PDP run on a toy task config, set up like main() does."""
    import main as pdp_main

    config = write_task_config(tmp_path / "tasks.json", sizes=sizes, reserved=reserved)
    n_classes = sum(sizes) + reserved + 1
    args = main_args([
        "--task_config", str(config), "--n_classes", str(n_classes), "--repo_name", "",
        "--n_tasks", str(len(sizes)), "--use_prompts", "1", "--local_query", "1",
        "--num_prompts", "4", "--prompt_len", "2", "--lambda_query", "0.1",
        "--freeze", "backbone,encoder,decoder", "--new_params", "class_embed,prompts",
        "--output_dir", str(tmp_path / "run"), *extra,
    ])
    pdp_main.setup_task_info(args)
    args.log_file = open(tmp_path / "log.txt", "a")  # noqa: SIM115 - closed with the tmp dir
    return args


def make_trainer(args: argparse.Namespace, task_id: int):
    """engine.local_trainer for task_id, prompts set to that task, freezing applied like main()."""
    from types import SimpleNamespace

    import engine

    trainer = engine.local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                                   local_evaluator=SimpleNamespace(), task_id=task_id)
    trainer.model.model.prompts.set_task_id(task_id - 1)
    trainer.resume()  # only applies --freeze when no path is given
    return trainer


def make_batch(labels_per_image: list[list[int]], seed: int = 0) -> dict:
    """A batch in the format of CocoDetection.collate_fn, with random pixels and boxes."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    batch_size = len(labels_per_image)
    labels = []
    for image_id, classes in enumerate(labels_per_image, start=1):
        n = len(classes)
        centers = 0.2 + 0.6 * torch.rand(n, 2, generator=generator)
        sizes = 0.1 + 0.2 * torch.rand(n, 2, generator=generator)
        labels.append({
            "class_labels": torch.tensor(classes, dtype=torch.int64),
            "boxes": torch.cat([centers, sizes], dim=1),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
            "size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
        })
    return {
        "pixel_values": torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator),
        "pixel_mask": torch.ones(batch_size, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.int64),
        "labels": labels,
    }
