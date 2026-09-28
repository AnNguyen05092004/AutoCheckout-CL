"""Tiny synthetic COCO-style fixtures shared by the V2/V3 tests. Not a test module itself."""

from __future__ import annotations

import numpy as np

from autocheckout.predictions import Predictions
from autocheckout.taskcfg import TaskConfig


def make_image(image_id: int, level: str = "easy") -> dict:
    return {"id": image_id, "file_name": f"img{image_id}.jpg", "width": 800, "height": 800, "level": level}


def make_ann(ann_id: int, image_id: int, category_id: int, box_xywh: list[float]) -> dict:
    x, y, w, h = box_xywh
    return {"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": [x, y, w, h],
             "area": w * h, "iscrowd": 0}


def make_category(category_id: int) -> dict:
    return {"id": category_id, "name": f"sku{category_id}", "supercategory": "toy"}


def make_gt(images: list[dict], annotations: list[dict], num_labels: int) -> dict:
    categories = [make_category(c) for c in range(num_labels)]
    return {"images": images, "annotations": annotations, "categories": categories}


def make_preds(rows: list[tuple[int, int, int, float, list[float]]], meta: dict | None = None) -> Predictions:
    """``rows``: list of (image_id, query, label, score, box_xyxy)."""
    if not rows:
        return Predictions([], [], [], [], np.zeros((0, 4)), meta=meta or {})
    image_id, query, label, score, boxes = zip(*rows, strict=True)
    return Predictions(list(image_id), list(query), list(label), list(score), list(boxes), meta=meta or {})


def three_task_config() -> TaskConfig:
    """Task 1: labels 0-1, task 2: labels 2-3, task 3: labels 4-5. No reserved slots."""
    return TaskConfig.from_dict({
        "name": "toy-3task",
        "seed": 0,
        "tasks": [
            {"task_id": 1, "offset": 0, "reserved": False, "classes": [
                {"label": 0, "name": "a0", "rpc_category_id": 1},
                {"label": 1, "name": "a1", "rpc_category_id": 2},
            ]},
            {"task_id": 2, "offset": 2, "reserved": False, "classes": [
                {"label": 2, "name": "b0", "rpc_category_id": 3},
                {"label": 3, "name": "b1", "rpc_category_id": 4},
            ]},
            {"task_id": 3, "offset": 4, "reserved": False, "classes": [
                {"label": 4, "name": "c0", "rpc_category_id": 5},
                {"label": 5, "name": "c1", "rpc_category_id": 6},
            ]},
        ],
    })
