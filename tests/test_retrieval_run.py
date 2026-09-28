"""B3 retrieval baseline end to end on tiny synthetic data: JPEGs, per-task COCO files, a fake
class-agnostic detector and a tiny randomly initialised DINOv2 (no weights are downloaded)."""

from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from transformers import Dinov2Config, Dinov2Model

from autocheckout.io import md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions, save_predictions
from baselines import retrieval
from tests.coco_helpers import make_ann, make_gt, make_image, three_task_config
from tools import eval_cl, eval_count

SIDE = 64
SLOTS = ([4, 4, 28, 28], [36, 36, 60, 60])  # xyxy; every image has one object in each slot
BACKGROUND_BOX = [40, 4, 60, 20]  # a low-score detector false positive on the empty table
COLORS = [(220, 30, 30), (30, 200, 30), (30, 30, 220), (230, 230, 40), (200, 40, 200), (40, 210, 210)]
# labels of the objects in (slot 0, slot 1) of each image; task t = labels {2t-2, 2t-1}
TRAIN = {1: [(0, 1), (1, 2), (0, 4)], 2: [(2, 3), (3, 0), (2, 5)], 3: [(4, 5), (5, 1), (4, 3)]}
EVAL = {"val": [(0, 2), (4, 1), (3, 5)], "test": [(1, 3), (5, 0), (2, 4), (0, 5)]}
LEVELS = ("easy", "medium", "hard")


class CountingModel(torch.nn.Module):
    """Tiny random DINOv2 that counts the crops it embeds."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        config = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2,
                              image_size=56, patch_size=14)
        self.inner = Dinov2Model(config).eval()
        self.crops = 0

    def forward(self, pixel_values: torch.Tensor):
        self.crops += len(pixel_values)
        return self.inner(pixel_values=pixel_values)


def _write_images_and_coco(path, image_dir, layouts, first_id, keep_labels):
    """Draw the images (ids ``first_id, ...``); write a COCO file annotating ``keep_labels`` only."""
    images, anns = [], []
    for i, layout in enumerate(layouts):
        image = make_image(first_id + i, LEVELS[i % len(LEVELS)])
        images.append(image)
        canvas = Image.new("RGB", (SIDE, SIDE), (128, 128, 128))
        for (x1, y1, x2, y2), label in zip(SLOTS, layout, strict=True):
            ImageDraw.Draw(canvas).rectangle([x1, y1, x2 - 1, y2 - 1], fill=COLORS[label])
            if label in keep_labels:
                anns.append(make_ann(len(anns) + 1, image["id"], label, [x1, y1, x2 - x1, y2 - y1]))
        canvas.save(image_dir / image["file_name"], quality=95)
    save_json(path, make_gt(images, anns, num_labels=6))


def _write_detector(path, ann_path, split, image_ids):
    """Class-agnostic detector: both objects of every image plus one low-score background box;
    rows in random order (the retrieval code must keep each row's embedding with its row)."""
    rows = [(image_id, q, score, box) for image_id in image_ids
            for q, (score, box) in enumerate(zip((0.9, 0.85, 0.02), (*SLOTS, BACKGROUND_BOX), strict=True))]
    rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    image_id, query, score, boxes = zip(*rows, strict=True)
    meta = {"task_id": 1, "seen_classes": 1, "split": split, "ann_file": str(ann_path),
            "ann_md5": md5_file(ann_path), "producer": "detector", "topk_per_image": 100}
    save_predictions(path, Predictions(image_id, query, [0] * len(rows), score, boxes, meta=meta))


@pytest.fixture
def data(tmp_path):
    cfg = three_task_config()
    cfg.save(tmp_path / "tasks.json")
    image_dir, task_dir, det_dir = tmp_path / "images", tmp_path / "tasks", tmp_path / "det"
    image_dir.mkdir()
    for t, layouts in TRAIN.items():  # only task t's objects are labelled in train_task_<t>
        train_path = task_dir / f"train_task_{t}.json"
        _write_images_and_coco(train_path, image_dir, layouts, 100 * t, cfg.task(t).labels)
    for split, first_id in (("val", 1), ("test", 11)):
        ann_path = task_dir / f"{split}_full.json"
        _write_images_and_coco(ann_path, image_dir, EVAL[split], first_id, range(6))
        _write_detector(det_dir / f"pred_{split}.npz", ann_path, split,
                        range(first_id, first_id + len(EVAL[split])))
    return {"cfg": cfg, "cfg_path": tmp_path / "tasks.json", "image_dir": image_dir, "task_dir": task_dir,
            "det_dir": det_dir, "emb_dir": tmp_path / "emb", "run_dir": tmp_path / "run"}


def _argv(data, command, *extra):
    argv = [command, "--emb-dir", str(data["emb_dir"])]
    if command != "predict":
        argv += ["--task-dir", str(data["task_dir"]), "--image-dir", str(data["image_dir"]), "--workers", "0"]
    if command != "memory":
        argv += ["--det-dir", str(data["det_dir"])]
    if command != "embed-dets":
        argv += ["--task-config", str(data["cfg_path"])]
    if command in ("memory", "run"):
        argv += ["--per-class", "1"]
    if command in ("predict", "run"):
        argv += ["--out-dir", str(data["run_dir"]), "--temperature", "0.001"]
    return argv + list(extra)


def _run(monkeypatch, data, command, *extra):
    model = CountingModel()
    monkeypatch.setattr(retrieval, "load_backbone", lambda name, device: model)
    retrieval.main(_argv(data, command, *extra))
    return model


@pytest.mark.parametrize("mode", ["prototype", "knn"])
def test_run_output_passes_the_evaluation_tools(monkeypatch, data, mode):
    _run(monkeypatch, data, "run", "--mode", mode, "--k", "3")
    cfg = data["cfg"]
    for task_id in (1, 2, 3):
        seen = cfg.seen_classes(task_id)
        for split in EVAL:
            preds = load_predictions(data["run_dir"] / f"task_{task_id}" / f"pred_{split}.npz")
            meta = preds.meta
            assert (meta["task_id"], meta["seen_classes"], meta["split"]) == (task_id, seen, split)
            assert (meta["producer"], meta["mode"]) == ("retrieval", mode)
            assert meta["ann_md5"] == md5_file(data["task_dir"] / f"{split}_full.json")
            assert preds.label.min() >= 0 and preds.label.max() < seen
            assert np.all((preds.score >= 0) & (preds.score <= 1))
            rows_per_box = Counter(zip(preds.image_id.tolist(), preds.query.tolist(), strict=True))
            assert max(rows_per_box.values()) <= min(5, seen)
            assert len(preds.top1_per_query()) == 3 * len(EVAL[split])  # one label per detected box

    val_ann, test_ann = data["task_dir"] / "val_full.json", data["task_dir"] / "test_full.json"
    cl = eval_cl.evaluate_run(data["run_dir"], "test", test_ann, cfg)
    count = eval_count.evaluate_run(data["run_dir"], val_ann, test_ann, cfg)
    assert set(cl["stages"]) == set(count["stages"]) == {1, 2, 3}
    # after the last task every object is a learned SKU, and all crops of one SKU look alike
    assert cl["stages"][3]["m1"]["mAP_A"]["AP50"] == pytest.approx(1.0)
    assert count["stages"][3]["test"]["overall"]["cAcc"] == pytest.approx(1.0)


def test_memory_holds_per_class_crops_of_the_task_file_only(monkeypatch, data):
    _run(monkeypatch, data, "run")
    for task_id in (1, 2, 3):
        meta, arrays = retrieval.load_arrays(data["emb_dir"] / f"memory_task_{task_id}.npz")
        assert sorted(arrays["labels"].tolist()) == list(data["cfg"].task(task_id).labels)  # 1 per class
        assert set(arrays["image_id"].tolist()) <= {100 * task_id + i for i in range(3)}
        assert (meta["train_file"], meta["per_class"]) == (f"train_task_{task_id}.json", 1)


def test_embeddings_are_computed_once_and_reused(monkeypatch, data):
    n_detections = 3 * sum(len(layouts) for layouts in EVAL.values())
    # detection crops once for the 3 tasks, plus 1 memory crop per class
    assert _run(monkeypatch, data, "run").crops == n_detections + 6
    pred_paths = sorted(data["run_dir"].rglob("pred_*.npz"))
    first = [load_predictions(p) for p in pred_paths]

    assert _run(monkeypatch, data, "run").crops == 0  # everything comes from the cache

    def no_model(name, device):
        raise AssertionError("predict must not load the model")

    monkeypatch.setattr(retrieval, "load_backbone", no_model)
    for p in pred_paths:
        p.unlink()
    for task_id in (1, 2, 3):
        retrieval.main(_argv(data, "predict", "--task", str(task_id)))
    for p, before in zip(pred_paths, first, strict=True):
        after = load_predictions(p)
        assert np.array_equal(after.label, before.label) and np.array_equal(after.score, before.score)


def test_cache_built_with_other_settings_is_refused(monkeypatch, data):
    _run(monkeypatch, data, "embed-dets")
    with pytest.raises(ValueError, match="delete it or use another --emb-dir"):
        _run(monkeypatch, data, "embed-dets", "--margin", "0.2")
    _run(monkeypatch, data, "memory", "--task", "1")
    with pytest.raises(ValueError, match="delete it or use another --emb-dir"):
        _run(monkeypatch, data, "memory", "--task", "1", "--seed", "1")
