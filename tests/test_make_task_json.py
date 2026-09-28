import os
from collections import Counter

import pytest
from pycocotools.coco import COCO
from synth_rpc import CATEGORIES, make_resized_rpc

from autocheckout.io import load_json, md5_file, save_json
from autocheckout.sampling import largest_remainder
from tools.make_split import main as make_split
from tools.make_task_config import build_task_config
from tools.make_task_json import combine_sources, main

CAP = 20
TASK_FILES = ("train_task_{}.json", "train_task_{}_capped.json", "train_task_{}_gt_full.json",
              "val_task_{}.json")


def run(root, out, *extra, sources=None):
    splits = root / "splits"
    sources = sources or [f"real={splits / 'train.json'}"]
    main(["--task-config", str(root / "tasks.json"), *[a for s in sources for a in ("--train-source", s)],
          "--val", str(splits / "val.json"), "--test", str(splits / "test.json"), "--out-dir", str(out),
          "--cap", str(CAP), *extra])
    return out


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    images, ann = make_resized_rpc(root)
    make_split(["--ann", str(ann), "--out-dir", str(root / "splits"), "--config-dir", str(root / "configs"),
                "--test-per-level", "6", "--val-per-level", "3", "--pilot", "9",
                "--min-test-objects", "0", "--min-val-objects", "0"])
    config = build_task_config(CATEGORIES, [6, 3, 3], reserved=2, seed=0, name="6-2x3_seed0")
    config.save(root / "tasks.json")
    splits = {name: load_json(root / "splits" / f"{name}.json") for name in ("train", "val", "test")}
    return root, images, config, splits, run(root, root / "tasks")


def test_task_files_hold_only_task_labels(pipeline):
    _, _, config, splits, out = pipeline
    for task in config.data_tasks:
        rpc_ids = {c.rpc_category_id for c in task.classes}
        for prefix, split in (("train_task", "train"), ("val_task", "val")):
            data = load_json(out / f"{prefix}_{task.task_id}.json")
            anns = splits[split]["annotations"]
            expected = {a["image_id"] for a in anns if a["category_id"] in rpc_ids}
            assert {img["id"] for img in data["images"]} == expected
            assert {a["category_id"] for a in data["annotations"]} <= set(task.labels)
            assert len(data["annotations"]) == sum(a["category_id"] in rpc_ids for a in anns)


def test_labels_come_from_the_task_config(pipeline):
    _, _, config, splits, out = pipeline
    original = {a["id"]: a for split in splits.values() for a in split["annotations"]}
    names, label_to_rpc = config.label_names(), config.label_to_rpc()
    for path in out.glob("*.json"):
        if path.name == "manifest.json":
            continue
        data = load_json(path)
        for ann in data["annotations"]:
            old = original[ann["id"]]
            assert label_to_rpc[ann["category_id"]] == old["category_id"]
            assert (ann["bbox"], ann["area"], ann["iscrowd"]) == (old["bbox"], old["area"], 0)
        labels = sorted({a["category_id"] for a in data["annotations"]})
        assert [c["id"] for c in data["categories"]] == labels
        for category in data["categories"]:
            assert category["name"] == names[category["id"]]
            assert category["rpc_category_id"] == label_to_rpc[category["id"]]


def test_images_keep_all_fields(pipeline):
    _, _, _, splits, out = pipeline
    by_id = {img["id"]: img for split in splits.values() for img in split["images"]}
    for img in load_json(out / "train_task_1.json")["images"]:
        assert img == by_id[img["id"]] | {"train_source": "real"}
    for img in load_json(out / "val_full.json")["images"]:
        assert img == by_id[img["id"]]


def test_gt_full_and_full_files_keep_every_object(pipeline):
    _, _, config, splits, out = pipeline
    for task in config.data_tasks:
        gt_full = load_json(out / f"train_task_{task.task_id}_gt_full.json")
        ids = {img["id"] for img in load_json(out / f"train_task_{task.task_id}.json")["images"]}
        assert {img["id"] for img in gt_full["images"]} == ids
        train_anns = splits["train"]["annotations"]
        assert len(gt_full["annotations"]) == sum(a["image_id"] in ids for a in train_anns)
        assert {a["category_id"] for a in gt_full["annotations"]} - set(task.labels)  # other tasks too
    for name in ("val", "test"):
        full = load_json(out / f"{name}_full.json")
        assert len(full["images"]) == len(splits[name]["images"])
        assert len(full["annotations"]) == len(splits[name]["annotations"])


def test_capped_files_are_stratified_and_deterministic(pipeline, tmp_path):
    root, _, config, _, out = pipeline
    capped_tasks = 0
    for task in config.data_tasks:
        full = load_json(out / f"train_task_{task.task_id}.json")
        capped = load_json(out / f"train_task_{task.task_id}_capped.json")
        if len(full["images"]) <= CAP:
            assert capped == full
            continue
        capped_tasks += 1
        levels = Counter(img["level"] for img in full["images"])
        assert Counter(img["level"] for img in capped["images"]) == largest_remainder(levels, CAP)
        assert {img["id"] for img in capped["images"]} <= {img["id"] for img in full["images"]}
    assert capped_tasks > 0
    again = run(root, tmp_path / "again")
    assert load_json(again / "manifest.json")["files"] == load_json(out / "manifest.json")["files"]


def test_manifest(pipeline):
    root, _, config, _, out = pipeline
    manifest = load_json(out / "manifest.json")
    expected = {f.format(t.task_id) for t in config.data_tasks for f in TASK_FILES}
    expected |= {"val_full.json", "test_full.json"}
    assert set(manifest["files"]) == expected
    assert {p.name for p in out.glob("*.json")} == expected | {"manifest.json"}
    for name, info in manifest["files"].items():
        data = load_json(out / name)
        assert info == {"images": len(data["images"]), "objects": len(data["annotations"]),
                        "md5": md5_file(out / name)}
    assert manifest["task_config"]["md5"] == md5_file(root / "tasks.json")
    assert manifest["tasks"] == [1, 2, 3] and set(manifest["train_sources"]) == {"real"}


def test_pilot_uses_the_same_tool(pipeline, tmp_path):
    root = pipeline[0]
    pilot = root / "splits" / "train_pilot.json"
    out = run(root, tmp_path / "pilot", "--tasks", "1,2", sources=[f"real={pilot}"])
    names = {p.name for p in out.glob("*.json")}
    assert "train_task_2_capped.json" in names and "train_task_3.json" not in names
    pilot_ids = {img["id"] for img in load_json(pilot)["images"]}
    assert {img["id"] for img in load_json(out / "train_task_1_gt_full.json")["images"]} <= pilot_ids
    with pytest.raises(SystemExit):
        run(root, tmp_path / "bad", "--tasks", "4")  # the reserved task has no data


def test_several_train_sources(pipeline, tmp_path):
    root, _, _, splits, _ = pipeline
    train = splits["train"]
    first_two = {img["id"] for img in train["images"][:2]}
    composite = {
        "images": [img | {"id": img["id"] + 10**6} for img in train["images"][:2]],
        "annotations": [a | {"id": a["id"] + 10**7, "image_id": a["image_id"] + 10**6}
                        for a in train["annotations"] if a["image_id"] in first_two],
        "categories": train["categories"],
    }
    save_json(tmp_path / "composite.json", composite)
    out = run(root, tmp_path / "multi",
              sources=[f"real={root / 'splits' / 'train.json'}", f"composite={tmp_path / 'composite.json'}"])
    images = [img for t in (1, 2, 3) for img in load_json(out / f"train_task_{t}_gt_full.json")["images"]]
    composite_ids = {img["id"] for img in images if img["train_source"] == "composite"}
    assert composite_ids == {i + 10**6 for i in first_two}
    assert set(load_json(out / "manifest.json")["train_sources"]) == {"real", "composite"}
    with pytest.raises(ValueError, match="image ids are not unique"):
        combine_sources([("real", train), ("again", train)])


def test_files_load_with_pycocotools_and_the_pdp_loader(pipeline):
    _, images, _, _, out = pipeline
    for path in out.glob("*.json"):
        if path.name != "manifest.json":
            COCO(str(path))
    os.environ.setdefault("USE_TF", "0")  # the TensorFlow in the Mac venv aborts on import
    from datasets.coco_hug import CocoDetection
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor

    dataset = CocoDetection(img_folder=str(images), ann_file=str(out / "train_task_2.json"),
                            processor=DeformableDetrImageProcessor())
    pixel_values, target = dataset[0]
    anns = dataset.coco.imgToAnns[dataset.ids[0]]
    assert pixel_values.shape[0] == 3 and len(anns) > 0
    assert target["class_labels"].tolist() == [a["category_id"] for a in anns]
    scale = (target["size"][0] / target["orig_size"][0]).item() ** 2  # the processor resizes 40 -> 800 px
    assert target["area"].tolist() == pytest.approx([a["area"] * scale for a in anns], rel=1e-4)
