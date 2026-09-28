"""Synthetic RPC-like raw checkout data for the data-tool tests.

Tiny square JPEGs (``SIDE`` px stand in for 1800 px) with RPC-style names
``YYYYMMDD-HH-MM-SS-<suffix>.jpg``, both sources, three levels and runs of shots of the same
basket. Besides regular runs (one suffix = one basket), it contains on purpose:

- ``shared``: suffix 7001 used in val2019 and test2019; one file name exists in both sources;
- ``same_multiset``: suffixes 7002 and 7003 (test2019, different days) with the same SKU multiset;
- ``spanning``: suffix 7004 reused on another day for a different basket of another level
  (3 hard + 2 medium shots);
- ``duplicate``: two different test2019 file names with byte-identical images.
"""

from __future__ import annotations

import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image

from autocheckout.groups import sku_multiset
from autocheckout.io import save_json
from autocheckout.rpc import LEVELS, SOURCES

SIDE = 90
SUPERCATEGORIES = {"drink": 5, "snack": 3, "candy": 2, "tissue": 2}
CATEGORIES = []
for _group, _count in SUPERCATEGORIES.items():
    for _ in range(_count):
        _id = len(CATEGORIES) + 1
        CATEGORIES.append({"supercategory": _group, "id": _id, "name": f"{_id}_{_group}"})
OBJECTS_PER_LEVEL = {"easy": (2, 3), "medium": (4, 5), "hard": (6, 7)}
REGULAR_RUNS = {"val2019": 3, "test2019": 8}  # runs of 3 shots per level


def make_raw_rpc(root: Path) -> dict[str, Any]:
    """Write ``root/{val2019,test2019}/*.jpg`` and ``root/instances_*.json``; return what was planted."""
    rng = random.Random(0)
    used: set[tuple] = set()

    def new_basket(level: str) -> list[int]:
        while True:
            low, high = OBJECTS_PER_LEVEL[level]
            cats = [rng.randint(1, len(CATEGORIES)) for _ in range(rng.randint(low, high))]
            if sku_multiset(cats) not in used:
                used.add(sku_multiset(cats))
                return cats

    # (source, start time, suffix, level, category ids of the basket, number of shots)
    runs: list[tuple[str, datetime, str, str, list[int], int]] = []
    start = datetime(2018, 8, 27, 9, 0, 0)
    for source in SOURCES:
        for level in LEVELS:
            for _ in range(REGULAR_RUNS[source]):
                runs.append((source, start + timedelta(minutes=len(runs)), str(100 + len(runs)), level,
                             new_basket(level), 3))
    shared_time = datetime(2018, 8, 28, 10, 0, 0)
    runs.append(("val2019", shared_time, "7001", "easy", new_basket("easy"), 3))
    runs.append(("test2019", shared_time + timedelta(seconds=22), "7001", "easy", new_basket("easy"), 3))
    twin = new_basket("medium")
    runs.append(("test2019", datetime(2018, 8, 28, 11, 0, 0), "7002", "medium", twin, 3))
    runs.append(("test2019", datetime(2018, 8, 29, 11, 0, 0), "7003", "medium", twin, 3))
    runs.append(("test2019", datetime(2018, 8, 28, 12, 0, 0), "7004", "hard", new_basket("hard"), 3))
    runs.append(("test2019", datetime(2018, 8, 30, 12, 0, 0), "7004", "medium", new_basket("medium"), 2))

    cocos = {source: {"images": [], "annotations": [], "categories": CATEGORIES} for source in SOURCES}
    for source in SOURCES:
        (root / source).mkdir(parents=True, exist_ok=True)
    color = 0
    for source, run_start, suffix, level, cats, shots in runs:
        coco = cocos[source]
        for shot in range(shots):
            name = f"{(run_start + timedelta(seconds=11 * shot)):%Y%m%d-%H-%M-%S}-{suffix}.jpg"
            image_id = len(coco["images"]) + 1
            coco["images"].append({"file_name": name, "width": SIDE, "height": SIDE, "id": image_id,
                                   "level": level})
            color += 1
            Image.new("RGB", (SIDE, SIDE), ((color * 37) % 256, (color * 91) % 256, 128)).save(
                root / source / name, quality=95)
            for cat in cats:
                w, h = rng.randint(9, 30), rng.randint(9, 30)
                x, y = rng.randint(0, SIDE - w), rng.randint(0, SIDE - h)
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                            "category_id": cat, "bbox": [x, y, w, h], "area": float(w * h),
                                            "iscrowd": 0})
    test_names = [img["file_name"] for img in cocos["test2019"]["images"]]
    shutil.copyfile(root / "test2019" / test_names[0], root / "test2019" / test_names[1])
    for source in SOURCES:
        save_json(root / f"instances_{source}.json", cocos[source])
    return {
        "cocos": cocos,
        "shared_file_name": f"{shared_time + timedelta(seconds=22):%Y%m%d-%H-%M-%S}-7001.jpg",
        "duplicate": sorted(test_names[:2]),
    }


def make_resized_rpc(root: Path, size: int = 40) -> tuple[Path, Path]:
    """Synthetic raw data run through DL2; returns (resized image folder, merged annotation)."""
    from tools.resize import main as resize

    make_raw_rpc(root / "raw")
    images, ann = root / "checkout", root / "ann" / "checkout.json"
    resize(["--raw", str(root / "raw"), "--out-dir", str(images), "--ann-out", str(ann), "--size", str(size),
            "--workers", "1"])
    return images, ann
