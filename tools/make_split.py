"""DL3: split the merged checkout annotation into train / val / test and train_pilot, by image group.

IMPLEMENTATION_PLAN.md, section 4.3:

- The unit is a group of shots of one basket (``autocheckout.groups``); a group's level is the
  most common level of its images (ties go to the easier level).
- val and test are drawn only from groups whose images all come from test2019; every group
  with a val2019 image stays in train; train is everything else.
- Stratified by level: for each level, its groups are shuffled with the seed and taken until the
  level has ``--test-per-level`` images in test, then ``--val-per-level`` images in val.
- Accepted if every SKU has at least ``--min-test-objects`` objects in test and
  ``--min-val-objects`` in val; otherwise the next seed is tried. Every seed tried is recorded.
- train_pilot: ``--pilot`` images of train drawn by group, the same number per level.

Writes ``<out-dir>/{train,val,test,train_pilot}.json`` (subsets of the merged file; each image
gets its ``group``) and ``<config-dir>/rpc_checkout_seed<k>.json`` (seeds tried, counts, the
val/test/train_pilot images as ``<source>/<orig_file_name>``, md5 of each split file).
Re-running with the same input writes byte-identical files.

    python -m tools.make_split --ann /data/rpc/ann/checkout_800.json --out-dir /data/rpc/splits \\
        --config-dir configs/splits
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from autocheckout.groups import coco_group_keys
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.rpc import LEVELS
from autocheckout.sampling import largest_remainder, take_stratified

SPLITS = ("train", "val", "test", "train_pilot")


def build_groups(coco: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{group key: {"images": [ids], "level": majority level, "levels": {...}, "sources": {...}}}``."""
    key_of = coco_group_keys(coco)
    groups: dict[str, dict[str, Any]] = {}
    for img in coco["images"]:
        group = groups.setdefault(key_of[img["id"]], {"images": [], "levels": Counter(), "sources": set()})
        group["images"].append(img["id"])
        group["levels"][img["level"]] += 1
        group["sources"].add(img["source"])
    for group in groups.values():
        levels = group["levels"]
        group["level"] = max(levels, key=lambda level: (levels[level], -LEVELS.index(level)))
    return groups


def units(groups: dict[str, dict[str, Any]], keys: list[str]) -> list[tuple[str, str, int]]:
    """``(key, level, images)`` of each group, the sampling units of ``take_stratified``."""
    return [(key, groups[key]["level"], len(groups[key]["images"])) for key in keys]


def draw_splits(groups: dict[str, dict[str, Any]], seed: int, test_per_level: int,
                val_per_level: int) -> dict[str, list[str]]:
    """Group keys of train / val / test for one seed."""
    rng = random.Random(seed)
    eligible = sorted(key for key, g in groups.items() if g["sources"] == {"test2019"})
    test = take_stratified(units(groups, eligible), {level: test_per_level for level in LEVELS}, rng)
    in_test = set(test)
    rest = [key for key in eligible if key not in in_test]
    val = take_stratified(units(groups, rest), {level: val_per_level for level in LEVELS}, rng)
    train = sorted(set(groups) - in_test - set(val))
    return {"train": train, "val": val, "test": test}


def draw_pilot(groups: dict[str, dict[str, Any]], train: list[str], size: int, seed: int) -> list[str]:
    targets = largest_remainder({level: 1 for level in LEVELS}, size)
    return take_stratified(units(groups, train), targets, random.Random(f"pilot-{seed}"))


def image_ids(groups: dict[str, dict[str, Any]], keys: list[str]) -> set[int]:
    return {image_id for key in keys for image_id in groups[key]["images"]}


def stratum_sizes(groups: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, int]:
    """Images per level, counting every image of a group at the group's level (as the sampling does)."""
    counts = Counter()
    for key in keys:
        counts[groups[key]["level"]] += len(groups[key]["images"])
    return {level: counts[level] for level in LEVELS}


def images_per_level(coco: dict[str, Any], ids: set[int]) -> dict[str, int]:
    counts = Counter(img["level"] for img in coco["images"] if img["id"] in ids)
    return {level: counts[level] for level in LEVELS}


def objects_per_sku(coco: dict[str, Any], ids: set[int]) -> dict[int, int]:
    counts = Counter(ann["category_id"] for ann in coco["annotations"] if ann["image_id"] in ids)
    return {c["id"]: counts[c["id"]] for c in coco["categories"]}


def choose_split(coco: dict[str, Any], groups: dict[str, dict[str, Any]], *, seed: int, max_tries: int,
                 test_per_level: int, val_per_level: int, min_test_objects: int,
                 min_val_objects: int) -> tuple[int, dict[str, list[str]], list[dict[str, Any]]]:
    """Try seeds ``seed, seed+1, ...`` until the per-SKU minimums hold; returns (seed, splits, tried)."""
    minimums = {"test": min_test_objects, "val": min_val_objects}
    targets = {"test": test_per_level, "val": val_per_level}
    tried = []
    for current in range(seed, seed + max_tries):
        splits = draw_splits(groups, current, test_per_level, val_per_level)
        shortfalls = {}
        for name in ("test", "val"):
            sizes = stratum_sizes(groups, splits[name])
            short = {level: n for level, n in sizes.items() if n < targets[name]}
            if short:
                raise SystemExit(f"seed {current}: not enough test2019-only groups to give {name} "
                                 f"{targets[name]} images per level (got {short})")
            counts = objects_per_sku(coco, image_ids(groups, splits[name]))
            shortfalls[name] = {str(cid): n for cid, n in counts.items() if n < minimums[name]}
        accepted = not any(shortfalls.values())
        tried.append({"seed": current, "accepted": accepted, "skus_below_minimum": shortfalls})
        if accepted:
            return current, splits, tried
    raise SystemExit(f"no seed in {seed}..{seed + max_tries - 1} meets the per-SKU minimums: {tried}")


def subset(coco: dict[str, Any], ids: set[int], group_of: dict[int, str]) -> dict[str, Any]:
    return {
        "images": [img | {"group": group_of[img["id"]]} for img in coco["images"] if img["id"] in ids],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in ids],
        "categories": coco["categories"],
    }


def check_disjoint(split_groups: dict[str, list[str]]) -> None:
    train, val, test = (set(split_groups[name]) for name in ("train", "val", "test"))
    if train & val or train & test or val & test:
        raise AssertionError("a group is in two of train/val/test")
    if not set(split_groups["train_pilot"]) <= train:
        raise AssertionError("train_pilot is not a subset of train")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann", type=Path, required=True, help="merged annotation from DL2")
    parser.add_argument("--out-dir", type=Path, required=True, help="folder for the split files")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/splits"))
    parser.add_argument("--seed", type=int, default=0, help="first seed to try")
    parser.add_argument("--max-tries", type=int, default=20)
    parser.add_argument("--test-per-level", type=int, default=2000)
    parser.add_argument("--val-per-level", type=int, default=500)
    parser.add_argument("--pilot", type=int, default=3000)
    parser.add_argument("--min-test-objects", type=int, default=50)
    parser.add_argument("--min-val-objects", type=int, default=15)
    args = parser.parse_args(argv)

    coco = load_json(args.ann)
    groups = build_groups(coco)
    seed, split_groups, tried = choose_split(
        coco, groups, seed=args.seed, max_tries=args.max_tries, test_per_level=args.test_per_level,
        val_per_level=args.val_per_level, min_test_objects=args.min_test_objects,
        min_val_objects=args.min_val_objects)
    split_groups["train_pilot"] = draw_pilot(groups, split_groups["train"], args.pilot, seed)
    check_disjoint(split_groups)

    group_of = {image_id: key for key, g in groups.items() for image_id in g["images"]}
    name_of = {img["id"]: f"{img['source']}/{img['orig_file_name']}" for img in coco["images"]}
    counts, md5s, file_lists = {}, {}, {}
    for name in SPLITS:
        ids = image_ids(groups, split_groups[name])
        data = subset(coco, ids, group_of)
        path = args.out_dir / f"{name}.json"
        save_json(path, data)
        md5s[f"{name}.json"] = md5_file(path)
        counts[name] = {"groups": len(split_groups[name]), "images": len(ids),
                        "objects": len(data["annotations"]), "images_per_level": images_per_level(coco, ids),
                        "min_objects_per_sku": min(objects_per_sku(coco, ids).values())}
        if name != "train":
            file_lists[name] = sorted(name_of[i] for i in ids)
        print(f"{name:12s} groups {counts[name]['groups']:6d}  images {len(ids):6d}  "
              f"objects {counts[name]['objects']:7d}  per level {counts[name]['images_per_level']}")

    spanning = sorted(key for key, g in groups.items() if len(g["levels"]) > 1)
    config = {
        "name": f"rpc_checkout_seed{seed}",
        "seed": seed,
        "seeds_tried": tried,
        "params": {k: getattr(args, k) for k in ("test_per_level", "val_per_level", "pilot",
                                                 "min_test_objects", "min_val_objects")},
        "ann_md5": md5_file(args.ann),
        "groups": {
            "total": len(groups),
            "test2019_only": sum(g["sources"] == {"test2019"} for g in groups.values()),
            "largest": max(len(g["images"]) for g in groups.values()),
            "spanning_levels": spanning,
        },
        "counts": counts,
        "image_key": "<source>/<orig_file_name>; train = every other image",
        "images": file_lists,
        "md5": md5s,
    }
    config_path = args.config_dir / f"rpc_checkout_seed{seed}.json"
    save_json(config_path, config, indent=1)
    print(f"seeds tried: {[t['seed'] for t in tried]}; accepted seed {seed}; "
          f"{len(spanning)} groups span levels; wrote {config_path}")


if __name__ == "__main__":
    main()
