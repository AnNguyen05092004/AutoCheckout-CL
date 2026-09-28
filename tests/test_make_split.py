from collections import Counter, defaultdict

import pytest

from autocheckout.io import load_json, md5_file
from autocheckout.rpc import LEVELS
from tools.make_split import SPLITS, build_groups, choose_split, image_ids, main, objects_per_sku

from synth_rpc import make_resized_rpc

PARAMS = ["--test-per-level", "6", "--val-per-level", "3", "--pilot", "9",
          "--min-test-objects", "1", "--min-val-objects", "0"]


def run_split(ann, out):
    main(["--ann", str(ann), "--out-dir", str(out / "splits"), "--config-dir", str(out / "configs"), *PARAMS])
    (config_path,) = (out / "configs").iterdir()
    return {name: load_json(out / "splits" / f"{name}.json") for name in SPLITS}, config_path


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    _, ann = make_resized_rpc(root)
    splits, config_path = run_split(ann, root)
    return load_json(ann), splits, config_path, root


def test_no_group_leaks_and_every_image_is_used(split):
    merged, splits, _, _ = split
    groups_of = {name: {img["group"] for img in data["images"]} for name, data in splits.items()}
    assert not groups_of["train"] & groups_of["val"]
    assert not groups_of["train"] & groups_of["test"]
    assert not groups_of["val"] & groups_of["test"]
    ids = [img["id"] for name in ("train", "val", "test") for img in splits[name]["images"]]
    assert sorted(ids) == [img["id"] for img in merged["images"]]
    for data in splits.values():
        kept = {img["id"] for img in data["images"]}
        assert {ann["image_id"] for ann in data["annotations"]} <= kept
        assert data["categories"] == merged["categories"]
    assert sum(len(splits[n]["annotations"]) for n in ("train", "val", "test")) == len(merged["annotations"])


def test_val_and_test_only_hold_test2019_groups(split):
    _, splits, _, _ = split
    sources = defaultdict(set)
    for name in ("train", "val", "test"):
        for img in splits[name]["images"]:
            sources[img["group"]].add(img["source"])
    for name in ("val", "test"):
        assert all(sources[img["group"]] == {"test2019"} for img in splits[name]["images"])
    train_groups = {img["group"] for img in splits["train"]["images"]}
    assert {g for g, s in sources.items() if "val2019" in s} <= train_groups
    assert "g7001" in train_groups  # suffix used in val2019 and test2019


def test_sizes_per_level_and_pilot(split):
    _, splits, config_path, _ = split
    config = load_json(config_path)
    for name, target in (("test", 6), ("val", 3)):
        per_level = Counter(img["level"] for img in splits[name]["images"])
        assert set(per_level) == set(LEVELS)
        assert target - 2 <= min(per_level.values()) and max(per_level.values()) < target + 6
        assert config["counts"][name]["images_per_level"] == {level: per_level[level] for level in LEVELS}
    train_ids = {img["id"] for img in splits["train"]["images"]}
    pilot_ids = {img["id"] for img in splits["train_pilot"]["images"]}
    assert pilot_ids <= train_ids and 9 <= len(pilot_ids) < 9 + 3 * 6
    pilot_groups = {img["group"] for img in splits["train_pilot"]["images"]}
    assert {img["id"] for img in splits["train"]["images"] if img["group"] in pilot_groups} == pilot_ids


def test_per_sku_minimums_and_config(split):
    _, splits, config_path, root = split
    config = load_json(config_path)
    assert config_path.name == f"rpc_checkout_seed{config['seed']}.json"
    assert config["seeds_tried"][-1] == {"seed": config["seed"], "accepted": True,
                                         "skus_below_minimum": {"test": {}, "val": {}}}
    for name, minimum in (("test", 1), ("val", 0)):
        counts = Counter(ann["category_id"] for ann in splits[name]["annotations"])
        lowest = min(counts[c["id"]] for c in splits[name]["categories"])
        assert lowest >= minimum and config["counts"][name]["min_objects_per_sku"] == lowest
    for name in SPLITS:
        assert config["md5"][f"{name}.json"] == md5_file(root / "splits" / f"{name}.json")
        if name != "train":
            assert config["images"][name] == sorted(f"{img['source']}/{img['orig_file_name']}"
                                                    for img in splits[name]["images"])


def test_rerun_is_byte_identical(split, tmp_path):
    _, _, first_config, first_root = split
    _, config_path = run_split(first_root / "ann" / "checkout.json", tmp_path)
    assert config_path.read_bytes() == first_config.read_bytes()
    for name in SPLITS:
        first, second = (root / "splits" / f"{name}.json" for root in (first_root, tmp_path))
        assert md5_file(first) == md5_file(second)


def test_failed_seeds_are_retried_and_recorded(split):
    merged = split[0]
    groups = build_groups(merged)
    kwargs = dict(max_tries=50, test_per_level=6, val_per_level=3, min_test_objects=1, min_val_objects=0)
    for start in range(30):
        seed, splits, tried = choose_split(merged, groups, seed=start, **kwargs)
        if len(tried) > 1:
            break
    else:
        pytest.fail("the fixture never needs a second seed")
    assert [t["seed"] for t in tried] == list(range(start, seed + 1))
    assert all(not t["accepted"] and any(t["skus_below_minimum"].values()) for t in tried[:-1])
    assert tried[-1]["accepted"]
    assert min(objects_per_sku(merged, image_ids(groups, splits["test"])).values()) >= 1


def test_impossible_requests_fail_loudly(split):
    merged = split[0]
    groups = build_groups(merged)
    base = dict(seed=0, max_tries=3, test_per_level=6, val_per_level=3, min_test_objects=0, min_val_objects=0)
    with pytest.raises(SystemExit, match="no seed in 0..2"):
        choose_split(merged, groups, **(base | {"min_test_objects": 10**6}))
    with pytest.raises(SystemExit, match="not enough test2019-only groups"):
        choose_split(merged, groups, **(base | {"test_per_level": 1000}))


def test_unstratified_split_for_groups_that_mix_levels(tmp_path):
    """Real RPC suffix groups hold three baskets of different levels: draw groups from one pool."""
    from tools.make_split import draw_pilot, draw_splits

    groups = {}
    for g in range(60):  # 60 groups of 9 images, 3 per level, all from test2019 except the last 5
        ids = list(range(9 * g, 9 * g + 9))
        groups[f"g{g}"] = {"images": ids, "levels": Counter(easy=3, medium=3, hard=3), "level": "easy",
                           "sources": {"test2019"} if g < 55 else {"val2019"}}
    splits = draw_splits(groups, seed=0, test_per_level=30, val_per_level=9, stratify="none")
    test_images = sum(len(groups[k]["images"]) for k in splits["test"])
    val_images = sum(len(groups[k]["images"]) for k in splits["val"])
    assert 90 <= test_images < 99 and 27 <= val_images < 36  # target 3 x per level, overshoot < one group
    assert not set(splits["test"]) & set(splits["val"])
    assert {f"g{g}" for g in range(55, 60)} <= set(splits["train"])
    pilot = draw_pilot(groups, splits["train"], 20, seed=0, stratify="none")
    assert set(pilot) <= set(splits["train"]) and 20 <= sum(len(groups[k]["images"]) for k in pilot) < 29
    assert draw_splits(groups, seed=0, test_per_level=30, val_per_level=9, stratify="none") == splits
