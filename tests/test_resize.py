import copy
from collections import defaultdict

import pytest
from PIL import Image
from synth_rpc import SIDE, make_raw_rpc

from autocheckout.io import load_json, save_json
from autocheckout.rpc import SOURCES
from tools.resize import main, merge_annotations, resize_one

SIZE = 40


@pytest.fixture(scope="module")
def resized(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    planted = make_raw_rpc(raw)
    data = tmp_path_factory.mktemp("data")
    main(["--raw", str(raw), "--out-dir", str(data / "checkout"), "--ann-out", str(data / "ann.json"),
          "--size", str(SIZE), "--workers", "2", "--draw", "5", "--draw-dir", str(data / "draw")])
    return raw, data, planted, load_json(data / "ann.json")


def test_images_are_renumbered_in_documented_order(resized):
    _, _, planted, merged = resized
    images = merged["images"]
    assert [img["id"] for img in images] == list(range(1, len(images) + 1))
    assert [(SOURCES.index(img["source"]), img["orig_file_name"]) for img in images] == sorted(
        (SOURCES.index(source), img["file_name"]) for source, coco in planted["cocos"].items()
        for img in coco["images"])
    raw_images = {(source, img["id"]): img for source, coco in planted["cocos"].items()
                  for img in coco["images"]}
    for img in images:
        orig = raw_images[img["source"], img["orig_id"]]
        assert (img["width"], img["height"]) == (SIZE, SIZE)
        assert (img["orig_width"], img["orig_height"]) == (SIDE, SIDE)
        assert img["orig_file_name"] == orig["file_name"] and img["level"] == orig["level"]


def test_colliding_file_names_get_the_source_prefix(resized):
    _, _, planted, merged = resized
    names = [img["file_name"] for img in merged["images"]]
    assert len(set(names)) == len(names)
    shared = planted["shared_file_name"]
    assert sorted(n for n in names if shared in n) == [f"test2019_{shared}", f"val2019_{shared}"]
    assert all(img["file_name"] == img["orig_file_name"] for img in merged["images"]
               if img["orig_file_name"] != shared)


def test_boxes_and_areas_are_scaled_and_no_object_is_lost(resized):
    _, _, planted, merged = resized
    raw_anns = defaultdict(list)
    for source, coco in planted["cocos"].items():
        for ann in coco["annotations"]:
            raw_anns[source, ann["image_id"]].append(ann)
    new_anns = defaultdict(list)
    for ann in merged["annotations"]:
        new_anns[ann["image_id"]].append(ann)
    assert len(merged["annotations"]) == sum(len(c["annotations"]) for c in planted["cocos"].values())
    assert [ann["id"] for ann in merged["annotations"]] == list(range(1, len(merged["annotations"]) + 1))
    scale = SIZE / SIDE
    for img in merged["images"]:
        old_anns = sorted(raw_anns[img["source"], img["orig_id"]], key=lambda a: a["id"])
        for old, new in zip(old_anns, new_anns[img["id"]], strict=True):
            assert new["category_id"] == old["category_id"] and new["iscrowd"] == 0
            assert new["bbox"] == pytest.approx([v * scale for v in old["bbox"]], abs=0.01)
            assert new["area"] == pytest.approx(old["area"] * scale * scale, abs=0.01)
    assert merged["categories"] == planted["cocos"]["val2019"]["categories"]


def test_resized_images_and_drawings_exist(resized):
    _, data, _, merged = resized
    for img in merged["images"]:
        with Image.open(data / "checkout" / img["file_name"]) as image:
            assert image.size == (SIZE, SIZE) and image.format == "JPEG"
    assert len(list((data / "checkout").iterdir())) == len(merged["images"])
    assert len(list((data / "draw").glob("*.jpg"))) == 5


def test_rerun_skips_done_images_and_redoes_wrong_ones(resized):
    raw, data, _, merged = resized
    out = data / "checkout"
    names = [img["file_name"] for img in merged["images"]]
    Image.new("RGB", (10, 10)).save(out / names[0])
    mtimes = {name: (out / name).stat().st_mtime_ns for name in names[1:]}
    ann_bytes = (data / "ann.json").read_bytes()
    main(["--raw", str(raw), "--out-dir", str(out), "--ann-out", str(data / "ann.json"), "--size", str(SIZE),
          "--workers", "1"])
    with Image.open(out / names[0]) as image:
        assert image.size == (SIZE, SIZE)
    assert {name: (out / name).stat().st_mtime_ns for name in names[1:]} == mtimes
    assert (data / "ann.json").read_bytes() == ann_bytes


def test_invalid_sources_fail_loudly(resized, tmp_path):
    raw, _, planted, _ = resized
    cases = [
        (lambda c: c["test2019"]["images"][0].update(height=SIDE - 10), "not square"),
        (lambda c: c["test2019"]["images"][0].pop("level"), "level"),
        (lambda c: c["val2019"]["annotations"][0].pop("area"), "area"),
        (lambda c: c["val2019"].update(categories=c["val2019"]["categories"][:-1]), "categories"),
    ]
    for mutate, message in cases:
        cocos = copy.deepcopy(planted["cocos"])
        mutate(cocos)
        with pytest.raises(ValueError, match=message):
            merge_annotations(cocos, SIZE)

    src = tmp_path / "wide.jpg"
    Image.new("RGB", (SIDE, SIDE - 10)).save(src)
    with pytest.raises(ValueError, match="annotation says"):
        resize_one((str(src), str(tmp_path / "out.jpg"), SIZE, 95, SIDE))


def test_lost_objects_stop_the_tool(resized, tmp_path):
    _, _, planted, _ = resized
    cocos = copy.deepcopy(planted["cocos"])
    cocos["test2019"]["annotations"][0]["image_id"] = 10**6  # annotation without an image
    for source in SOURCES:
        save_json(tmp_path / f"instances_{source}.json", cocos[source])
    with pytest.raises(SystemExit, match="object count changed"):
        main(["--raw", str(tmp_path), "--out-dir", str(tmp_path / "out"),
              "--ann-out", str(tmp_path / "a.json")])
