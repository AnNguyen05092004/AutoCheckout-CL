import copy
from collections import Counter

import pytest

from autocheckout.io import load_json
from tools.audit_rpc import MISSING, annotation_stats, image_rows, main

from synth_rpc import SIDE, make_raw_rpc


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    planted = make_raw_rpc(root)
    out = tmp_path_factory.mktemp("audit")
    main(["--raw", str(root), "--out", str(out), "--md5", "--workers", "2"])
    return load_json(out / "audit.json"), (out / "audit.md").read_text(), planted


def test_counts_fields_and_sizes(audited):
    result, _, planted = audited
    for source, coco in planted["cocos"].items():
        stats = result["sources"][source]
        assert stats["images"] == len(coco["images"])
        assert stats["objects"] == len(coco["annotations"])
        assert stats["images_per_level"] == dict(Counter(img["level"] for img in coco["images"]))
        assert stats["image_fields"]["level"] == len(coco["images"])
        assert stats["annotation_fields"]["area"] == stats["annotation_fields"]["iscrowd"] == stats["objects"]
        assert stats["file_sizes"] == stats["json_image_sizes"] == {f"{SIDE}x{SIDE}": len(coco["images"])}
        assert stats["missing_files"] == stats["size_differs_from_json"] == 0
        assert stats["jpg_files_not_in_json"] == 0
    assert result["categories"]["identical_in_all_sources"]
    assert result["categories"]["skus_per_supercategory"] == {"candy": 2, "drink": 5, "snack": 3, "tissue": 2}
    per_sku = result["objects_per_sku"]["per_sku"]
    total = sum(len(c["annotations"]) for c in planted["cocos"].values())
    assert sum(v["total"] for v in per_sku.values()) == total


def test_overlap_and_grouping(audited):
    result, markdown, planted = audited
    assert result["overlap"]["shared_image_ids"] == len(planted["cocos"]["val2019"]["images"])
    assert result["overlap"]["shared_file_names"] == 1
    assert result["overlap"]["shared_file_name_examples"] == [planted["shared_file_name"]]
    g = result["grouping"]
    assert g["suffixes"] == 37  # 33 regular runs + 7001..7004
    assert g["suffixes_in_both_sources"] == 1 and g["images_with_suffix_in_both_sources"] == 6
    assert g["suffixes_with_one_multiset"] == 35  # 7001 and 7004 hold two baskets each
    assert g["multisets_shared_by_several_suffixes"] == 1  # 7002 / 7003
    assert g["groups"] == 36 and g["suffix_groups_merged_by_multiset"] == 1
    assert g["groups_with_several_suffixes"] == 1 and g["images_in_groups_with_several_suffixes"] == 6
    assert g["groups_spanning_levels"] == 1 and g["groups_spanning_sources"] == 1
    assert g["largest_groups"][:3] == [6, 6, 5]
    assert "Final groups: 36" in markdown


def test_md5_duplicates(audited):
    result, _, planted = audited
    assert result["md5"]["duplicate_sets"] == 1
    assert result["md5"]["sets"] == [[f"test2019/{name}" for name in planted["duplicate"]]]
    assert result["md5"]["sets_spanning_groups"] == 0  # both shots of the same run


def test_missing_fields_are_counted(audited):
    _, _, planted = audited
    coco = copy.deepcopy(planted["cocos"]["val2019"])
    del coco["images"][0]["level"]
    del coco["annotations"][0]["area"]
    stats = annotation_stats(coco)
    assert stats["images_per_level"][MISSING] == 1
    assert stats["image_fields"]["level"] == len(coco["images"]) - 1
    assert stats["annotation_fields"]["area"] == len(coco["annotations"]) - 1


def test_unparseable_names_are_reported(audited):
    _, _, planted = audited
    cocos = copy.deepcopy(planted["cocos"])
    cocos["test2019"]["images"][0]["file_name"] = "IMG_0001.jpg"
    rows, unparseable = image_rows(cocos)
    assert unparseable == ["test2019/IMG_0001.jpg"]
    assert len(rows) == sum(len(c["images"]) for c in cocos.values()) - 1
