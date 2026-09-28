import pytest

from autocheckout.groups import coco_group_keys, group_keys, parse_file_name, sku_multiset


def test_parse_file_name():
    assert parse_file_name("20180827-13-42-20-204.jpg") == ("20180827-13-42-20", "204")
    for bad in ("20180827-13-42-20.jpg", "20180827-13-42-20-204.png", "val2019_20180827-13-42-20-204.jpg"):
        with pytest.raises(ValueError, match="does not match"):
            parse_file_name(bad)


def test_sku_multiset_ignores_order():
    assert sku_multiset([3, 1, 3]) == sku_multiset([1, 3, 3]) == ((1, 1), (3, 2))


def test_group_keys_union_by_suffix_and_multiset():
    a, b, c = sku_multiset([1, 2]), sku_multiset([3]), sku_multiset([4, 4])
    names = ["20180101-10-00-00-5.jpg",   # 0: suffix 5
             "20180101-10-00-11-5.jpg",   # 1: suffix 5, other basket -> same group as 0
             "20180102-09-00-00-12.jpg",  # 2: suffix 12, same multiset as 1 -> joins 0/1
             "20180103-09-00-00-30.jpg",  # 3: suffix 30, own basket
             "20180103-09-00-11-40.jpg",  # 4: empty image
             "20180103-09-00-22-41.jpg"]  # 5: empty image, must not merge with 4
    multisets = [a, b, b, c, (), ()]
    keys = group_keys(names, multisets)
    assert keys == ["g5", "g5", "g5", "g30", "g40", "g41"]
    # Keys do not depend on the input order.
    order = [5, 3, 1, 4, 2, 0]
    assert group_keys([names[i] for i in order], [multisets[i] for i in order]) == [keys[i] for i in order]


def test_group_key_is_smallest_suffix_in_numeric_order():
    names = ["20180101-10-00-00-100.jpg", "20180101-10-00-11-99.jpg"]
    same = sku_multiset([7])
    assert group_keys(names, [same, same]) == ["g99", "g99"]


def test_coco_group_keys_uses_orig_file_name():
    coco = {"images": [{"id": 1, "file_name": "val2019_20180101-10-00-00-5.jpg",
                        "orig_file_name": "20180101-10-00-00-5.jpg"},
                       {"id": 2, "file_name": "20180101-10-00-11-5.jpg"}],
            "annotations": [{"image_id": 1, "category_id": 1}, {"image_id": 2, "category_id": 2}]}
    assert coco_group_keys(coco) == {1: "g5", 2: "g5"}
