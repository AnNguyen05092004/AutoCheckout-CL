from autocheckout.io import load_json, save_json
from tools.subset_tasks import main, subset


def test_subset_keeps_first_images_and_their_annotations(tmp_path):
    coco = {"images": [{"id": i} for i in (5, 1, 3)], "categories": [{"id": 0}],
            "annotations": [{"id": 10 + i, "image_id": i} for i in (1, 3, 5, 5)]}
    small = subset(coco, 2)
    assert [img["id"] for img in small["images"]] == [1, 3]
    assert [ann["image_id"] for ann in small["annotations"]] == [1, 3]
    assert small["categories"] == coco["categories"]

    save_json(tmp_path / "src" / "train_task_1.json", coco)
    save_json(tmp_path / "src" / "manifest.json", {"files": {}})
    main(["--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"), "--images", "1"])
    assert load_json(tmp_path / "dst" / "train_task_1.json")["images"] == [{"id": 1}]
    assert not (tmp_path / "dst" / "manifest.json").exists()
