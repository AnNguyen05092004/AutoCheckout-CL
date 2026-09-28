import numpy as np

from autocheckout.predictions import Predictions, load_predictions, save_predictions


def toy_predictions():
    # image 1: query 0 has two classes (0.9 for label 2, 0.3 for label 1); query 5 one class.
    # image 2: query 0 has two classes, the best one listed second.
    return Predictions(
        image_id=[1, 1, 1, 2, 2],
        query=[0, 0, 5, 0, 0],
        label=[2, 1, 0, 3, 4],
        score=[0.9, 0.3, 0.5, 0.2, 0.7],
        boxes=[[0, 0, 10, 20]] * 5,
        meta={"task_id": 1, "split": "val"},
    )


def test_save_load_round_trip(tmp_path):
    preds = toy_predictions()
    path = tmp_path / "pred_val.npz"
    save_predictions(path, preds)
    loaded = load_predictions(path)
    assert loaded.meta == preds.meta
    for name in ("image_id", "query", "label", "score", "boxes"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(preds, name))
    assert not list(tmp_path.glob("*.tmp*"))


def test_top1_per_query_keeps_best_class_of_each_query():
    top1 = toy_predictions().top1_per_query()
    rows = sorted(zip(top1.image_id.tolist(), top1.query.tolist(), top1.label.tolist(), strict=True))
    assert rows == [(1, 0, 2), (1, 5, 0), (2, 0, 4)]


def test_to_coco_results_converts_boxes_and_labels():
    results = toy_predictions().to_coco_results(label_to_category={0: 10, 1: 11, 2: 12, 3: 13, 4: 14})
    assert results[0] == {"image_id": 1, "category_id": 12, "bbox": [0.0, 0.0, 10.0, 20.0],
                          "score": results[0]["score"]}
    assert abs(results[0]["score"] - 0.9) < 1e-6


def test_empty_predictions():
    empty = Predictions([], [], [], [], np.zeros((0, 4)))
    assert len(empty.top1_per_query()) == 0
    assert empty.to_coco_results() == []


def test_saved_files_get_normal_permissions(tmp_path):
    import os

    from autocheckout.io import default_file_mode, save_json

    save_predictions(tmp_path / "p.npz", toy_predictions())
    save_json(tmp_path / "x.json", {"a": 1})
    for name in ("p.npz", "x.json"):
        assert os.stat(tmp_path / name).st_mode & 0o777 == default_file_mode()
