import pytest
from pycocotools.cocoeval import COCOeval

from autocheckout.cl_metrics import (
    check_ann_md5,
    check_stage_meta,
    discover_stage_predictions,
    filter_predictions_overlapping_unlearned,
    group_ap,
    images_with_labels,
    load_coco,
    m1_stage,
    m2_stage,
    matrix_forgetting,
)
from tests.coco_helpers import make_ann, make_gt, make_image, make_preds, three_task_config


def test_perfect_predictions_give_ap_1():
    gt = make_gt(
        images=[make_image(1)],
        annotations=[make_ann(1, 1, 0, [10, 10, 20, 20])],
        num_labels=1,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([(1, 0, 0, 0.9, [10, 10, 30, 30])])
    stats = group_ap(coco_gt, preds, [0])
    assert stats["AP"] == pytest.approx(1.0)
    assert stats["AP50"] == pytest.approx(1.0)
    assert stats["AP75"] == pytest.approx(1.0)


def test_missed_object_gives_hand_computable_ap():
    # Two images, one GT box of the same class each; only the first gets a matching detection.
    # COCO AP is the mean precision over 101 recall thresholds (0.00..1.00 step 0.01): the
    # single true positive reaches recall 0.5 at precision 1.0, so recall thresholds 0.00..0.50
    # (51 points) interpolate to precision 1.0 and 0.51..1.00 (50 points) to 0 -> AP = 51/101.
    gt = make_gt(
        images=[make_image(1), make_image(2)],
        annotations=[make_ann(1, 1, 0, [10, 10, 20, 20]), make_ann(2, 2, 0, [10, 10, 20, 20])],
        num_labels=1,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([(1, 0, 0, 0.9, [10, 10, 30, 30])])  # image 2's object is never detected
    stats = group_ap(coco_gt, preds, [0])
    expected = 51 / 101
    assert stats["AP"] == pytest.approx(expected)
    assert stats["AP50"] == pytest.approx(expected)
    assert stats["AP75"] == pytest.approx(expected)


def test_wrong_class_prediction_does_not_count_as_a_match():
    gt = make_gt(
        images=[make_image(1)],
        annotations=[make_ann(1, 1, 0, [10, 10, 20, 20])],
        num_labels=2,
    )
    coco_gt = load_coco(gt)
    # Box matches perfectly but the predicted label (1) is not the GT label (0).
    preds = make_preds([(1, 0, 1, 0.9, [10, 10, 30, 30])])
    stats = group_ap(coco_gt, preds, [0])
    assert stats["AP"] == 0.0
    assert stats["AP50"] == 0.0


def _mini_gt_and_filtered_preds(gt: dict, preds, labels: list[int]):
    """Hand-build the per-group GT file the original PDP pipeline would have produced
    (only the group's classes and only images containing one of them), and restrict
    detections to that image subset the way a DataLoader over that file would."""
    coco_gt = load_coco(gt)
    image_ids = set(images_with_labels(coco_gt, labels))
    mini = {
        "images": [img for img in gt["images"] if img["id"] in image_ids],
        "annotations": [a for a in gt["annotations"]
                         if a["category_id"] in labels and a["image_id"] in image_ids],
        "categories": [c for c in gt["categories"] if c["id"] in labels],
    }
    mini_coco = load_coco(mini)
    results = [r for r in preds.to_coco_results() if r["image_id"] in image_ids]
    return mini_coco, results


def test_m1_matches_coco_eval_on_a_hand_built_per_group_file():
    gt = make_gt(
        images=[make_image(1), make_image(2), make_image(3)],
        annotations=[
            make_ann(1, 1, 0, [10, 10, 20, 20]),
            make_ann(2, 1, 2, [50, 50, 15, 15]),  # image 1 also has a group-2 object
            make_ann(3, 2, 1, [5, 5, 10, 10]),
            make_ann(4, 3, 2, [0, 0, 5, 5]),
        ],
        num_labels=3,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([
        (1, 0, 0, 0.9, [10, 10, 30, 30]),
        (1, 1, 2, 0.8, [50, 50, 65, 65]),
        (2, 0, 1, 0.3, [5, 5, 15, 15]),  # weak, imperfect box
        (3, 0, 2, 0.95, [0, 0, 5, 5]),
    ])

    group1_labels = [0, 1]
    ours = group_ap(coco_gt, preds, group1_labels)

    mini_coco, results = _mini_gt_and_filtered_preds(gt, preds, group1_labels)
    coco_dt = mini_coco.loadRes(results) if results else mini_coco
    coco_eval = COCOeval(mini_coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    reference = {"AP": coco_eval.stats[0], "AP50": coco_eval.stats[1], "AP75": coco_eval.stats[2]}

    assert ours["AP"] == pytest.approx(reference["AP"])
    assert ours["AP50"] == pytest.approx(reference["AP50"])
    assert ours["AP75"] == pytest.approx(reference["AP75"])


def test_m1_stage_map_c_p_a_and_matrix_row():
    cfg = three_task_config()
    gt = make_gt(
        images=[make_image(1), make_image(2), make_image(3)],
        annotations=[
            make_ann(1, 1, 0, [10, 10, 20, 20]),
            make_ann(2, 2, 2, [10, 10, 20, 20]),
            make_ann(3, 3, 4, [10, 10, 20, 20]),
        ],
        num_labels=6,
    )
    coco_gt = load_coco(gt)
    # Stage 2: perfect detections for labels 0 and 2 (learned so far); label 4 not learned yet.
    preds = make_preds([
        (1, 0, 0, 0.9, [10, 10, 30, 30]),
        (2, 0, 2, 0.9, [10, 10, 30, 30]),
    ], meta={"task_id": 2, "seen_classes": 4})

    result = m1_stage(coco_gt, preds, cfg, stage=2)
    assert result["mAP_C"]["AP50"] == pytest.approx(1.0)  # task 2's own class (label 2)
    assert result["mAP_P"]["AP50"] == pytest.approx(1.0)  # task 1's class (label 0)
    assert result["mAP_A"]["AP50"] == pytest.approx(1.0)  # labels 0..3
    assert result["matrix_row"] == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}


def test_m1_stage_has_no_map_p_at_stage_1():
    cfg = three_task_config()
    gt = make_gt(images=[make_image(1)], annotations=[make_ann(1, 1, 0, [10, 10, 20, 20])], num_labels=6)
    coco_gt = load_coco(gt)
    preds = make_preds([(1, 0, 0, 0.9, [10, 10, 30, 30])], meta={"task_id": 1, "seen_classes": 2})
    result = m1_stage(coco_gt, preds, cfg, stage=1)
    assert result["mAP_P"] is None
    assert result["matrix_row"] == {1: pytest.approx(1.0)}


def test_m2_removes_exactly_the_predictions_overlapping_unlearned_gt():
    # seen_classes = 2 (labels 0, 1 learned); label 2 is not learned yet.
    gt = make_gt(
        images=[make_image(1)],
        annotations=[
            make_ann(1, 1, 0, [0, 0, 10, 10]),   # learned GT box A
            make_ann(2, 1, 2, [50, 50, 10, 10]),  # unlearned GT box B
        ],
        num_labels=3,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([
        (1, 0, 0, 0.9, [0, 0, 10, 10]),      # matches learned box A -> kept
        (1, 1, 1, 0.8, [50, 50, 60, 60]),    # overlaps unlearned box B (IoU=1) -> removed
        (1, 2, 0, 0.5, [200, 200, 210, 210]),  # no overlap with anything -> kept
    ])
    filtered = filter_predictions_overlapping_unlearned(preds, coco_gt, seen_classes=2)
    kept_queries = sorted(filtered.query.tolist())
    assert kept_queries == [0, 2]


def test_m2_overall_and_by_level():
    gt = make_gt(
        images=[make_image(1, level="easy"), make_image(2, level="hard")],
        annotations=[
            make_ann(1, 1, 0, [10, 10, 20, 20]),
            make_ann(2, 2, 0, [10, 10, 20, 20]),
            make_ann(3, 2, 2, [50, 50, 10, 10]),  # unlearned, image 2
        ],
        num_labels=3,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([
        (1, 0, 0, 0.9, [10, 10, 30, 30]),  # perfect match, easy
        (2, 0, 0, 0.9, [10, 10, 30, 30]),  # perfect match, hard
        (2, 1, 1, 0.7, [50, 50, 60, 60]),  # overlaps unlearned box -> removed by M2
    ])
    result = m2_stage(coco_gt, preds, seen_classes=2)
    assert result["overall"]["AP50"] == pytest.approx(1.0)
    assert result["by_level"]["easy"]["AP50"] == pytest.approx(1.0)
    assert result["by_level"]["hard"]["AP50"] == pytest.approx(1.0)


def test_matrix_forgetting_arithmetic():
    matrix = {1: {1: 0.9}, 2: {1: 0.5, 2: 0.8}, 3: {1: 0.4, 2: 0.6, 3: 0.7}}
    result = matrix_forgetting(matrix)
    assert result["per_group"] == {1: pytest.approx(0.5), 2: pytest.approx(0.2)}
    assert result["average"] == pytest.approx(0.35)
    assert result["avg_last_row"] == pytest.approx((0.4 + 0.6 + 0.7) / 3)


def test_matrix_forgetting_handles_a_single_stage_gracefully():
    # e.g. a joint-training upper bound (E0) that only has its last stage.
    matrix = {5: {1: 0.6, 2: 0.7, 3: 0.5, 4: 0.8, 5: 0.9}}
    result = matrix_forgetting(matrix)
    assert result["per_group"] == {}
    assert result["average"] is None
    assert result["avg_last_row"] == pytest.approx((0.6 + 0.7 + 0.5 + 0.8 + 0.9) / 5)


def test_discover_stage_predictions_skips_missing_and_unrelated_dirs(tmp_path):
    (tmp_path / "task_1").mkdir()
    (tmp_path / "task_1" / "pred_test.npz").write_bytes(b"")
    (tmp_path / "task_2").mkdir()  # no pred_test.npz: e.g. only pred_val.npz exists
    (tmp_path / "task_2" / "pred_val.npz").write_bytes(b"")
    (tmp_path / "task_5").mkdir()  # joint-training run: only the last stage present
    (tmp_path / "task_5" / "pred_test.npz").write_bytes(b"")
    (tmp_path / "not_a_task_dir").mkdir()

    stages = discover_stage_predictions(tmp_path, "test")
    assert set(stages) == {1, 5}


def test_check_ann_md5_refuses_mismatch():
    check_ann_md5({"ann_md5": "abc"}, "abc", context="ok")  # no raise
    check_ann_md5({}, "abc", context="no key present")  # no raise: nothing to check
    with pytest.raises(ValueError, match="ann_md5"):
        check_ann_md5({"ann_md5": "abc"}, "def", context="task_1/pred_test.npz")


def test_check_stage_meta_cross_checks_task_config():
    cfg = three_task_config()
    assert check_stage_meta({"task_id": 2, "seen_classes": 4}, cfg, 2) == 4
    assert check_stage_meta({}, cfg, 2) == 4  # meta without the keys: trust the task config
    with pytest.raises(ValueError, match="seen_classes"):
        check_stage_meta({"seen_classes": 3}, cfg, 2)
    with pytest.raises(ValueError, match="task_id"):
        check_stage_meta({"task_id": 3}, cfg, 2)
