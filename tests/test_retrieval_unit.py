"""B3 retrieval baseline: pure parts (crop geometry, memory selection, classifier, output rows)."""

from collections import Counter

import numpy as np
import pytest
import torch

from autocheckout.predictions import Predictions
from baselines.retrieval import box_rows, class_confidence, classify, crop_box, select_annotations

CPU = torch.device("cpu")


# --- crop geometry -------------------------------------------------------------------------


def test_crop_box_grows_by_margin_of_box_size():
    # 20 x 40 box, margin 0.1: 2 px left/right, 4 px top/bottom
    assert crop_box([10, 20, 30, 60], 0.1, 100, 100) == (8, 16, 32, 64)


def test_crop_box_clips_to_image():
    # 55 x 30 box, margin 0.5: grows 27.5 / 15 px, then clipped at 0 (left) and 100 (bottom)
    assert crop_box([-5, 90, 50, 120], 0.5, 100, 100) == (0, 75, 78, 100)


def test_crop_box_rounds_outwards_and_keeps_one_pixel():
    assert crop_box([10.2, 10.7, 20.1, 20.9], 0.0, 100, 100) == (10, 10, 21, 21)
    assert crop_box([50, 50, 50, 50], 0.1, 100, 100) == (50, 50, 51, 51)
    assert crop_box([120, 10, 130, 20], 0.0, 100, 100) == (99, 10, 100, 20)  # box right of the image


# --- memory selection ----------------------------------------------------------------------


def _ann(ann_id, image_id, label):
    return {"id": ann_id, "image_id": image_id, "category_id": label, "bbox": [0, 0, 1, 1]}


# label 0: 3 boxes in image 1, 3 in image 2, 1 in image 3; label 1: 2 boxes in image 1;
# label 9 belongs to another task.
ANNS = ([_ann(i, 1, 0) for i in range(3)] + [_ann(10 + i, 2, 0) for i in range(3)] + [_ann(20, 3, 0)]
        + [_ann(30 + i, 1, 1) for i in range(2)] + [_ann(40 + i, 4, 9) for i in range(5)])


def test_select_annotations_only_given_labels_at_most_per_class():
    picked = select_annotations(ANNS, [0, 1], per_class=3, seed=0)
    labels = [a["category_id"] for a in picked]
    assert labels.count(0) == 3 and labels.count(1) == 2 and 9 not in labels


def test_select_annotations_spreads_over_images():
    def images(per_class):
        return sorted(a["image_id"] for a in select_annotations(ANNS, [0], per_class=per_class, seed=0))

    assert images(3) == [1, 2, 3]
    # 5 of 7: one per image first, then a second one in the two images that have more
    assert images(5) == [1, 1, 2, 2, 3]


def test_select_annotations_follows_the_seed():
    def ids(seed):
        return [a["id"] for a in select_annotations(ANNS, [0, 1], per_class=3, seed=seed)]

    assert ids(0) == ids(0)
    assert any(ids(seed) != ids(0) for seed in range(1, 6))


def test_select_annotations_refuses_a_label_without_box():
    with pytest.raises(ValueError, match="label 5"):
        select_annotations(ANNS, [0, 5], per_class=3, seed=0)


# --- classifier ----------------------------------------------------------------------------


def _unit(*rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


# Class 0: one embedding very close to the query and two far away (its prototype is far);
# class 1: a tight cluster fairly close to the query; class 2: far. The prototype classifier
# therefore picks class 1, the 1-nearest-neighbour classifier picks class 0.
QUERY = _unit([1, 0.2, 0])
MEMORY = _unit([1, 0.25, 0], [0, 0, 1], [0, -1, 0], [0.9, 0.5, 0], [0.8, 0.6, 0], [0.85, 0.55, 0], [-1, 0, 0])
MEMORY_LABELS = np.array([0, 0, 0, 1, 1, 1, 2])


def test_prototype_mode_picks_the_nearest_class_mean():
    labels, conf = classify(QUERY, MEMORY, MEMORY_LABELS, 3, mode="prototype", k=1, temperature=0.07,
                            labels_per_box=3, device=CPU)
    assert labels.tolist() == [[1, 0, 2]]
    assert np.all(np.diff(conf, axis=1) <= 0) and conf.sum() == pytest.approx(1.0)


def test_knn_mode_votes_among_the_k_nearest():
    labels, conf = classify(QUERY, MEMORY, MEMORY_LABELS, 3, mode="knn", k=1, temperature=0.07,
                            labels_per_box=2, device=CPU)
    assert labels[0, 0] == 0 and conf[0].tolist() == pytest.approx([1.0, 0.0])
    # k = 4: the close class-0 embedding and the 3 of class 1; class 1 wins the weighted vote
    labels, conf = classify(QUERY, MEMORY, MEMORY_LABELS, 3, mode="knn", k=4, temperature=0.07,
                            labels_per_box=3, device=CPU)
    assert labels[0, :2].tolist() == [1, 0] and conf[0, 2] == 0.0
    assert conf.sum() == pytest.approx(1.0)


def test_class_confidence_is_a_softmax_summed_per_class():
    query, memory = torch.from_numpy(QUERY), torch.from_numpy(MEMORY)
    memory_labels = torch.from_numpy(MEMORY_LABELS)
    weights = torch.softmax(query @ memory.T / 0.5, dim=1)[0]
    conf = class_confidence(query, memory, memory_labels, 3, temperature=0.5)[0]
    expected = [weights[MEMORY_LABELS == c].sum().item() for c in range(3)]
    assert conf.tolist() == pytest.approx(expected)


@pytest.mark.parametrize("labels", [np.array([0, 0, 1]), np.array([0, 1, 2, 3])])
def test_classify_refuses_memory_not_covering_exactly_the_seen_classes(labels):
    memory = _unit(*np.ones((len(labels), 3)))
    with pytest.raises(ValueError, match="every label"):
        classify(QUERY, memory, labels, 3, mode="prototype", k=1, temperature=0.07, labels_per_box=3,
                 device=CPU)


# --- output rows ---------------------------------------------------------------------------


def _detections():
    """Image 1: boxes 0 and 1; image 2: box 0."""
    return Predictions(image_id=[1, 1, 2], query=[0, 1, 0], label=[0, 0, 0], score=[0.9, 0.5, 0.8],
                       boxes=[[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 5, 5]])


LABELS = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
CONF = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [1.0, 0.0, 0.0]], dtype=np.float32)


def test_box_rows_one_row_per_box_and_label():
    rows = box_rows(_detections(), LABELS, CONF, topk_per_image=100)
    # the last box has a single label with confidence > 0; zero-confidence rows are dropped
    per_box = Counter(zip(rows.image_id.tolist(), rows.query.tolist(), strict=True))
    assert per_box == {(1, 0): 3, (1, 1): 3, (2, 0): 1}
    assert rows.label.max() < 3
    assert np.all((rows.score >= 0) & (rows.score <= 1))
    top1 = rows.top1_per_query()
    assert len(top1) == 3
    assert sorted(zip(top1.query.tolist(), top1.label.tolist(), top1.score.tolist(), strict=True)) == [
        (0, 1, pytest.approx(0.63)), (0, 2, pytest.approx(0.8)), (1, 0, pytest.approx(0.3))]


def test_box_rows_keeps_the_best_rows_of_each_image():
    rows = box_rows(_detections(), LABELS, CONF, topk_per_image=2)
    kept = sorted(zip(rows.image_id.tolist(), rows.query.tolist(), rows.label.tolist(), strict=True))
    # image 1 scores: 0.63 (box 0, label 1), 0.3 (box 1, label 0), then 0.18, 0.15, ...
    assert kept == [(1, 0, 1), (1, 1, 0), (2, 0, 2)]
