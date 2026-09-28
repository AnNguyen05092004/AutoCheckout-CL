"""F6: pseudo-labels of earlier classes (PPG) — candidate selection, prototype check, label range."""

from types import SimpleNamespace

import torch
from ppg import prototype_matrix, select_candidates, select_pseudo_labels

import engine

PREV = 3  # classes 0..2 learned before; 3..4 current task; 5 = unused last slot
NUM_CLASSES = 6
LOGIT = {0.9: 2.2, 0.4: -0.405, 0.3: -0.85, 0.05: -2.9}  # sigmoid(logit) ~ key


def fake_teacher(entries, num_queries=8, dim=4):
    """entries: {query: (class, probability)}; every other logit is very low."""
    torch.manual_seed(0)
    logits = torch.full((1, num_queries, NUM_CLASSES), -9.0)
    for query, (cls, prob) in entries.items():
        logits[0, query, cls] = LOGIT[prob]
    boxes = torch.rand(1, num_queries, 4) * 0.5 + 0.1
    features = torch.randn(1, num_queries, dim)
    return logits, boxes, features


def test_candidates_are_ranked_per_query_and_exclude_current_and_future_classes():
    logits, boxes, _ = fake_teacher({0: (3, 0.9), 1: (1, 0.4), 2: (5, 0.9), 3: (0, 0.3), 4: (2, 0.9)})
    logits[0, 4, 0] = LOGIT[0.4]  # query 4 also likes class 0 less than class 2
    scores, labels, queries, cand_boxes = select_candidates(logits, boxes, PREV, topk=3)
    assert queries[0].tolist() == [4, 1, 3]  # queries 0 and 2 only score high on classes >= PREV
    assert labels[0].tolist() == [2, 1, 0]
    assert (labels < PREV).all()
    assert len(set(queries[0].tolist())) == len(queries[0])  # one candidate per query
    torch.testing.assert_close(cand_boxes[0], boxes[0, [4, 1, 3]])
    torch.testing.assert_close(scores[0], torch.sigmoid(torch.tensor([LOGIT[0.9], LOGIT[0.4], LOGIT[0.3]])))


def test_ppg_keeps_high_scores_and_prototype_verified_medium_scores():
    logits, boxes, features = fake_teacher({0: (0, 0.9), 1: (1, 0.4), 2: (2, 0.4), 3: (1, 0.3), 4: (0, 0.05)})
    features[0, 3] = -features[0, 1]  # query 3 points away from the class-1 prototype
    prototypes = {0: torch.randn(4), 1: features[0, 1].clone(), 2: -features[0, 2].clone()}
    protos, valid = prototype_matrix(prototypes, PREV, "cpu")
    scores, labels, queries, cand_boxes = select_candidates(logits, boxes, PREV, topk=5)
    feats = features[0, queries[0]]  # features taken at the candidates' query indices
    kwargs = dict(tau_high=0.5, tau_low=0.2, sim_thresh=0.5)

    kept_boxes, kept_labels = select_pseudo_labels(scores[0], labels[0], cand_boxes[0], feats, protos, valid,
                                                   mode="ppg", **kwargs)
    # query 0: high confidence; query 1: medium, same direction as the class-1 prototype;
    # query 2: medium but opposite to its prototype; query 3: medium (0.3) but not similar enough;
    # query 4: below tau_low
    assert kept_labels.tolist() == [0, 1]
    torch.testing.assert_close(kept_boxes, boxes[0, [0, 1]])

    _, threshold_labels = select_pseudo_labels(scores[0], labels[0], cand_boxes[0], feats, protos, valid,
                                               mode="threshold", **kwargs)
    assert threshold_labels.tolist() == [0]


def test_medium_candidates_of_classes_without_prototype_are_dropped():
    logits, boxes, features = fake_teacher({1: (1, 0.4)})
    protos, valid = prototype_matrix({0: torch.randn(4), 1: torch.zeros(4), 2: torch.randn(4)}, PREV, "cpu")
    scores, labels, queries, cand_boxes = select_candidates(logits, boxes, PREV, topk=2)
    feats = features[0, queries[0]]
    _, kept = select_pseudo_labels(scores[0], labels[0], cand_boxes[0], feats, protos, valid,
                                   mode="ppg", tau_high=0.5, tau_low=0.2, sim_thresh=-1.0)
    assert kept.tolist() == []


def test_trainer_appends_pseudo_labels_to_targets():
    logits, boxes, features = fake_teacher({0: (0, 0.9), 1: (2, 0.9)})
    teacher_out = SimpleNamespace(logits=logits, pred_boxes=boxes, last_hidden_state=features)
    trainer = SimpleNamespace(
        PREV_INTRODUCED_CLS=PREV, device=torch.device("cpu"), _old_prototypes=None,
        class_prototypes={c: torch.zeros(4) for c in range(NUM_CLASSES)},
        args=SimpleNamespace(pseudo_topk=50, pseudo="ppg", pseudo_thresh_high=0.5, pseudo_thresh_low=0.2,
                             prototype_sim_thresh=0.5))
    targets = [{"class_labels": torch.tensor([3]), "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]])}]
    engine.local_trainer.add_pseudo_labels(trainer, teacher_out, targets)
    assert targets[0]["class_labels"].tolist() == [3, 0, 2]
    torch.testing.assert_close(targets[0]["boxes"][1:], boxes[0, [0, 1]])


def test_original_selection_accepts_the_first_current_class():
    """Documents the original bug kept behind --ppg_legacy 1: `labels <= PREV` lets class PREV through."""
    trainer = SimpleNamespace(PREV_INTRODUCED_CLS=PREV, class_prototypes={}, args=SimpleNamespace())
    old_results = [{"scores": torch.tensor([0.9]), "labels": torch.tensor([PREV]), "boxes": torch.rand(1, 4)}]
    targets = [{"class_labels": torch.tensor([4]), "boxes": torch.rand(1, 4)}]
    engine.local_trainer.generate_old_class_pseudo_labels(trainer, old_results, targets)
    assert targets[0]["class_labels"].tolist() == [4, PREV]
