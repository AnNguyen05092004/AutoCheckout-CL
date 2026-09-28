"""V4: pseudo-label audit — matching logic and an end-to-end run on the toy data."""

import pytest
import torch

import ppg_audit
from pdp_helpers import run_main


def test_matching_by_score_iou_and_class():
    gt = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50.]])
    gt_labels = torch.tensor([1, 2, 1])
    boxes = torch.tensor([[0, 0, 10, 9], [0, 1, 10, 10], [20, 20, 30, 31], [60, 60, 70, 70.]])
    labels = torch.tensor([1, 1, 3, 1])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    outcomes, matched = ppg_audit.match_pseudo_labels(boxes, labels, scores, gt, gt_labels)
    # the second box duplicates the first object (already taken) -> fp; the third has the wrong class
    assert outcomes == ["tp", "fp", "wrong_class", "fp"] and matched == [0, -1, 1, -1]
    assert ppg_audit.match_pseudo_labels(boxes, labels, scores, gt[:0], gt_labels[:0])[0] == ["fp"] * 4


@pytest.mark.slow
def test_audit_end_to_end(tmp_path, monkeypatch):
    args = run_main(tmp_path, monkeypatch)
    audit_args = ppg_audit.parse_args(["--audit_task", "2", "--audit_images", "4"])
    for key, value in vars(args).items():  # same model arguments as the run
        if key not in ("audit_task", "audit_images"):
            setattr(audit_args, key, value)
    audit_args.output_dir = str(tmp_path / "run")
    result = ppg_audit.run(audit_args)

    assert (tmp_path / "run/task_2/ppg_audit.json").exists()
    assert result["images"] == min(4, result["images"])
    counts = result["all"]
    assert counts.get("tp", 0) + counts.get("wrong_class", 0) + counts.get("fp", 0) == sum(
        sum(v for k, v in b.items() if k != "precision") for b in result["branches"].values())
    assert sum(sum(v for k, v in s.items() if k != "recall") for s in result["by_supercategory"].values()) \
        == result["earlier_objects"]
