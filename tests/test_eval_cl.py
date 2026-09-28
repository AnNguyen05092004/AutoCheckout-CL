import sys

import pytest

from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, save_predictions
from tests.coco_helpers import make_ann, make_gt, make_image, three_task_config
from tools import eval_cl


def _write_run(tmp_path):
    cfg = three_task_config()
    cfg_path = tmp_path / "tasks.json"
    cfg.save(cfg_path)

    gt = make_gt(
        images=[make_image(1), make_image(2)],
        annotations=[make_ann(1, 1, 0, [10, 10, 20, 20]), make_ann(2, 2, 2, [10, 10, 20, 20])],
        num_labels=6,
    )
    ann_path = tmp_path / "test_full.json"
    save_json(ann_path, gt)
    ann_md5 = md5_file(ann_path)

    # A real model's predictions at stage t cover every class learned so far, not just the
    # newest task; stage 2's file must still detect stage 1's object for mAP@P to be meaningful.
    run_dir = tmp_path / "run"
    stage_rows = {1: [(1, 0, 0)], 2: [(1, 0, 0), (2, 0, 2)]}
    for stage, rows in stage_rows.items():
        stage_dir = run_dir / f"task_{stage}"
        image_id, query, label = zip(*rows, strict=True)
        meta = {"task_id": stage, "seen_classes": cfg.seen_classes(stage), "split": "test",
                "ann_md5": ann_md5}
        preds = Predictions(
            image_id=list(image_id), query=list(query), label=list(label),
            score=[0.9] * len(rows), boxes=[[10, 10, 30, 30]] * len(rows),
            meta=meta,
        )
        save_predictions(stage_dir / "pred_test.npz", preds)
    return run_dir, ann_path, cfg_path


def test_evaluate_run_end_to_end(tmp_path):
    run_dir, ann_path, cfg_path = _write_run(tmp_path)
    cfg = three_task_config()
    result = eval_cl.evaluate_run(run_dir, "test", ann_path, cfg)
    assert set(result["stages"]) == {1, 2}
    assert result["stages"][2]["m1"]["mAP_P"]["AP50"] == pytest.approx(1.0)
    assert result["matrix"][2] == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}
    markdown = eval_cl.to_markdown(result)
    assert "Accuracy matrix" in markdown


def test_main_writes_metrics_file(tmp_path, monkeypatch, capsys):
    run_dir, ann_path, cfg_path = _write_run(tmp_path)
    argv = ["eval_cl", "--run-dir", str(run_dir), "--split", "test", "--ann", str(ann_path),
            "--task-config", str(cfg_path)]
    monkeypatch.setattr(sys, "argv", argv)
    eval_cl.main()
    out_path = run_dir / "metrics_cl_test.json"
    assert out_path.is_file()
    saved = load_json(out_path)
    assert saved["task_config"] == "toy-3task"
    captured = capsys.readouterr()
    assert "Wrote" in captured.out


def test_ann_md5_mismatch_is_refused(tmp_path):
    run_dir, ann_path, cfg_path = _write_run(tmp_path)
    # Corrupt the annotation file after predictions were generated against the original.
    gt = load_json(ann_path)
    gt["images"].append({"id": 99, "file_name": "extra.jpg", "width": 800, "height": 800, "level": "easy"})
    save_json(ann_path, gt)

    cfg = three_task_config()
    with pytest.raises(ValueError, match="ann_md5"):
        eval_cl.evaluate_run(run_dir, "test", ann_path, cfg)
