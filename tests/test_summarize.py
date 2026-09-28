import sys

from autocheckout.io import save_json
from tools import summarize


def _ap(v):
    return {"AP": v, "AP50": v, "AP75": v, "n_images": 1}


def _write_experiment(tmp_path, name, stages, matrix, forgetting):
    run_dir = tmp_path / name
    cl = {
        "run_dir": str(run_dir), "split": "test", "ann": "x", "ann_md5": "x", "task_config": name,
        "stages": {
            str(t): {
                "task_id": t, "seen_classes": 2 * t,
                "m1": {"mAP_C": _ap(stages[t]), "mAP_A": _ap(stages[t]),
                       "mAP_P": _ap(stages[t]) if t > 1 else None},
                "m2": {"overall": _ap(stages[t]),
                       "by_level": {"easy": _ap(stages[t]), "medium": _ap(stages[t]),
                                    "hard": _ap(stages[t])}},
            }
            for t in stages
        },
        "matrix": {str(t): {str(g): v for g, v in row.items()} for t, row in matrix.items()},
        "forgetting": forgetting,
    }
    save_json(run_dir / "metrics_cl_test.json", cl)

    last_stage = max(stages)
    count = {
        "run_dir": str(run_dir), "task_config": name,
        "stages": {str(last_stage): {"test": {"overall": {"cAcc": 0.5, "ACD": 1.0, "mCCD": 0.2, "mCIoU": 0.8,
                                                            "K": 4, "K_eff": 4}}}},
    }
    save_json(run_dir / "metrics_count_test.json", count)
    return run_dir


def test_summarize_end_to_end(tmp_path):
    run_a = _write_experiment(
        tmp_path, "E3",
        stages={1: 0.9, 2: 0.7},
        matrix={1: {1: 0.9}, 2: {1: 0.6, 2: 0.8}},
        forgetting={"per_group": {1: 0.3}, "average": 0.3, "avg_last_row": 0.7},
    )
    run_b = _write_experiment(
        tmp_path, "E0",
        stages={2: 0.95},
        matrix={2: {1: 0.95, 2: 0.95}},
        forgetting={"per_group": {}, "average": None, "avg_last_row": 0.95},
    )

    out = tmp_path / "results" / "summary"
    argv = ["summarize", "--runs", str(run_a), str(run_b), "--out", str(out)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        summarize.main()
    finally:
        sys.argv = old_argv

    assert out.with_suffix(".md").is_file()
    assert out.with_name("summary_stages.csv").is_file()
    assert out.with_name("summary_final.csv").is_file()
    assert out.with_name("summary_mapA.png").is_file()
    assert out.with_name("summary_matrix_E3.png").is_file()
    assert out.with_name("summary_matrix_E0.png").is_file()

    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "E3" in markdown and "E0" in markdown
    stages_csv = out.with_name("summary_stages.csv").read_text(encoding="utf-8")
    assert "E3" in stages_csv


def test_build_tables_skips_runs_without_cl_metrics(tmp_path):
    empty_run = tmp_path / "no_metrics"
    empty_run.mkdir()
    run_a = _write_experiment(tmp_path, "E3", stages={1: 0.9}, matrix={1: {1: 0.9}},
                               forgetting={"per_group": {}, "average": None, "avg_last_row": 0.9})
    stage_rows, final_rows = summarize.build_tables([run_a, empty_run])
    assert {row["experiment"] for row in stage_rows} == {"E3"}
    assert {row["experiment"] for row in final_rows} == {"E3"}
