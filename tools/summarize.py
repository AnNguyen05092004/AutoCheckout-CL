"""CLI for V6: turn saved metrics JSONs from one or more runs into comparison tables and plots.

Usage::

    python -m tools.summarize --runs RUN1 RUN2 ... --out results/summary

Reads only ``<run>/metrics_cl_test.json`` (V2) and ``<run>/metrics_count_test.json`` (V3)
from each run directory -- never predictions -- so it can be re-run instantly after any
metrics change. Each run directory's own name is used as the experiment label (e.g. ``E3``,
``E4``). Writes:

- ``<out>.md``: per-stage mAP@C/P/A (AP50, AP) and M2 AP50/AP table, plus a per-experiment
  final-stage summary (forgetting, cAcc/ACD/mCCD/mCIoU).
- ``<out>_stages.csv`` and ``<out>_final.csv``: the same two tables as CSV.
- ``<out>_mapA.png``: mAP@A (AP50) vs stage, one line per experiment.
- ``<out>_matrix_<experiment>.png``: accuracy-matrix heatmap, one per experiment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autocheckout.io import load_json  # noqa: E402

STAGE_FIELDS = [
    "experiment", "stage",
    "mAP_C_AP50", "mAP_C_AP", "mAP_P_AP50", "mAP_P_AP", "mAP_A_AP50", "mAP_A_AP",
    "M2_AP50", "M2_AP",
]
FINAL_FIELDS = ["experiment", "last_stage", "forgetting_avg", "avg_last_row", "cAcc", "ACD", "mCCD", "mCIoU"]


def load_run_metrics(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cl_path = run_dir / "metrics_cl_test.json"
    count_path = run_dir / "metrics_count_test.json"
    cl = load_json(cl_path) if cl_path.is_file() else None
    count = load_json(count_path) if count_path.is_file() else None
    return cl, count


def build_tables(runs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for run_dir in runs:
        name = run_dir.name
        cl, count = load_run_metrics(run_dir)
        if cl is None:
            continue  # nothing to report for this run (metrics_cl_test.json missing)
        stages = cl["stages"]
        for stage_key in sorted(stages, key=int):
            s = stages[stage_key]
            c, p, a, m2 = s["m1"]["mAP_C"], s["m1"]["mAP_P"], s["m1"]["mAP_A"], s["m2"]["overall"]
            stage_rows.append({
                "experiment": name, "stage": int(stage_key),
                "mAP_C_AP50": c["AP50"], "mAP_C_AP": c["AP"],
                "mAP_P_AP50": p["AP50"] if p else None, "mAP_P_AP": p["AP"] if p else None,
                "mAP_A_AP50": a["AP50"], "mAP_A_AP": a["AP"],
                "M2_AP50": m2["AP50"], "M2_AP": m2["AP"],
            })

        last_stage = max(int(k) for k in stages)
        forgetting = cl["forgetting"]
        final_row = {
            "experiment": name, "last_stage": last_stage,
            "forgetting_avg": forgetting["average"], "avg_last_row": forgetting["avg_last_row"],
            "cAcc": None, "ACD": None, "mCCD": None, "mCIoU": None,
        }
        if count is not None and str(last_stage) in count["stages"]:
            overall = count["stages"][str(last_stage)]["test"]["overall"]
            final_row.update({k: overall[k] for k in ("cAcc", "ACD", "mCCD", "mCIoU")})
        final_rows.append(final_row)
    return stage_rows, final_rows


def _fmt(x: Any) -> str:
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"


def write_csv(rows: list[dict[str, Any]], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def to_markdown(stage_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> str:
    lines = ["# Experiment summary", "", "## Per stage",
              "| experiment | stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | "
              "mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in stage_rows:
        lines.append(
            f"| {row['experiment']} | {row['stage']} | {_fmt(row['mAP_C_AP50'])} | {_fmt(row['mAP_C_AP'])} | "
            f"{_fmt(row['mAP_P_AP50'])} | {_fmt(row['mAP_P_AP'])} | {_fmt(row['mAP_A_AP50'])} | "
            f"{_fmt(row['mAP_A_AP'])} | {_fmt(row['M2_AP50'])} | {_fmt(row['M2_AP'])} |"
        )
    lines += ["", "## Final stage (forgetting and counting metrics)",
              "| experiment | last stage | forgetting (avg) | avg last row | cAcc | ACD | mCCD | mCIoU |",
              "|---|---|---|---|---|---|---|---|"]
    for row in final_rows:
        lines.append(
            f"| {row['experiment']} | {row['last_stage']} | {_fmt(row['forgetting_avg'])} | "
            f"{_fmt(row['avg_last_row'])} | {_fmt(row['cAcc'])} | {_fmt(row['ACD'])} | "
            f"{_fmt(row['mCCD'])} | {_fmt(row['mCIoU'])} |"
        )
    return "\n".join(lines)


def plot_map_a(stage_rows: list[dict[str, Any]], out_path: Path) -> None:
    by_experiment: dict[str, list[tuple[int, float]]] = {}
    for row in stage_rows:
        by_experiment.setdefault(row["experiment"], []).append((row["stage"], row["mAP_A_AP50"]))

    fig, ax = plt.subplots()
    for experiment, points in sorted(by_experiment.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=experiment)
    ax.set_xlabel("stage")
    ax.set_ylabel("mAP@A (AP50)")
    ax.set_title("mAP@A vs stage")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_matrix_heatmap(matrix: dict[str, dict[str, float]], out_path: Path, title: str) -> None:
    stages = sorted(matrix, key=int)
    groups = sorted({g for row in matrix.values() for g in row}, key=int)
    data = np.full((len(stages), len(groups)), np.nan)
    for i, stage in enumerate(stages):
        for j, group in enumerate(groups):
            if group in matrix[stage]:
                data[i, j] = matrix[stage][group]

    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(groups)), labels=groups)
    ax.set_yticks(range(len(stages)), labels=stages)
    ax.set_xlabel("task group")
    ax.set_ylabel("stage")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="AP50")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="V6: summarize experiment metrics into tables and plots")
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    stage_rows, final_rows = build_tables(args.runs)
    if not stage_rows:
        raise ValueError("none of the given runs has a metrics_cl_test.json")

    write_csv(stage_rows, STAGE_FIELDS, args.out.with_name(args.out.name + "_stages.csv"))
    write_csv(final_rows, FINAL_FIELDS, args.out.with_name(args.out.name + "_final.csv"))
    markdown = to_markdown(stage_rows, final_rows)
    args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")

    plot_map_a(stage_rows, args.out.with_name(args.out.name + "_mapA.png"))
    for run_dir in args.runs:
        cl, _ = load_run_metrics(run_dir)
        if cl is None:
            continue
        plot_matrix_heatmap(
            cl["matrix"], args.out.with_name(f"{args.out.name}_matrix_{run_dir.name}.png"),
            title=f"Accuracy matrix: {run_dir.name}",
        )

    print(markdown)
    print(f"\nWrote {args.out}.md, {args.out}_stages.csv, {args.out}_final.csv "
          f"and plots under {args.out.parent}")


if __name__ == "__main__":
    main()
