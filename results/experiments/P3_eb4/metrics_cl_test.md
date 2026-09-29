# Class-incremental metrics: /data/runs/P3_eb4 (test)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.8237 | 0.7033 | - | - | 0.8237 | 0.7033 | 0.8604 | 0.7340 |
| 2 | 0.6499 | 0.5520 | 0.8230 | 0.7014 | 0.7743 | 0.6598 | 0.8179 | 0.6960 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 2)
| stage \ group | 1 | 2 |
|---|---|---|
| 1 | 0.8237 | - |
| 2 | 0.8230 | 0.6499 |

Forgetting (average over groups < 2): 0.0007
Average of last row: 0.7365

Wrote /data/runs/P3_eb4/metrics_cl_test.json
