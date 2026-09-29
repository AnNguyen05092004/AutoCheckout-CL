# Class-incremental metrics: /data/runs/P2_eb4 (test)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0999 | 0.0835 | - | - | 0.0999 | 0.0835 | 0.1588 | 0.1324 |
| 2 | 0.1170 | 0.0967 | 0.0735 | 0.0615 | 0.0726 | 0.0606 | 0.1020 | 0.0852 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 2)
| stage \ group | 1 | 2 |
|---|---|---|
| 1 | 0.0999 | - |
| 2 | 0.0735 | 0.1170 |

Forgetting (average over groups < 2): 0.0264
Average of last row: 0.0952

Wrote /data/runs/P2_eb4/metrics_cl_test.json
