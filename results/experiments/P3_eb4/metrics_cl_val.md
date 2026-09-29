# Class-incremental metrics: /data/runs/P3_eb4 (val)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.8289 | 0.7057 | - | - | 0.8289 | 0.7057 | 0.8643 | 0.7359 |
| 2 | 0.6692 | 0.5656 | 0.8272 | 0.7025 | 0.7821 | 0.6640 | 0.8246 | 0.6997 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 2)
| stage \ group | 1 | 2 |
|---|---|---|
| 1 | 0.8289 | - |
| 2 | 0.8272 | 0.6692 |

Forgetting (average over groups < 2): 0.0017
Average of last row: 0.7482

Wrote /data/runs/P3_eb4/metrics_cl_val.json
