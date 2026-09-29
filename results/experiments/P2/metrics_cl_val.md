# Class-incremental metrics: /data/runs/P2 (val)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0416 | 0.0346 | - | - | 0.0416 | 0.0346 | 0.0653 | 0.0545 |
| 2 | 0.0168 | 0.0134 | 0.0386 | 0.0320 | 0.0329 | 0.0272 | 0.0445 | 0.0369 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 2)
| stage \ group | 1 | 2 |
|---|---|---|
| 1 | 0.0416 | - |
| 2 | 0.0386 | 0.0168 |

Forgetting (average over groups < 2): 0.0030
Average of last row: 0.0277

Wrote /data/runs/P2/metrics_cl_val.json
