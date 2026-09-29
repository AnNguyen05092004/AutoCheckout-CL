# Class-incremental metrics: /data/runs/P2_eb4 (val)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1103 | 0.0926 | - | - | 0.1103 | 0.0926 | 0.1706 | 0.1425 |
| 2 | 0.1308 | 0.1056 | 0.0829 | 0.0699 | 0.0822 | 0.0685 | 0.1137 | 0.0948 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 2)
| stage \ group | 1 | 2 |
|---|---|---|
| 1 | 0.1103 | - |
| 2 | 0.0829 | 0.1308 |

Forgetting (average over groups < 2): 0.0274
Average of last row: 0.1069

Wrote /data/runs/P2_eb4/metrics_cl_val.json
