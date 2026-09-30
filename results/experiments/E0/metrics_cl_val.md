# Class-incremental metrics: /data/runs/E0 (val)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.9957 | 0.8620 | 0.9954 | 0.8573 | 0.9953 | 0.8578 | 0.9953 | 0.8578 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 5)
| stage \ group | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.9959 | 0.9958 | 0.9959 | 0.9935 | 0.9957 |

Forgetting (average over groups < 5): -
Average of last row: 0.9954

Wrote /data/runs/E0/metrics_cl_val.json
