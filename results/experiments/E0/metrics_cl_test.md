# Class-incremental metrics: /data/runs/E0 (test)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.9938 | 0.8539 | 0.9921 | 0.8550 | 0.9923 | 0.8548 | 0.9923 | 0.8548 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 5)
| stage \ group | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 5 | 0.9921 | 0.9917 | 0.9936 | 0.9930 | 0.9938 |

Forgetting (average over groups < 5): -
Average of last row: 0.9928

Wrote /data/runs/E0/metrics_cl_test.json
