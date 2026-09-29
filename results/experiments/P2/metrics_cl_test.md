# Class-incremental metrics: /data/runs/P2 (test)

| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0343 | 0.0286 | - | - | 0.0343 | 0.0286 | 0.0562 | 0.0469 |
| 2 | 0.0095 | 0.0074 | 0.0327 | 0.0273 | 0.0268 | 0.0223 | 0.0372 | 0.0309 |

## Accuracy matrix (AP50, task group as column, after stage as row; last = 2)
| stage \ group | 1 | 2 |
|---|---|---|
| 1 | 0.0343 | - |
| 2 | 0.0327 | 0.0095 |

Forgetting (average over groups < 2): 0.0016
Average of last row: 0.0211

Wrote /data/runs/P2/metrics_cl_test.json
