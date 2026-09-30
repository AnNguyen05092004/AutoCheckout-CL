# E4_noF14: E4 as run on 29/09, before F14 (duplicate pseudo-labels were kept). Val mAP@A AP50 0.938 after task 5,
# test cAcc 0.100; kept as the comparison for F14. Its run directory was /data/runs/E4 until 30/09.
source "$REPO/configs/exp/E4.sh"
EXP=E4_noF14
ARGS+=(--pseudo_dedup_iou 0)
