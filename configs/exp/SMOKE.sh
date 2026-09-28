# SMOKE (R4, GPU, real images): pilot tasks 1-2 on 100 images per file (tools/subset_tasks.py), one epoch.
# Checks the whole path on real data before the pilot: training with teacher/PPG, prototypes, resume
# checkpoints, predictions and both evaluation tools.
source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/smoke
EXP=SMOKE
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 1)
