# P2 (pilot): tasks 100+25 on train_pilot (3,000 images), 4 epochs, with every fix F1-F12.
source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/pilot_100-4x25_seed0
EXP=P2
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 4)
