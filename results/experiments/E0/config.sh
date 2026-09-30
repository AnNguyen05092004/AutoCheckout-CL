# E0 (upper bound, B1a): full fine-tuning on the 200 classes at once, 12 epochs, on the union of the
# capped task images with every label; written as task_5 so it is evaluated like the last stage.
source "$REPO/configs/exp/common.sh"
EXP=E0
START_TASK=5
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" --joint 1 --train_suffix _capped --epochs 12)
