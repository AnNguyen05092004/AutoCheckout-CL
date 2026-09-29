# DET (B1c): class-agnostic detector for E5, trained on the task-1 images with every box (option b,
# decided 28/09); predictions on the shared val/test files, read by baselines/retrieval.py.
source "$REPO/configs/exp/common.sh"
EXP=DET
SKIP_EVAL=1
N_TASKS=1
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" "${STANDARD_ARGS[@]}"
    --task_config "$REPO/configs/tasks_agnostic.json" --n_classes 2
    --task_ann_dir "$DATA/tasks/agnostic_task1" --pred_ann_dir "$TASK_DIR")
# Run on 28-29/09 without TF32 (common.sh now enables it).
ARGS+=(--tf32 0)
