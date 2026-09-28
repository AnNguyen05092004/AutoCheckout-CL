# FSA for the pilot (P3): full fine-tuning on pilot task 1, exported for --repo_name.
source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/pilot_100-4x25_seed0
EXP=FSA_pilot
N_TASKS=1
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 4 --save_hf 1)
# Run on 28/09 with the effective batch of the original code (common.sh now uses 4).
ARGS+=(--eff_batch_size 32)
