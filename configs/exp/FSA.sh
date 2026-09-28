# FSA (B1b, improvement I1): full fine-tuning on task 1 (capped), exported for PDP task 1 (--repo_name).
source "$REPO/configs/exp/common.sh"
EXP=FSA
N_TASKS=1
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" "${STANDARD_ARGS[@]}" --save_hf 1)
