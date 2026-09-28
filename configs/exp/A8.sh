# A8 (ablation of E4, 3 tasks): shared parameters frozen after task 1 (I5).
source "$REPO/configs/exp/common.sh"
EXP=A8
N_TASKS=3
REUSE_TASK1=E4
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --freeze_shared_after_task1 1)
