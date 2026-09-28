# A7 (ablation of E4, 3 tasks): nearest-prototype check on (I4).
source "$REPO/configs/exp/common.sh"
EXP=A7
N_TASKS=3
REUSE_TASK1=E4
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --prototype_nearest 1)
