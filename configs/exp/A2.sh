# A2 (ablation of E4, 3 tasks): no shared prompt pool (L_DDL off as well).
source "$REPO/configs/exp/common.sh"
EXP=A2
N_TASKS=3
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --use_shared 0)
