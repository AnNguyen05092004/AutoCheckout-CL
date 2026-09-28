# A3 (ablation of E4, 3 tasks): no private prompt pool (L_DDL off as well).
source "$REPO/configs/exp/common.sh"
EXP=A3
N_TASKS=3
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --use_private 0)
