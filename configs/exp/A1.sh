# A1 (ablation of E4, 3 tasks): PPG replaced by a fixed threshold (tau_h only, paper Table 5 row 1).
source "$REPO/configs/exp/common.sh"
EXP=A1
N_TASKS=3
REUSE_TASK1=E4
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --pseudo threshold)
