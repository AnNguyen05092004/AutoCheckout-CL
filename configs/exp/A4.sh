# A4 (ablation of E4, 3 tasks): top-5 teacher queries instead of 50.
source "$REPO/configs/exp/common.sh"
EXP=A4
N_TASKS=3
REUSE_TASK1=E4
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --pseudo_topk 5)
