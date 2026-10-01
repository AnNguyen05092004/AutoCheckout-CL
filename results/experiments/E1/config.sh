# E1 (lower bound): sequential training with one shared prompt pool, no private pool, no pseudo-labels;
# on the FSA base like E2-E4.
source "$REPO/configs/exp/common.sh"
EXP=E1
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${FSA_ARGS[@]}" --use_private 0 --pseudo none)
