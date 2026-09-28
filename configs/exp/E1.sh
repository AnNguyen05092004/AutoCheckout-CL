# E1 (lower bound): sequential training with one shared prompt pool, no private pool, no pseudo-labels.
source "$REPO/configs/exp/common.sh"
EXP=E1
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" --use_private 0 --pseudo none)
