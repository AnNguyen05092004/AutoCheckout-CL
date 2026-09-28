# E3: PDP with all fixes F1-F12, as in the paper (no improvements I1-I5, no augmentation).
source "$REPO/configs/exp/common.sh"
EXP=E3
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}")
