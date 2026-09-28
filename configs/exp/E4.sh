# E4 (main method): E3 + FSA (I1) + augmentation (I2) + no pseudo-label on current-task boxes (I3).
source "$REPO/configs/exp/common.sh"
EXP=E4
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}")
