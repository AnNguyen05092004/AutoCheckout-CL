# E3_coco (reproduction of the paper): PDP with all fixes F1-F13 on the frozen COCO checkpoint, as in the
# paper; no improvement (I1-I5) and no augmentation. Shows why the FSA start is needed on RPC.
source "$REPO/configs/exp/common.sh"
EXP=E3_coco
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}")
