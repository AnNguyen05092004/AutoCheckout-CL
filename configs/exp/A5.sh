# A5 (ablation of E4, 3 tasks): no FSA: PDP task 1 starts from the COCO checkpoint.
source "$REPO/configs/exp/common.sh"
EXP=A5
N_TASKS=3
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --repo_name SenseTime/deformable-detr)
