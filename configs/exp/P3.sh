# P3 (pilot): P2 + FSA. Needs FSA_pilot first.
source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/pilot_100-4x25_seed0
EXP=P3
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 4 --repo_name "$RUNS/FSA_pilot/task_1/hf_model")
# Run on 28/09 with the effective batch of the original code and without TF32 (common.sh now uses 4 and TF32).
ARGS+=(--eff_batch_size 32 --tf32 0)
