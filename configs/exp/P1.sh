# P1 (pilot): as P2 but with the original behaviour of every fix except those needed to run
# (F1 task config, F2 pool size, F8 paths/epochs). The kernel (F11) only changes speed and the
# evaluation (F9) is shared by all experiments, so both stay on.
source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/pilot_100-4x25_seed0
EXP=P1
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 4
    --init_new_prompts 0 --ddl_lambda 0 --query_loss_grad 0 --teacher_prompts 0 --ppg_legacy 1
    --bg_thres_topk 5 --proto_correct_only 0 --shuffle 0)
