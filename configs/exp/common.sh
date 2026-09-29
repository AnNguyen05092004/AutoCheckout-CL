# Settings shared by every experiment; sourced by configs/exp/<name>.sh (REPO and DATA come from
# scripts/run_exp.sh). IMPLEMENTATION_PLAN.md section 7 lists the experiments.
TASK_CFG=$REPO/configs/tasks_100-4x25_seed0.json
TASK_DIR=$DATA/tasks/100-4x25_seed0

# Model and optimisation as in the PDP code / paper. BATCH_SIZE 4: benchmark on the L4 (28/09), 0.336 s/image and
# 6.2 GB peak for a task >= 2 step (batch 2: 0.374 s/image, 4.5 GB).
# Effective batch 4, i.e. no gradient accumulation (the original code accumulates to 32). The pilot (28/09) at 32
# was far too short: 364 optimiser steps on 2,910 images left FSA_pilot at a val mAP50 of 0.085, against 0.675 at 4
# (FSA_pilot_eb4, same data and compute). A task of RPC has about 10x fewer images than one of COCO in the paper.
# TF32 matrix multiplications (--tf32 1): 0.302 s/image against 0.333 for a task >= 2 step on the L4 (29/09).
BATCH_SIZE=${BATCH_SIZE:-4}
COMMON_ARGS=(
    --task_config "$TASK_CFG" --n_classes 225
    --task_ann_dir "$TASK_DIR" --train_img_dir "$DATA/checkout_800" --test_img_dir "$DATA/checkout_800"
    --repo_name SenseTime/deformable-detr --accelerator gpu --n_gpus 1 --require_kernel 1
    --batch_size "$BATCH_SIZE" --eff_batch_size 4 --num_workers 4 --tf32 1
    --lr 1e-4 --lr_old 1e-5 --eval_epochs 100
    --use_prompts 1 --num_prompts 100 --prompt_len 10 --local_query 1 --lambda_query 0.1
    --freeze backbone,encoder,decoder --new_params class_embed,prompts
)
# Standard configuration (plan 3.4): at most 6,000 images per task, 6 epochs.
STANDARD_ARGS=(--train_suffix _capped --epochs 6)

# Full fine-tuning without prompts (B1): E0, FSA, the class-agnostic detector of E5.
# 'backbone' matches the Hugging Face parameter names (the original default 'backbone.0' matches none).
FINETUNE_ARGS=(--use_prompts 0 --local_query 0 --pseudo none --freeze "" --optim_groups detr
    --lr_backbone_names backbone)

# FSA start (I1): PDP on the Deformable DETR fine-tuned on task 1 (configs/exp/FSA.sh, must be finished first).
# E1-E4 all start from it (decided 29/09): on the frozen COCO checkpoint, PDP reached a pilot val mAP50 of 0.11
# against 0.83 on the FSA base, so a comparison of continual-learning methods is only meaningful on FSA.
# E3_coco keeps the paper's COCO start as the reproduction.
FSA_ARGS=(--repo_name "$RUNS/FSA/task_1/hf_model")

# PDP with the improvements of E4 (plan 6.6): FSA start (I1), augmentation (I2), no double labels (I3).
E4_ARGS=("${FSA_ARGS[@]}" --augment 1 --pseudo_gt_iou 0.5)

# Order of the runs: FSA -> E1-E4 -> ablations that reuse E4's task 1 (A1, A4, A7, A8); DET -> E5.
