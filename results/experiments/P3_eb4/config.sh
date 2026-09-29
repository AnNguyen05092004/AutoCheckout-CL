# P3 without gradient accumulation (effective batch 4), on FSA_pilot_eb4: diagnostic, see P2_eb4.sh.
source "$REPO/configs/exp/P3.sh"
EXP=P3_eb4
ARGS+=(--eff_batch_size 4 --repo_name "$RUNS/FSA_pilot_eb4/task_1/hf_model")
