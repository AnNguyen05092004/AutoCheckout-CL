# P2 without gradient accumulation (effective batch 4 instead of 32): 8x more optimiser steps for the same
# compute. Diagnostic after P2 (28/09): task-1 val mAP50 0.042, and the class of a well-localised box was
# right only 14% of the time, while the loss was still falling (364 optimiser steps in task 1).
source "$REPO/configs/exp/P2.sh"
EXP=P2_eb4
ARGS+=(--eff_batch_size 4)
