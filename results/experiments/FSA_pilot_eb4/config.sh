# FSA_pilot without gradient accumulation (effective batch 4): diagnostic, see P2_eb4.sh. FSA_pilot (28/09)
# reached a val mAP50 of only 0.085 after 364 optimiser steps, with the loss still falling.
source "$REPO/configs/exp/FSA_pilot.sh"
EXP=FSA_pilot_eb4
ARGS+=(--eff_batch_size 4)
