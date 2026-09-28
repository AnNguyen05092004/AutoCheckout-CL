#!/usr/bin/env bash
# Pilot (plan section 7): P2, P1, FSA_pilot, P3, then the pseudo-label audit V4 of P2's task 2; the VM is
# stopped at the end, also after an error. Results in /data/runs/<name>/ and /data/runs/<name>.log.
#
#   tmux new -s pilot 'bash ~/AutoCheckout-CL/scripts/run_pilot.sh'
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-/data/runs}
DATA=${DATA:-/data/rpc}
LOG=$RUNS/pilot_chain.log
trap 'echo "pilot chain finished $(date "+%F %T")" >> "$LOG"; ${SHUTDOWN_CMD:-sudo shutdown -h now}' EXIT
source "$HOME/venvs/pdp/bin/activate"
cd "$REPO"
for exp in P2 P1 FSA_pilot P3; do
    bash scripts/run_exp.sh "configs/exp/$exp.sh" > "$RUNS/$exp.log" 2>&1
    echo "$exp exit=$? $(date "+%F %T")" >> "$LOG"
done
# V4 on P2, task 2 (1,000 fully annotated training images of the pilot)
eval "ARGS=($(REPO=$REPO DATA=$DATA RUNS=$RUNS bash -c 'source configs/exp/P2.sh; printf "%q " "${ARGS[@]}"'))"
(cd pdp && python ppg_audit.py "${ARGS[@]}" --output_dir "$RUNS/P2" --audit_task 2 --audit_images 1000) \
    > "$RUNS/P2_ppg_audit.log" 2>&1
echo "V4 exit=$? $(date "+%F %T")" >> "$LOG"
