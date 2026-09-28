#!/usr/bin/env bash
# Run the experiments listed in $RUNS/queue.txt one after the other, then stop the VM (also after an error).
#
#   echo E3 >> /data/runs/queue.txt        # one config name per line (configs/exp/<name>.sh); '#' starts a comment
#   tmux new -d -s queue 'bash ~/AutoCheckout-CL/scripts/run_queue.sh'
#
# The queue file is read again before each experiment, so names can be appended while it runs.
# A finished experiment gets the line "<name> exit=<status> <date>" in $RUNS/queue.log and is never run
# again. One without that line (the VM stopped midway, e.g. Spot preemption) is run again by the same
# command, and run_exp.sh continues it where it stopped. Output of each experiment: $RUNS/<name>.log.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-/data/runs}
CONFIG_DIR=${CONFIG_DIR:-$REPO/configs/exp}
QUEUE=$RUNS/queue.txt
LOG=$RUNS/queue.log
trap 'echo "queue finished $(date "+%F %T")" >> "$LOG"; ${SHUTDOWN_CMD:-sudo shutdown -h now}' EXIT
if [[ -f $HOME/venvs/pdp/bin/activate ]]; then
    source "$HOME/venvs/pdp/bin/activate"
fi
cd "$REPO"
touch "$LOG"

next_experiment() {
    local name
    [[ -f $QUEUE ]] || return 0
    while read -r name _; do
        [[ -z $name || $name == \#* ]] && continue
        grep -q "^$name exit=" "$LOG" || { echo "$name"; return 0; }
    done < "$QUEUE"
}

while exp=$(next_experiment) && [[ -n $exp ]]; do
    bash scripts/run_exp.sh "$CONFIG_DIR/$exp.sh" >> "$RUNS/$exp.log" 2>&1
    echo "$exp exit=$? $(date "+%F %T")" >> "$LOG"
done
