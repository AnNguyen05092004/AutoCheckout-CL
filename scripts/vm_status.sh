#!/usr/bin/env bash
# Status of the VM and of the runs, from the Mac (read-only: never touches the jobs).
#
#   bash scripts/vm_status.sh           # one-off summary
#   bash scripts/vm_status.sh follow    # live output of the running experiment; Ctrl+C only stops watching
#
# The pilot chain writes each experiment's output to /data/runs/<name>.log, not to its tmux window,
# so `tmux attach` shows an empty screen: follow the log instead.
set -uo pipefail
GC=(--zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0)
status=$(gcloud compute instances describe auto-cl "${GC[@]}" --format="value(status)")
echo "VM auto-cl: $status"
[[ $status == RUNNING ]] || exit 0
if [[ ${1:-} == follow ]]; then
    exec gcloud compute ssh auto-cl "${GC[@]}" -- -t \
        'latest=$(ls -t /data/runs/*.log | grep -v -e pilot_chain -e ppg_audit -e /queue.log | head -1); echo "== $latest"; tail -n 5 -F "$latest"'
fi
gcloud compute ssh auto-cl "${GC[@]}" --command '
    for f in /data/runs/queue.log /data/runs/pilot_chain.log; do
        [ -f $f ] && { echo "== $(basename $f) (one line per finished experiment)"; cat $f; }
    done
    latest=$(ls -t /data/runs/*.log 2>/dev/null | grep -v -e pilot_chain -e ppg_audit -e /queue.log | head -1)
    if [ -n "$latest" ]; then
        echo "== current: $(basename "$latest" .log)"
        grep -a -E "^-- task|^== " "$latest" | tail -2
        tail -c 400 "$latest" | tr "\r" "\n" | grep -a -v "^$" | tail -1 | cut -c1-170
    fi
    echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    echo "== disk"; df -h / | tail -1'
