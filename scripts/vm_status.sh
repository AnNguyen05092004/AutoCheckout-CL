#!/usr/bin/env bash
# Status of the VM and of the runs, from the Mac (read-only: never touches the jobs).
#
#   bash scripts/vm_status.sh
set -uo pipefail
GC=(--zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0)
status=$(gcloud compute instances describe auto-cl "${GC[@]}" --format="value(status)")
echo "VM auto-cl: $status"
[[ $status == RUNNING ]] || exit 0
gcloud compute ssh auto-cl "${GC[@]}" --command '
    echo "== chain (one line per finished experiment)"; cat /data/runs/pilot_chain.log 2>/dev/null || echo "(none)"
    latest=$(ls -t /data/runs/*.log 2>/dev/null | grep -v -e pilot_chain -e ppg_audit | head -1)
    if [ -n "$latest" ]; then
        echo "== current: $(basename "$latest" .log)"
        grep -E "^-- task|^== " "$latest" | tail -2
        tail -c 400 "$latest" | tr "\r" "\n" | grep -v "^$" | tail -1 | cut -c1-170
    fi
    echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    echo "== disk"; df -h / | tail -1'
