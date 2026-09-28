#!/usr/bin/env bash
# Send the committed code to the VM as a git bundle (the repo is not on GitHub, decision QD-7).
#
#   bash scripts/sync_to_vm.sh        # run on the Mac, from anywhere
#
# The VM copy (~/AutoCheckout-CL) is set to exactly the Mac's `main`: edits made on the VM to tracked
# files are overwritten; untracked files (data configs generated there, runs) are kept. Uncommitted
# changes on the Mac are not sent.
set -euo pipefail
VM_NAME=${VM_NAME:-auto-cl}
ZONE=${ZONE:-us-central1-c}
PROJECT=${PROJECT:-project-95a0d104-9d0f-4aa1-ba0}
REPO=$(cd "$(dirname "$0")/.." && pwd)

if [[ -n $(git -C "$REPO" status --porcelain --untracked-files=no) ]]; then
    echo "warning: uncommitted changes on the Mac are not synced" >&2
fi
BUNDLE=$(mktemp -d)/autocheckout.bundle
git -C "$REPO" bundle create "$BUNDLE" main
gcloud compute scp "$BUNDLE" "$VM_NAME:autocheckout.bundle" --zone="$ZONE" --project="$PROJECT"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command '
    set -e
    if [ -d ~/AutoCheckout-CL/.git ]; then
        git -C ~/AutoCheckout-CL fetch -q ~/autocheckout.bundle main
        git -C ~/AutoCheckout-CL reset -q --hard FETCH_HEAD
    else
        git clone -q -b main ~/autocheckout.bundle ~/AutoCheckout-CL
    fi
    git -C ~/AutoCheckout-CL log --oneline -1'
rm -f "$BUNDLE"
