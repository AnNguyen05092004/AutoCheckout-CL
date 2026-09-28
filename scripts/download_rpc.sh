#!/usr/bin/env bash
# Download RPC from Kaggle on the VM and unzip only the checkout images (T0.5).
#
#   bash ~/AutoCheckout-CL/scripts/download_rpc.sh
#
# Credentials, never committed: a Kaggle API token (KGAT_..., new format) in ~/.kaggle/access_token or
# in $KAGGLE_API_TOKEN; or a legacy ~/.kaggle/kaggle.json. The Kaggle CLI available for Python 3.10
# (1.7.4.5) does not read new-format tokens, so the download API is called with curl. Interrupted
# downloads resume. The HuggingFace mirrors are not usable: they lack the original file names and the
# `level` field (plan section 4.1).
set -euo pipefail
DATA=${DATA:-/data/rpc}
RAW=$DATA/raw
URL=https://www.kaggle.com/api/v1/datasets/download/diyer22/retail-product-checkout-dataset
ZIP=retail-product-checkout-dataset.zip

if [[ -n ${KAGGLE_API_TOKEN:-} ]]; then
    AUTH=(-H "Authorization: Bearer $KAGGLE_API_TOKEN")
elif [[ -f ~/.kaggle/access_token ]]; then
    AUTH=(-H "Authorization: Bearer $(cat ~/.kaggle/access_token)")
elif [[ -f ~/.kaggle/kaggle.json ]]; then
    AUTH=(-u "$(python3 -c 'import json, os; d = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"))); print(d["username"] + ":" + d["key"])')")
else
    echo "no Kaggle credentials: put the API token in ~/.kaggle/access_token (see the guide, 6.1)" >&2
    exit 1
fi

mkdir -p "$RAW"
cd "$RAW"
if [[ ! -d retail_product_checkout/test2019 ]]; then
    if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
        curl -fL --retry 5 -C - "${AUTH[@]}" -o "$ZIP" "$URL"
        unzip -tq "$ZIP"
    fi
    unzip -q -o "$ZIP" \
        'retail_product_checkout/val2019/*' 'retail_product_checkout/test2019/*' \
        'retail_product_checkout/instances_val2019.json' 'retail_product_checkout/instances_test2019.json'
fi
n_val=$(ls retail_product_checkout/val2019 | wc -l)
n_test=$(ls retail_product_checkout/test2019 | wc -l)
echo "val2019:  $n_val images (expected 6000)"
echo "test2019: $n_test images (expected 24000)"
if [[ $n_val -eq 6000 && $n_test -eq 24000 ]]; then
    rm -f "$ZIP"  # 25.3 GB; download again if the single-product images are needed
fi
df -h /
