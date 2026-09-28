#!/usr/bin/env bash
# Download RPC from Kaggle on the VM and unzip only the checkout images (T0.5).
#
#   bash ~/AutoCheckout-CL/scripts/download_rpc.sh
#
# Needs ~/.kaggle/kaggle.json (copied from the Mac, never committed). The HuggingFace mirrors are not
# usable: they lack the original file names and the `level` field (plan section 4.1).
set -euo pipefail
DATA=${DATA:-/data/rpc}
RAW=$DATA/raw
mkdir -p "$RAW"
cd "$RAW"
if [[ ! -d retail_product_checkout/test2019 ]]; then
    [[ -f retail-product-checkout-dataset.zip ]] || \
        kaggle datasets download -d diyer22/retail-product-checkout-dataset -p "$RAW"
    unzip -q -o retail-product-checkout-dataset.zip \
        'retail_product_checkout/val2019/*' 'retail_product_checkout/test2019/*' \
        'retail_product_checkout/instances_val2019.json' 'retail_product_checkout/instances_test2019.json'
fi
echo "val2019:  $(ls retail_product_checkout/val2019 | wc -l) images (expected 6000)"
echo "test2019: $(ls retail_product_checkout/test2019 | wc -l) images (expected 24000)"
if [[ $(ls retail_product_checkout/val2019 | wc -l) -eq 6000 && $(ls retail_product_checkout/test2019 | wc -l) -eq 24000 ]]; then
    rm -f retail-product-checkout-dataset.zip  # 15.9 GB; download again if single-product images are needed
fi
df -h /
