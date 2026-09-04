#!/usr/bin/env bash
set -euo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"
ROOT="$HOME/tea_ws/data/raw/v3_extract/teaLeafBD/teaLeafBD"

python "$REPO/src/audit/duplicate_audit.py" \
  --manifest "$REPO/data/manifests/v3_manifest.csv" \
  --image-root "$ROOT" \
  --outdir "$REPO/data/audit" \
  --phash-candidate-threshold 32 \
  --phash-screen-threshold 64 \
  --dhash-confirm-threshold 48 \
  --ssim-confirm-threshold 0.90
