#!/usr/bin/env bash
set -euo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"
ROOT="$HOME/tea_ws/data/raw/v3_extract/teaLeafBD/teaLeafBD"

echo "=== contact sheets for human review ==="
python "$REPO/src/audit/contact_sheets.py" \
  --pairs "$REPO/data/audit/near_duplicate_candidates.csv" \
  --image-root "$ROOT" \
  --outdir "$REPO/figures/contact_sheets" \
  --max-phash 32 --per-sheet 6

echo
echo "=== class distribution and bias audit ==="
python "$REPO/src/audit/bias_audit.py" \
  --manifest "$REPO/data/manifests/v3_manifest.csv" \
  --outdir "$REPO/data/audit"

echo
echo "=== leakage-aware split construction ==="
python "$REPO/src/data/build_splits.py" \
  --manifest "$REPO/data/manifests/v3_manifest.csv" \
  --groups "$REPO/data/audit/group_assignment.csv" \
  --clusters "$REPO/data/audit/duplicate_clusters.csv" \
  --outdir "$REPO/data/splits" \
  --seed 42
