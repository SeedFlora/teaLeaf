#!/usr/bin/env bash
# CPU-only analyses for the revision, safe to run while the GPU is training.
set -uo pipefail
REPO="${TEA_REPO:-/mnt/h/tea/tea-leaf-android-feia}"
source "$REPO/environment/activate_tf.sh"
export CUDA_VISIBLE_DEVICES=
export TF_CPP_MIN_LOG_LEVEL=2
CACHE="$HOME/tea_ws/cache"
RUNS="$HOME/tea_ws/runs"
EVAL="$HOME/tea_ws/eval"
BEST=mobilenetv3_large__primary__seed42

echo "================= paired export analysis ================="
python "$REPO/src/export/parity_analysis.py" \
  --exported-dir "$REPO/models/exported" --cache-dir "$CACHE" \
  --eval-dir "$EVAL/$BEST" --outdir "$REPO/tables/export"

echo ""
echo "================= robustness with abstention audit ================="
cp "$REPO/tables/robustness/robustness.json" "$HOME/tea_ws/eval/robustness_submitted.json"
python "$REPO/src/evaluation/robustness.py" \
  --checkpoint "$RUNS/$BEST/best.keras" --cache-dir "$CACHE" \
  --split-prefix primary --partition test \
  --calibration "$REPO/tables/calibration/calibration_params.json" \
  --outdir "$REPO/tables/robustness"
echo "================= CPU ANALYSES COMPLETE ================="
