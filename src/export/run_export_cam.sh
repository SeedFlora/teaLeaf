#!/usr/bin/env bash
set -uo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"
RUN="${1:-mobilenetv3_small__primary__seed42}"

python "$REPO/src/export/export_with_cam.py" \
  --checkpoint "$HOME/tea_ws/runs/$RUN/best.keras" \
  --cache-dir "$HOME/tea_ws/cache" \
  --calibration "$REPO/tables/calibration/calibration_params.json" \
  --split-prefix primary \
  --outdir "$REPO/models/exported" \
  --assets-dir "$REPO/android/app/src/main/assets" \
  --variant fp16 \
  --model-version 1.0.0 2>&1 \
  | grep -vE 'oneDNN|absl::|StatusFromEnv|external/local|computation placer|WARNING: All log|cuda_|deprecated|warnings\.warn|tf_tfl|Ignored |^  %|^\}|func\.return|tf\.entry_function|^module|loc\(|TensorSpec|Captures:|Output Type|Saved artifact|^\* Endpoint'
