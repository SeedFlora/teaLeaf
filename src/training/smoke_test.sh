#!/usr/bin/env bash
# Short run to prove the training path works end to end before committing hours of
# compute: data loads, augmentation compiles on GPU, macro-F1 checkpointing fires,
# and the config is written.
set -euo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"

python "$REPO/src/training/train.py" \
  --backbone mobilenetv3_large \
  --cache-dir "$HOME/tea_ws/cache" \
  --split-prefix primary \
  --outdir "$HOME/tea_ws/runs/smoke_mobilenetv3_large" \
  --seed 42 \
  --warmup-epochs 1 \
  --finetune-epochs 2 \
  --patience 5 \
  --mixed-precision
