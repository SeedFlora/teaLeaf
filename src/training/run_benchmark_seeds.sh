#!/usr/bin/env bash
# Phase D: seed replication for the three architectures that had a single run.
#
# Reviewers of the FEIA submission asked that every architecture be compared on
# mean and variability over the same seeds rather than on seed 42 alone, because
# the observed seed-to-seed spread (3.06 macro-F1 points for MobileNetV3-Large)
# exceeds several inter-model gaps. Phase B already replicated MobileNetV3-Large
# and EfficientNet-B0 at seeds 123 and 2026; this phase completes the grid so
# all five backbones have exactly three seeds (42, 123, 2026) under identical
# hyperparameters, split and early-stopping protocol.
#
# Completed runs (train_config.json present) are skipped, so this is idempotent.
set -uo pipefail
REPO="${TEA_REPO:-/mnt/h/tea/tea-leaf-android-feia}"
source "$REPO/environment/activate_tf.sh"

CACHE="$HOME/tea_ws/cache"
RUNS="$HOME/tea_ws/runs"
mkdir -p "$RUNS"

# Identical to run_benchmark.sh -- the only variable across runs is the seed.
COMMON="--cache-dir $CACHE --warmup-epochs 5 --finetune-epochs 35 --patience 8 --batch-size 32 --mixed-precision"

run () {           # backbone split seed
  local bb="$1" sp="$2" sd="$3"
  local tag="${bb}__${sp}__seed${sd}"
  local out="$RUNS/$tag"
  if [ -f "$out/train_config.json" ]; then
    echo "[skip] $tag already complete"
    return 0
  fi
  echo ""
  echo "############ $tag ############"
  date -u +"start %Y-%m-%dT%H:%M:%SZ"
  python "$REPO/src/training/train.py" \
    --backbone "$bb" --split-prefix "$sp" --seed "$sd" \
    --outdir "$out" $COMMON 2>&1 \
    | grep -E "epoch |best val|wall clock|train \(|stage |early stopping|Error|Traceback" || true
  date -u +"end   %Y-%m-%dT%H:%M:%SZ"
}

echo "================= PHASE D: seed replication for single-run architectures ================="
for sd in 123 2026; do
  run mobilenetv3_small primary "$sd"
  run efficientnetv2b0  primary "$sd"
  run convnext_tiny     primary "$sd"
done

echo ""
echo "================= PHASE D COMPLETE ================="
ls -1 "$RUNS" | sed 's/^/  /'
