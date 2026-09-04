#!/usr/bin/env bash
# Full G4 benchmark.
#
# Budget shape, stated openly: a complete 3-seed sweep of all five backbones was not
# affordable, so the brief's documented fallback is used -- one complete main run for
# every architecture, additional seeds for the leading candidates, and bootstrap
# confidence intervals computed later on the fixed test predictions.
#
# Phase A  five architectures, seed 42            -> accuracy/efficiency Pareto
# Phase B  two leading architectures, seeds 123 and 2026 -> seed variability
# Phase C  one architecture on the naive split, three seeds -> leakage sensitivity (RQ1)
#
# Phase C deliberately reuses the same architecture, hyperparameters and seeds as its
# primary-split counterpart, so the only variable between them is the split protocol.
set -uo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"
CACHE="$HOME/tea_ws/cache"
RUNS="$HOME/tea_ws/runs"
mkdir -p "$RUNS"

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

echo "================= PHASE A: architecture sweep (seed 42) ================="
for bb in mobilenetv3_small mobilenetv3_large efficientnetb0 efficientnetv2b0 convnext_tiny; do
  run "$bb" primary 42
done

echo ""
echo "================= PHASE B: seed replication ================="
for sd in 123 2026; do
  run mobilenetv3_large primary "$sd"
  run efficientnetb0    primary "$sd"
done

echo ""
echo "================= PHASE C: leakage sensitivity (naive split) ================="
for sd in 42 123 2026; do
  run mobilenetv3_large naive "$sd"
done

echo ""
echo "================= COMPLETE ================="
ls -1 "$RUNS" | sed 's/^/  /'
