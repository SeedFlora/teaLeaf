#!/usr/bin/env bash
# Score every completed run on its frozen partitions, then run the analyses that depend
# on those predictions. Safe to re-run: per-run evaluation is skipped when metrics.json
# already exists, so this can be invoked while the benchmark is still finishing.
#
# Additional deployment-candidate stages: paired export analysis, the activation-map
# sanity study, figures, and a frozen machine-readable results file.
set -uo pipefail
REPO="${TEA_REPO:-/mnt/h/tea/tea-leaf-android-feia}"
source "$REPO/environment/activate_tf.sh"

CACHE="$HOME/tea_ws/cache"
RUNS="$HOME/tea_ws/runs"
EVAL="$HOME/tea_ws/eval"
IMAGES="$HOME/tea_ws/data/raw/v3_extract/teaLeafBD/teaLeafBD"
mkdir -p "$EVAL"

QUIET='grep -vE "oneDNN|absl::|StatusFromEnv|external/local|computation placer|WARNING: All log|cuda_|Interpreter is deprecated|warnings.warn|tf_tfl_flatbuffer|Ignored output_format|Ignored drop_control|FutureWarning|binary_"'

echo "================= per-run evaluation ================="
for d in "$RUNS"/*/; do
  name=$(basename "$d")
  [ -f "$d/train_config.json" ] || { echo "[skip] $name (unfinished)"; continue; }
  [ -f "$EVAL/$name/metrics.json" ] && { echo "[skip] $name (already scored)"; continue; }

  split=$(python -c "import json;print(json.load(open('$d/train_config.json'))['split_prefix'])")
  echo ""
  echo "--- $name (split=$split) ---"
  python "$REPO/src/evaluation/evaluate.py" \
    --checkpoint "$d/best.keras" \
    --cache-dir "$CACHE" \
    --split-prefix "$split" \
    --partitions validation,test \
    --outdir "$EVAL/$name" 2>&1 | eval $QUIET
done

echo ""
echo "================= cross-model comparison (three seeds, seed-matched McNemar) ================="
python "$REPO/src/evaluation/compare_models.py" \
  --runs-dir "$RUNS" --eval-dir "$EVAL" \
  --outdir "$REPO/tables" --partition test 2>&1 | eval $QUIET

# The deployment candidate is fixed: MobileNetV3-Large at seed 42. It was selected before
# the seed replication and is the graph that was exported, benchmarked on the handset and
# shipped in the application; every downstream artifact refers to it.
BEST=mobilenetv3_large__primary__seed42
[ -f "$EVAL/$BEST/metrics.json" ] || { echo "deployment candidate $BEST not scored; stopping"; exit 1; }

echo ""
echo "================= deployment candidate: $BEST ================="

if [ ! -f "$REPO/tables/calibration/calibration_params.json" ]; then
  echo ""
  echo "--- calibration and abstention (fitted on validation only) ---"
  python "$REPO/src/calibration/calibrate.py" \
    --eval-dir "$EVAL/$BEST" --split-prefix primary \
    --outdir "$REPO/tables/calibration" 2>&1 | eval $QUIET
else
  echo "[skip] calibration already fitted (tables/calibration/calibration_params.json)"
fi

if [ ! -f "$REPO/models/exported/export_report.json" ]; then
  echo ""
  echo "--- FP32/FP16/INT8 export and parity ---"
  python "$REPO/src/export/export_tflite.py" \
    --checkpoint "$RUNS/$BEST/best.keras" --cache-dir "$CACHE" \
    --split-prefix primary \
    --outdir "$REPO/models/exported" 2>&1 | eval $QUIET
else
  echo "[skip] export already done (models/exported/export_report.json); the shipped files are not regenerated"
fi

echo ""
echo "--- robustness under capture degradation, with the abstention audit ---"
python "$REPO/src/evaluation/robustness.py" \
  --checkpoint "$RUNS/$BEST/best.keras" --cache-dir "$CACHE" \
  --split-prefix primary --partition test \
  --calibration "$REPO/tables/calibration/calibration_params.json" \
  --reference-predictions "$EVAL/$BEST/predictions_primary_test.npz" \
  --outdir "$REPO/tables/robustness" 2>&1 | eval $QUIET

echo ""
echo "--- paired comparison of every exported variant with the reference ---"
python "$REPO/src/export/parity_analysis.py" \
  --exported-dir "$REPO/models/exported" --cache-dir "$CACHE" \
  --eval-dir "$EVAL/$BEST" --outdir "$REPO/tables/export" 2>&1 | eval $QUIET

echo ""
echo "--- activation-map sanity analysis ---"
python "$REPO/src/evaluation/cam_analysis.py" \
  --checkpoint "$RUNS/$BEST/best.keras" --cache-dir "$CACHE" \
  --eval-dir "$EVAL/$BEST" --calibration "$REPO/tables/calibration/calibration_params.json" \
  --tables-outdir "$REPO/tables/cam" --figures-outdir "$REPO/figures/cam" 2>&1 | eval $QUIET

echo ""
echo "--- figures ---"
python "$REPO/src/figures/dataset_figure.py" --image-root "$IMAGES" \
  --splits-dir "$REPO/data/splits" --audit-dir "$REPO/data/audit" --outdir "$REPO/figures/dataset" > /dev/null
python "$REPO/src/figures/confusion_figure.py" --parity "$REPO/tables/export/parity_analysis.json" \
  --outdir "$REPO/figures/confusion"
python "$REPO/src/figures/robustness_figure.py" --robustness "$REPO/tables/robustness/robustness.json" \
  --outdir "$REPO/figures/robustness"

echo ""
echo "--- frozen machine-readable results ---"
python "$REPO/src/evaluation/freeze_results.py"

echo ""
echo "================= COMPLETE ================="
echo "deployment candidate: $BEST"
