#!/usr/bin/env bash
# Build the forensic manifest for the locally supplied teaLeafBD release.
set -euo pipefail

VENV="$HOME/tea_ws/venv"
REPO="/mnt/d/tea/tea-leaf-android-feia"
ROOT="$HOME/tea_ws/data/raw/v3_extract/teaLeafBD/teaLeafBD"

source "$VENV/bin/activate"

echo "=== environment ==="
python - <<'PY'
import importlib
for m in ["tensorflow", "numpy", "PIL", "imagehash", "cv2", "sklearn", "scipy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'n/a')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {m}: IMPORT_FAILED {exc}")
PY

echo "=== building manifest ==="
python "$REPO/src/audit/build_manifest.py" \
  --root "$ROOT" \
  --version "v3_local_archive" \
  --out "$REPO/data/manifests/v3_manifest.csv" \
  --workers 16
