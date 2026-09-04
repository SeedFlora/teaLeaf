#!/usr/bin/env bash
# Render the cross-class pairs that survive pixel confirmation only at the loose
# screening distance, so a human can judge whether they are genuine label conflicts
# (the same leaf filed under two classes) or coincidental look-alikes.
set -euo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"
ROOT="$HOME/tea_ws/data/raw/v3_extract/teaLeafBD/teaLeafBD"

python - "$REPO" "$ROOT" <<'PY'
import csv, sys
from pathlib import Path
repo, root = Path(sys.argv[1]), Path(sys.argv[2])

rows = [r for r in csv.DictReader((repo/"data/audit/near_duplicate_candidates.csv").open(encoding="utf-8"))
        if r["passes_pixel_confirmation"] == "True" and r["label_a"] != r["label_b"]]
rows.sort(key=lambda r: -float(r["ssim"]))
out = repo/"data/audit/cross_class_pairs.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"cross-class confirmed pairs: {len(rows)}")
print(f"  phash range: {min(int(r['phash_hamming']) for r in rows)} - {max(int(r['phash_hamming']) for r in rows)}")
print(f"  ssim  range: {min(float(r['ssim']) for r in rows):.4f} - {max(float(r['ssim']) for r in rows):.4f}")
n_tight = sum(1 for r in rows if int(r["phash_hamming"]) <= 32)
print(f"  within operative threshold 32: {n_tight}")
PY

python "$REPO/src/audit/contact_sheets.py" \
  --pairs "$REPO/data/audit/cross_class_pairs.csv" \
  --image-root "$ROOT" \
  --outdir "$REPO/figures/cross_class_review" \
  --max-phash 64 --per-sheet 5

