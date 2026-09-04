#!/usr/bin/env bash
# Build the pixel caches on the WSL ext4 filesystem. Writing ~2 GB of .npy to the
# /mnt/d drvfs mount would be slow and would be re-read every epoch; the repo keeps
# only the metadata and checksums.
set -euo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

REPO="/mnt/d/tea/tea-leaf-android-feia"
ROOT="$HOME/tea_ws/data/raw/v3_extract/teaLeafBD/teaLeafBD"
CACHE="$HOME/tea_ws/cache"

python "$REPO/src/training/build_cache.py" \
  --splits-dir "$REPO/data/splits" \
  --image-root "$ROOT" \
  --outdir "$CACHE" \
  --eval-size 224 --train-size 256 \
  --prefixes primary,naive

cp "$CACHE/cache_metadata.json" "$REPO/data/manifests/cache_metadata.json"
echo "cache metadata copied into the repository"
du -sh "$CACHE"
