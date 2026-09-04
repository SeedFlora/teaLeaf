#!/usr/bin/env bash
# Extract the locally supplied teaLeafBD archive into the WSL ext4 working area.
# The source archive on /mnt/d is treated as immutable and is never modified.
set -euo pipefail

SRC_ZIP="${1:-/mnt/d/tea/teaLeafBD.zip}"
DEST="${2:-$HOME/tea_ws/data/raw/v3_extract}"

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
mkdir -p "$DEST"

python3 - "$SRC_ZIP" "$DEST" <<'PY'
import sys, zipfile, time
src, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
with zipfile.ZipFile(src) as z:
    names = z.namelist()
    z.extractall(dest)
print(f"entries={len(names)} elapsed_s={time.time() - t0:.1f}")
PY

echo "--- directory tree (depth 3) ---"
find "$DEST" -maxdepth 3 -type d | sort

echo "--- per-directory file counts ---"
find "$DEST" -mindepth 1 -type d -print0 | while IFS= read -r -d '' d; do
  n=$(find "$d" -maxdepth 1 -type f | wc -l)
  [ "$n" -gt 0 ] && echo "$n	$d"
done

echo "--- total files ---"
find "$DEST" -type f | wc -l
