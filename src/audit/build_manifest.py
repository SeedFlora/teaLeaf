"""Build the forensic per-image manifest for a teaLeafBD release.

For every file under the dataset root this records identity (SHA-256), container
facts (size, extension, detected format), decoded image properties (dimensions,
mode, corruption status), capture metadata (EXIF presence, camera make/model,
orientation), perceptual hashes used later for near-duplicate clustering, and
simple quality statistics (Laplacian-variance blur proxy, brightness, contrast).

Nothing is deleted or modified: files that fail to decode are flagged in place so
that a later quarantine step can act on them explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ExifTags

warnings.filterwarnings("ignore", category=UserWarning)
Image.MAX_IMAGE_PIXELS = None
# Report truncated files rather than silently padding them.
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}

FIELDS = [
    "dataset_version", "relative_path", "class_folder_raw", "normalized_class", "filename",
    "extension", "detected_format", "mime_type", "sha256", "size_bytes",
    "width", "height", "aspect_ratio", "megapixels", "color_mode",
    "decode_ok", "corruption_status", "truncated",
    "has_exif", "exif_make", "exif_model", "exif_orientation", "exif_datetime", "exif_software",
    "icc_profile", "jpeg_quality_estimate", "is_progressive",
    "mean_brightness", "std_contrast", "laplacian_var_blur", "sharpness_flag",
    "phash16", "dhash16", "ahash8", "whash8", "colorhash",
]

# Raw dataset folder names are preserved verbatim in `class_folder_raw`; this map
# only provides a separate normalized key. Raw labels are never overwritten.
NORMALIZED_CLASS = {
    "1. Tea algal leaf spot": "tea_algal_leaf_spot",
    "2. Brown Blight": "brown_blight",
    "3. Gray Blight": "gray_blight",
    "4. Helopeltis": "helopeltis_damage",
    "5. Red spider": "red_spider_mite_damage",
    "6. Green mirid bug": "green_mirid_bug_damage",
    "7. Healthy leaf": "healthy_leaf",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def process(args: tuple[str, str, str]) -> dict:
    path_s, root_s, version = args
    path, root = Path(path_s), Path(root_s)
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    class_folder = parts[0] if len(parts) > 1 else ""

    row = {k: "" for k in FIELDS}
    row.update(
        dataset_version=version,
        relative_path=rel,
        class_folder_raw=class_folder,
        normalized_class=NORMALIZED_CLASS.get(class_folder, ""),
        filename=path.name,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        decode_ok="False",
        corruption_status="unknown",
        truncated="False",
    )

    try:
        row["sha256"] = sha256_of(path)
    except Exception as exc:  # noqa: BLE001
        row["corruption_status"] = f"hash_failed:{type(exc).__name__}"
        return row

    raw = path.read_bytes()

    # Pass 1: verify() detects structural corruption without full decoding.
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        verify_ok = True
    except Exception as exc:  # noqa: BLE001
        verify_ok = False
        row["corruption_status"] = f"verify_failed:{type(exc).__name__}"

    # Pass 2: full decode plus metadata extraction.
    try:
        with Image.open(io.BytesIO(raw)) as im:
            row["detected_format"] = im.format or ""
            row["mime_type"] = Image.MIME.get(im.format or "", "")
            row["color_mode"] = im.mode
            row["is_progressive"] = str(bool(im.info.get("progressive", False)))
            row["icc_profile"] = str(bool(im.info.get("icc_profile")))
            q = im.info.get("quality")
            row["jpeg_quality_estimate"] = str(q) if q else ""

            exif = None
            try:
                exif = im.getexif()
            except Exception:  # noqa: BLE001
                exif = None
            if exif and len(exif) > 0:
                row["has_exif"] = "True"
                row["exif_make"] = str(exif.get(EXIF_TAGS.get("Make", -1), "")).strip("\x00 ")
                row["exif_model"] = str(exif.get(EXIF_TAGS.get("Model", -1), "")).strip("\x00 ")
                row["exif_orientation"] = str(exif.get(EXIF_TAGS.get("Orientation", -1), ""))
                row["exif_datetime"] = str(exif.get(EXIF_TAGS.get("DateTime", -1), ""))
                row["exif_software"] = str(exif.get(EXIF_TAGS.get("Software", -1), ""))
            else:
                row["has_exif"] = "False"

            rgb = im.convert("RGB")
            w, h = rgb.size
            row["width"], row["height"] = w, h
            row["aspect_ratio"] = round(w / h, 6) if h else ""
            row["megapixels"] = round(w * h / 1e6, 4)

            # Quality statistics on a fixed-size grayscale thumbnail so that the
            # blur proxy is comparable across differing source resolutions.
            small = rgb.resize((256, 256), Image.BILINEAR)
            g = np.asarray(small.convert("L"), dtype=np.float32)
            row["mean_brightness"] = round(float(g.mean()), 3)
            row["std_contrast"] = round(float(g.std()), 3)
            k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            c = sum(
                g[1 + dy: g.shape[0] - 1 + dy, 1 + dx: g.shape[1] - 1 + dx] * k[dy + 1, dx + 1]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if k[dy + 1, dx + 1] != 0
            )
            lv = float(np.var(c))
            row["laplacian_var_blur"] = round(lv, 3)
            row["sharpness_flag"] = "blurry" if lv < 100 else ("soft" if lv < 300 else "sharp")

            import imagehash
            row["phash16"] = str(imagehash.phash(rgb, hash_size=16))
            row["dhash16"] = str(imagehash.dhash(rgb, hash_size=16))
            row["ahash8"] = str(imagehash.average_hash(rgb, hash_size=8))
            row["whash8"] = str(imagehash.whash(rgb, hash_size=8))
            row["colorhash"] = str(imagehash.colorhash(rgb))

        row["decode_ok"] = "True"
        row["corruption_status"] = "ok" if verify_ok else row["corruption_status"] + "|decoded_anyway"
    except Exception as exc:  # noqa: BLE001
        row["decode_ok"] = "False"
        prev = row["corruption_status"]
        note = f"decode_failed:{type(exc).__name__}:{str(exc)[:80]}"
        row["corruption_status"] = note if prev in ("unknown", "") else f"{prev}|{note}"

    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    root = Path(a.root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"scanning {len(files)} files under {root} with {a.workers} workers", flush=True)

    tasks = [(str(p), str(root), a.version) for p in files]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, row in enumerate(ex.map(process, tasks, chunksize=16), 1):
            rows.append(row)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    rows.sort(key=lambda r: r["relative_path"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["decode_ok"] == "True")
    print(f"\nwrote {len(rows)} rows -> {out}")
    print(f"decode_ok={ok}  decode_failed={len(rows) - ok}")
    print(f"unique_sha256={len({r['sha256'] for r in rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
