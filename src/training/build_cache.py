"""Precompute fixed-size pixel caches so the GPU, not the filesystem, is the bottleneck.

Two caches are written per split, and the distinction is deliberate:

EVAL cache (224x224) is produced by resizing the ORIGINAL image directly to the
    network input size with bilinear interpolation. This is exactly the operation
    the Android app performs, so validation and test numbers are computed on the
    same pixels the phone will see. No crop is involved, which removes a whole
    family of desktop/mobile mismatch bugs.

TRAIN cache (256x256) is a slightly larger direct resize, giving random-resized-crop
    augmentation something to crop from without ever reading the source JPEG again.

Caching a resized copy is safe here because the audit established that every image
is EXIF orientation 1, so no rotation normalization is being silently skipped.

Only the training partition is ever augmented; the eval cache is used verbatim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_ORDER = [
    "1. Tea algal leaf spot",
    "2. Brown Blight",
    "3. Gray Blight",
    "4. Helopeltis",
    "5. Red spider",
    "6. Green mirid bug",
    "7. Healthy leaf",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def build(rows: list[dict], root: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), size, size, 3), dtype=np.uint8)
    y = np.zeros(len(rows), dtype=np.int32)
    for i, r in enumerate(rows):
        with Image.open(root / r["relative_path"]) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            x[i] = np.asarray(im, dtype=np.uint8)
        y[i] = CLASS_TO_IDX[r["class_folder_raw"]]
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(rows)}", flush=True)
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eval-size", type=int, default=224)
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--prefixes", default="primary,naive")
    a = ap.parse_args()

    splits = Path(a.splits_dir)
    root = Path(a.image_root)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "class_order": CLASS_ORDER,
        "eval_size": a.eval_size,
        "train_size": a.train_size,
        "eval_preprocessing": (
            f"PIL convert('RGB') then resize(({a.eval_size},{a.eval_size}), BILINEAR) applied to the "
            "original image. No crop. Pixel values stay uint8 [0,255]; the network's embedded "
            "preprocessing layer performs normalization, so the same raw pixels are fed on desktop "
            "and on Android."
        ),
        "train_preprocessing": (
            f"Direct resize to {a.train_size}x{a.train_size} as an augmentation source only; "
            "random resized crop to the eval size is applied on GPU at training time."
        ),
        "splits": {},
    }

    for prefix in a.prefixes.split(","):
        for part in ("train", "validation", "test"):
            p = splits / f"{prefix}_{part}.csv"
            if not p.exists():
                print(f"skip missing {p}")
                continue
            with p.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

            size = a.train_size if part == "train" else a.eval_size
            print(f"building {prefix}/{part}: {len(rows)} images at {size}px", flush=True)
            x, y = build(rows, root, size)

            xp = outdir / f"{prefix}_{part}_x.npy"
            yp = outdir / f"{prefix}_{part}_y.npy"
            np.save(xp, x)
            np.save(yp, y)

            counts = np.bincount(y, minlength=len(CLASS_ORDER)).tolist()
            meta["splits"][f"{prefix}_{part}"] = {
                "n": len(rows),
                "size": size,
                "x_file": xp.name,
                "y_file": yp.name,
                "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "class_counts": {CLASS_ORDER[i]: c for i, c in enumerate(counts)},
                "bytes": int(x.nbytes),
            }
            print(f"  -> {xp.name} {x.shape} {x.nbytes/1e6:.0f} MB  counts={counts}")

    (outdir / "cache_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwritten: {outdir/'cache_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
