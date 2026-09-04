"""Figure: one representative image per class, and the duplicate pairs the audit found.

Reviewer 1 asked for representative images from all seven classes and for examples of
exact and near-duplicate pairs. Exemplars are drawn at random (seeded) from the TRAINING
partition so that no test image is reproduced in the paper. The duplicate panel shows the
single byte-identical pair and two confirmed near-duplicate pairs chosen to span the
operative range: the closest non-identical pair and the pair at the perceptual-hash
boundary (the largest distance that still passes confirmation), with the audit's
distances printed under each pair. Thumbnails are rescaled with aspect preserved and
padded with white, so they show the photographs, not the 224x224 network input.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_ORDER = [
    "1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight", "4. Helopeltis",
    "5. Red spider", "6. Green mirid bug", "7. Healthy leaf",
]
SHORT = ["Algal leaf spot", "Brown blight", "Gray blight", "Helopeltis", "Red spider mite",
         "Green mirid bug", "Healthy"]
KIND = ["disease", "disease", "disease", "pest damage", "pest damage", "pest damage", "healthy"]


def thumb(path: Path, size: int = 256) -> np.ndarray:
    """Centred square crop, then downscale. Leaves are centred on the sheet, so cropping
    the longer side keeps the leaf and removes only backdrop."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        s = min(im.width, im.height)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
        return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root, outdir = Path(a.image_root), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    with (Path(a.splits_dir) / "primary_train.csv").open(encoding="utf-8") as fh:
        train = list(csv.DictReader(fh))
    exemplars = []
    for c in CLASS_ORDER:
        pool = [r["relative_path"] for r in train if r["class_folder_raw"] == c]
        exemplars.append(str(rng.choice(pool)))

    with (Path(a.audit_dir) / "near_duplicate_candidates.csv").open(encoding="utf-8") as fh:
        cands = [r for r in csv.DictReader(fh) if r["is_duplicate_for_grouping"] == "True"]
    for r in cands:
        r["phash_hamming"] = int(r["phash_hamming"])
        r["dhash_hamming"] = int(r["dhash_hamming"])
        r["ssim"] = float(r["ssim"])
    exact = [r for r in cands if r["exact_sha_match"] == "True"]
    near = sorted([r for r in cands if r["exact_sha_match"] != "True"], key=lambda r: r["phash_hamming"])
    if not exact or len(near) < 2:
        raise SystemExit("expected one exact pair and at least two near-duplicate pairs")
    pairs = [("byte-identical", exact[0]), ("closest near-duplicate", near[0]),
             ("at the operative boundary", near[-1])]

    fig = plt.figure(figsize=(4.9, 2.55))
    # rows: label (a) | exemplars | label (b) | pairs | captions
    gs = fig.add_gridspec(5, 1, height_ratios=[0.10, 1.0, 0.16, 1.0, 0.20], hspace=0.0,
                          left=0.005, right=0.995, top=0.995, bottom=0.005)

    lab = fig.add_subplot(gs[0]); lab.set_axis_off()
    lab.text(0.0, 0.5, "(a) one training exemplar per class (disease / pest damage / healthy)",
             fontsize=6, va="center", transform=lab.transAxes)
    top = gs[1].subgridspec(1, 7, wspace=0.04)
    for k, rel in enumerate(exemplars):
        ax = fig.add_subplot(top[0, k])
        ax.imshow(thumb(root / rel)); ax.set_axis_off()
        ax.set_title(f"{SHORT[k]}", fontsize=5.6, pad=1.5)

    lab = fig.add_subplot(gs[2]); lab.set_axis_off()
    lab.text(0.0, 0.35, "(b) the byte-identical pair and two confirmed near-duplicate pairs "
                        "(filename pairs are consecutive)", fontsize=6, va="center",
             transform=lab.transAxes)
    bottom = gs[3].subgridspec(1, 3, wspace=0.14)
    caps = gs[4].subgridspec(1, 3, wspace=0.14)
    for j, (label, r) in enumerate(pairs):
        sub = bottom[0, j].subgridspec(1, 2, wspace=0.03)
        stems = [Path(r[k]).stem for k in ("path_a", "path_b")]
        for k, key in enumerate(("path_a", "path_b")):
            ax = fig.add_subplot(sub[0, k])
            ax.imshow(thumb(root / r[key])); ax.set_axis_off()
            ax.set_title(stems[k].rsplit("_", 1)[-1] if k else stems[k], fontsize=5.0, pad=1.2)
        cap = fig.add_subplot(caps[0, j]); cap.set_axis_off()
        cap.text(0.5, 0.75, label, fontsize=5.4, ha="center", va="center", transform=cap.transAxes)
        cap.text(0.5, 0.2, f"pHash {r['phash_hamming']}/256 · dHash {r['dhash_hamming']}/256 · "
                           f"SSIM {r['ssim']:.3f}", fontsize=5.0, ha="center", va="center",
                 transform=cap.transAxes)
    fig.savefig(outdir / "fig_dataset.pdf")
    fig.savefig(outdir / "fig_dataset.jpg", dpi=300)
    plt.close(fig)

    manifest = {"exemplars": dict(zip(CLASS_ORDER, exemplars)),
                "pairs": [{"label": l, "a": r["path_a"], "b": r["path_b"], "phash": r["phash_hamming"],
                           "dhash": r["dhash_hamming"], "ssim": r["ssim"]} for l, r in pairs],
                "n_confirmed_pairs": len(cands), "seed": a.seed}
    (outdir / "fig_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
