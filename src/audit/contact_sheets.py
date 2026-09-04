"""Render side-by-side contact sheets for every confirmed duplicate pair.

Automated similarity scores are evidence, not proof. These sheets let a human
confirm at a glance that a flagged pair really is the same leaf photographed twice
rather than two similar leaves, and they are the artifact a reviewer would ask for.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw

NUM_RE = re.compile(r"(\d+)(?=\.[A-Za-z]+$)")
THUMB = 380
PAD = 10
BAR = 46


def seq(p: str) -> int | None:
    m = NUM_RE.search(p.split("/")[-1])
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-phash", type=int, default=32)
    ap.add_argument("--per-sheet", type=int, default=6)
    a = ap.parse_args()

    root = Path(a.image_root)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(Path(a.pairs).open(encoding="utf-8"))
            if r["is_duplicate_for_grouping"] == "True" and int(r["phash_hamming"]) <= a.max_phash]
    rows.sort(key=lambda r: -float(r["ssim"]))
    print(f"rendering {len(rows)} confirmed pairs (pHash <= {a.max_phash})")

    def thumb(rel: str) -> Image.Image:
        with Image.open(root / rel) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB), Image.LANCZOS)
            canvas = Image.new("RGB", (THUMB, THUMB), (24, 24, 24))
            canvas.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
            return canvas

    sheets = 0
    for start in range(0, len(rows), a.per_sheet):
        chunk = rows[start:start + a.per_sheet]
        w = THUMB * 2 + PAD * 3
        h = (THUMB + BAR + PAD) * len(chunk) + PAD
        sheet = Image.new("RGB", (w, h), (245, 245, 245))
        d = ImageDraw.Draw(sheet)

        y = PAD
        for r in chunk:
            sheet.paste(thumb(r["path_a"]), (PAD, y))
            sheet.paste(thumb(r["path_b"]), (PAD * 2 + THUMB, y))
            sa, sb = seq(r["path_a"]), seq(r["path_b"])
            gap = abs(sa - sb) if sa is not None and sb is not None else "?"
            d.text((PAD, y + THUMB + 4),
                   f"{r['path_a'].split('/')[-1]}   <->   {r['path_b'].split('/')[-1]}",
                   fill=(0, 0, 0))
            d.text((PAD, y + THUMB + 20),
                   f"SSIM={r['ssim']}  pHash={r['phash_hamming']}/256  "
                   f"dHash={r['dhash_hamming']}/256  sequence gap={gap}  "
                   f"class={r['label_a']}"
                   + ("  [EXACT SHA-256 MATCH]" if r["exact_sha_match"] == "True" else ""),
                   fill=(70, 70, 70))
            y += THUMB + BAR + PAD

        p = outdir / f"contact_sheet_{sheets + 1:02d}.jpg"
        sheet.save(p, quality=88, optimize=True)
        sheets += 1

    print(f"wrote {sheets} contact sheets -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

