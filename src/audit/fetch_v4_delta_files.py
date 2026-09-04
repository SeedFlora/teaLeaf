"""Download and inspect the files that version 4 adds relative to version 3.

These are kept in data/raw/v4/ purely as audit evidence. They are deliberately
NOT merged into the version-3 tree that the experiments read from: mixing files
across dataset releases would destroy the provenance guarantee that every
training image belongs to one citable, hash-verified release.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATASET_ID = "744vznw5k2"
API = "https://data.mendeley.com/public-api"
UA = "teaLeafBD-research-audit/1.0 (academic dataset version audit)"

TARGETS = ["brown_blight_00507.jpg", "brown_blight_00508.jpg"]


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)

    folders = get_json(f"{API}/datasets/{DATASET_ID}/folders/4")
    bb = next(f for f in folders if f["name"] == "2. Brown Blight")
    files = get_json(f"{API}/datasets/{DATASET_ID}/files?folder_id={bb['id']}&version=4")
    wanted = {f["filename"]: f for f in files if f["filename"] in TARGETS}

    rows = []
    for name in TARGETS:
        f = wanted.get(name)
        if f is None:
            print(f"NOT FOUND in v4 listing: {name}")
            continue
        cd = f["content_details"]
        dest = outdir / name
        req = urllib.request.Request(cd["download_url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        dest.write_bytes(data)

        digest = hashlib.sha256(data).hexdigest()
        expected = cd["sha256_hash"].lower()
        ok = digest == expected

        # Decode check without extra dependencies: verify the JPEG SOI/EOI markers
        # and read the SOF frame header for dimensions.
        width = height = None
        if data[:2] == b"\xff\xd8":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                    height = int.from_bytes(data[i + 5:i + 7], "big")
                    width = int.from_bytes(data[i + 7:i + 9], "big")
                    break
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                i += 2 + int.from_bytes(data[i + 2:i + 4], "big")

        rows.append({
            "filename": name,
            "bytes": len(data),
            "sha256_downloaded": digest,
            "sha256_official": expected,
            "hash_match": ok,
            "jpeg_soi": data[:2] == b"\xff\xd8",
            "jpeg_eoi": data[-2:] == b"\xff\xd9",
            "width": width,
            "height": height,
            "content_type": cd.get("content_type", ""),
            "created_date": cd.get("created_date", ""),
        })
        print(f"{name}: {len(data)} bytes  hash_match={ok}  {width}x{height}  created={cd.get('created_date','')}")

    (outdir / "v4_delta_files_evidence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with (outdir / "v4_delta_files_evidence.csv").open("w", newline="", encoding="utf-8") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\nEvidence written to {outdir}")
    print("NOTE: these files are audit evidence only and are NOT part of the experiment source tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
