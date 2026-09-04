"""Fetch the authoritative per-file manifests for teaLeafBD from the Mendeley Data public API.

Mendeley exposes, for every file in a published dataset version, the filename,
byte size and a repository-side SHA-256 digest.  Pulling those manifests lets us
perform an exact file-level comparison between version 3 and version 4 -- and to
verify the locally supplied archive against the official release -- without
downloading several gigabytes of image data.

No authentication is used and no access control is bypassed; these are the same
public endpoints the Mendeley Data web interface itself calls.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

DATASET_ID = "744vznw5k2"
API = "https://data.mendeley.com/public-api"
UA = "teaLeafBD-research-audit/1.0 (academic dataset version audit)"


def get_json(url: str, retries: int = 4) -> object:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def fetch_version(version: int) -> list[dict]:
    """Return one row per file in the given dataset version."""
    folders = get_json(f"{API}/datasets/{DATASET_ID}/folders/{version}")
    by_id = {f["id"]: f for f in folders}

    def folder_path(fid: str) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        cur = fid
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            parts.append(by_id[cur]["name"])
            cur = by_id[cur].get("parent_folder_id")
        return "/".join(reversed(parts))

    rows: list[dict] = []
    # "root" plus every declared folder; Mendeley lists files per folder only.
    targets = [("root", "")] + [(f["id"], folder_path(f["id"])) for f in folders]

    for fid, path in targets:
        files = get_json(f"{API}/datasets/{DATASET_ID}/files?folder_id={fid}&version={version}")
        if not isinstance(files, list):
            print(f"  WARNING folder {fid!r} returned non-list: {files!r}", file=sys.stderr)
            continue
        for f in files:
            cd = f.get("content_details") or {}
            rows.append(
                {
                    "version": version,
                    "folder_id": fid,
                    "folder_path": path,
                    "filename": f.get("filename", ""),
                    "relative_path": f"{path}/{f.get('filename','')}" if path else f.get("filename", ""),
                    "size_bytes": f.get("size") or cd.get("size") or "",
                    "sha256": (cd.get("sha256_hash") or "").lower(),
                    "content_type": cd.get("content_type", ""),
                    "created_date": cd.get("created_date", ""),
                    "file_id": f.get("id", ""),
                }
            )
        print(f"  v{version} folder '{path or '(root)'}': {len([r for r in rows if r['folder_id'] == fid])} files")
    return rows


def main() -> int:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    fields = [
        "version", "folder_id", "folder_path", "filename", "relative_path",
        "size_bytes", "sha256", "content_type", "created_date", "file_id",
    ]

    for version in (3, 4):
        print(f"=== fetching version {version} ===")
        rows = fetch_version(version)
        out = outdir / f"mendeley_official_manifest_v{version}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        n_hash = sum(1 for r in rows if r["sha256"])
        print(f"v{version}: {len(rows)} files, {n_hash} with sha256 -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
