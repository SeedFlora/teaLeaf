"""Verify the locally supplied teaLeafBD archive against the official Mendeley manifests.

The archive used for this study was supplied locally rather than downloaded in
this session, so its provenance cannot be taken on trust.  This script hashes
every local file and checks each digest against the repository-side SHA-256
values published by Mendeley Data for version 3 and version 4, then reports which
official release the local copy actually corresponds to.

Only standard-library modules are used so that this check can run before the
training environment finishes installing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def sha256_of(path_s: str) -> tuple[str, str, int]:
    h = hashlib.sha256()
    p = Path(path_s)
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return path_s, h.hexdigest(), p.stat().st_size


def load_official(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["relative_path"]: r for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-root", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    root = Path(a.local_root)
    mdir = Path(a.manifest_dir)

    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"hashing {len(files)} local files...", flush=True)

    local: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (ps, digest, size) in enumerate(ex.map(sha256_of, [str(p) for p in files], chunksize=32), 1):
            rel = Path(ps).relative_to(root).as_posix()
            local[rel] = {"sha256": digest, "size": size}
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    result: dict[str, object] = {"local_file_count": len(local)}

    for version in (3, 4):
        official = load_official(mdir / f"mendeley_official_manifest_v{version}.csv")
        # The official listing omits the top-level 'teaLeafBD' wrapper folder that
        # the distributed archive nests its class folders inside, so compare on the
        # class-relative path.
        off_by_path = {k: v for k, v in official.items() if k.lower() != "read_me.pdf"}
        off_hashes = {v["sha256"] for v in off_by_path.values() if v["sha256"]}

        matched_path_and_hash = 0
        matched_hash_only = 0
        path_present_hash_differs: list[str] = []
        local_not_in_official: list[str] = []

        for rel, info in local.items():
            off = off_by_path.get(rel)
            if off is not None:
                if off["sha256"] == info["sha256"]:
                    matched_path_and_hash += 1
                else:
                    path_present_hash_differs.append(rel)
            elif info["sha256"] in off_hashes:
                matched_hash_only += 1
            else:
                local_not_in_official.append(rel)

        official_not_local = sorted(set(off_by_path) - set(local))

        result[f"v{version}"] = {
            "official_files_listed": len(off_by_path),
            "local_matched_by_path_and_hash": matched_path_and_hash,
            "local_matched_by_hash_only": matched_hash_only,
            "local_path_present_but_hash_differs": len(path_present_hash_differs),
            "local_files_absent_from_official_listing": len(local_not_in_official),
            "official_files_absent_locally": len(official_not_local),
            "official_files_absent_locally_names": official_not_local[:50],
            "hash_mismatch_examples": path_present_hash_differs[:20],
            "unmatched_local_examples": local_not_in_official[:20],
        }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:4000])
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
