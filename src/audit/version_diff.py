"""File-level comparison of teaLeafBD Mendeley version 3 against version 4.

Inputs are the authoritative per-file manifests fetched from the Mendeley Data
public API (filename, byte size, repository-side SHA-256).

Known limitation, recorded explicitly rather than silently ignored: the public
API caps a folder listing at 1000 entries and honours no pagination parameter we
could find (limit/offset/page/marker were all tested and ignored).  Two classes
therefore have truncated official listings -- "3. Gray Blight" (1013 files) and
"6. Green mirid bug" (1282 files).  Every other class, including "2. Brown
Blight" where the version discrepancy lives, is covered completely.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TRUNCATED_FOLDERS = {"3. Gray Blight", "6. Green mirid bug"}
API_FOLDER_CAP = 1000


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["relative_path"]: r for r in csv.DictReader(fh)}


def main() -> int:
    mdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    v3 = load(mdir / "mendeley_official_manifest_v3.csv")
    v4 = load(mdir / "mendeley_official_manifest_v4.csv")

    p3, p4 = set(v3), set(v4)
    only3, only4, shared = sorted(p3 - p4), sorted(p4 - p3), sorted(p3 & p4)

    changed = [p for p in shared if v3[p]["sha256"] != v4[p]["sha256"]]
    identical = [p for p in shared if v3[p]["sha256"] == v4[p]["sha256"]]

    # Content-level view: a renamed-but-identical file shows up here.
    h3 = defaultdict(list)
    h4 = defaultdict(list)
    for p, r in v3.items():
        h3[r["sha256"]].append(p)
    for p, r in v4.items():
        h4[r["sha256"]].append(p)
    hash_only3 = sorted(set(h3) - set(h4))
    hash_only4 = sorted(set(h4) - set(h3))

    rows: list[dict] = []
    for p in only3:
        rows.append({"relative_path": p, "status": "only_in_v3", "sha256_v3": v3[p]["sha256"],
                     "sha256_v4": "", "size_v3": v3[p]["size_bytes"], "size_v4": ""})
    for p in only4:
        rows.append({"relative_path": p, "status": "only_in_v4", "sha256_v3": "",
                     "sha256_v4": v4[p]["sha256"], "size_v3": "", "size_v4": v4[p]["size_bytes"]})
    for p in changed:
        rows.append({"relative_path": p, "status": "content_changed", "sha256_v3": v3[p]["sha256"],
                     "sha256_v4": v4[p]["sha256"], "size_v3": v3[p]["size_bytes"], "size_v4": v4[p]["size_bytes"]})

    diff_csv = outdir / "version_diff.csv"
    with diff_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["relative_path", "status", "sha256_v3", "sha256_v4", "size_v3", "size_v4"])
        w.writeheader()
        w.writerows(rows)

    # Per-class counts as reported by the API, flagged where truncation applies.
    def per_class(man: dict[str, dict]) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for r in man.values():
            c[r["folder_path"] or "(root)"] += 1
        return dict(c)

    c3, c4 = per_class(v3), per_class(v4)

    summary = {
        "v3_files_listed": len(v3),
        "v4_files_listed": len(v4),
        "only_in_v3": len(only3),
        "only_in_v4": len(only4),
        "shared_paths": len(shared),
        "shared_identical_content": len(identical),
        "shared_content_changed": len(changed),
        "content_hashes_unique_to_v3": len(hash_only3),
        "content_hashes_unique_to_v4": len(hash_only4),
        "only_in_v3_paths": only3,
        "only_in_v4_paths": only4,
        "content_changed_paths": changed,
        "per_class_v3_listed": c3,
        "per_class_v4_listed": c4,
        "api_folder_cap": API_FOLDER_CAP,
        "folders_truncated_by_api_cap": sorted(TRUNCATED_FOLDERS),
        "coverage_caveat": (
            "The Mendeley public API returns at most 1000 files per folder and ignored every "
            "pagination parameter tested (limit, offset, page, marker). Listings for "
            "'3. Gray Blight' and '6. Green mirid bug' are therefore truncated at 1000, so the "
            "diff is complete for all other classes but partial for those two."
        ),
    }

    (outdir / "version_diff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=2))
    print("\nonly_in_v3:", only3 or "(none)")
    print("only_in_v4:", only4 or "(none)")
    print("content_changed:", changed or "(none)")
    print("\nper-class listed counts (v3 -> v4):")
    for k in sorted(set(c3) | set(c4)):
        flag = "  [TRUNCATED AT API CAP]" if k in TRUNCATED_FOLDERS else ""
        print(f"  {k:<28} {c3.get(k, 0):>5} -> {c4.get(k, 0):>5}{flag}")
    print(f"\nwritten: {diff_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
