"""Locate gaps in the sequential file numbering of each teaLeafBD class folder.

The official version-4 README states a per-class image count and shows that file
names are sequential (e.g. brown_blight_00001 ... brown_blight_00508).  Any class
whose local file count is lower than the README count must therefore have missing
sequence numbers, and those numbers identify precisely which images are absent.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Per-class counts transcribed from Table 1 of the official version-4 read_me.pdf
# (Mendeley Data 10.17632/744vznw5k2.4, file sha256
# f9e64e7d640120ddae2c52dfb5d948c0228840d9a3fb8fa9e32a01d43f411685).
README_V4_COUNTS = {
    "1. Tea algal leaf spot": 418,
    "2. Brown Blight": 508,
    "3. Gray Blight": 1013,
    "4. Helopeltis": 607,
    "5. Red spider": 515,
    "6. Green mirid bug": 1282,
    "7. Healthy leaf": 935,
}

NUM_RE = re.compile(r"(\d+)(?=\.[A-Za-z]+$)")


def analyze(root: Path) -> dict:
    report: dict[str, dict] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        numbers: set[int] = set()
        unparsed: list[str] = []
        for f in files:
            m = NUM_RE.search(f.name)
            if m:
                numbers.add(int(m.group(1)))
            else:
                unparsed.append(f.name)

        expected = README_V4_COUNTS.get(class_dir.name)
        lo, hi = (min(numbers), max(numbers)) if numbers else (0, 0)
        missing = sorted(set(range(lo, hi + 1)) - numbers) if numbers else []

        report[class_dir.name] = {
            "actual_file_count": len(files),
            "readme_v4_count": expected,
            "delta_vs_readme": (len(files) - expected) if expected is not None else None,
            "numbering_min": lo,
            "numbering_max": hi,
            "distinct_numbers": len(numbers),
            "missing_numbers_within_range": missing,
            "unparsed_filenames": unparsed,
        }
    return report


def main() -> int:
    root = Path(sys.argv[1])
    report = analyze(root)
    print(json.dumps(report, indent=2))

    total_actual = sum(v["actual_file_count"] for v in report.values())
    total_readme = sum(v["readme_v4_count"] or 0 for v in report.values())
    print(f"\nTOTAL actual={total_actual} readme_v4={total_readme} delta={total_actual - total_readme}")

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "per_class": report,
                    "total_actual": total_actual,
                    "total_readme_v4": total_readme,
                    "delta": total_actual - total_readme,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
