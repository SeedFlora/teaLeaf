"""Test whether confirmed near-duplicate pairs coincide with consecutive capture order.

teaLeafBD ships no leaf id, capture-session id or timestamp, so any duplicate
grouping rests on a similarity threshold the analyst chose. That is the sharpest
objection to a leakage audit on this dataset.

Filename sequence numbers provide an *independent* signal. Visual similarity alone
can be coincidental -- two different leaves may simply look alike. Visual similarity
that lands specifically on adjacent sequence numbers cannot plausibly be
coincidental at scale, because adjacency is not something a perceptual hash can see.

This script quantifies that coincidence and compares it against the rate expected if
confirmed pairs were positioned at random within their class, giving a
threshold-independent check on whether the flagged pairs are genuine repeat captures.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

NUM_RE = re.compile(r"(\d+)(?=\.[A-Za-z]+$)")


def seq(path: str) -> int | None:
    m = NUM_RE.search(path.split("/")[-1])
    return int(m.group(1)) if m else None


def main() -> int:
    audit = Path(sys.argv[1])
    out = Path(sys.argv[2])
    operative = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    rows = [r for r in csv.DictReader((audit / "near_duplicate_candidates.csv").open(encoding="utf-8"))]
    confirmed = [r for r in rows if r['is_duplicate_for_grouping'] == 'True' or r['passes_pixel_confirmation'] == 'True']
    operative_pairs = [r for r in confirmed if int(r["phash_hamming"]) <= operative]

    manifest = list(csv.DictReader(Path(sys.argv[4]).open(encoding="utf-8"))) if len(sys.argv) > 4 else []
    class_sizes = Counter(r["class_folder_raw"] for r in manifest)

    def analyze(pairs: list[dict], tag: str) -> dict:
        gaps: list[int] = []
        same_class = 0
        cross_class = 0
        adjacent = 0
        within_5 = 0
        by_class: dict[str, int] = defaultdict(int)

        for r in pairs:
            a, b = seq(r["path_a"]), seq(r["path_b"])
            if r["label_a"] == r["label_b"]:
                same_class += 1
                by_class[r["label_a"]] += 1
                if a is not None and b is not None:
                    g = abs(a - b)
                    gaps.append(g)
                    if g == 1:
                        adjacent += 1
                    if g <= 5:
                        within_5 += 1
            else:
                cross_class += 1

        # Expected adjacency if same-class pairs were placed uniformly at random:
        # for a class of n images there are C(n,2) possible pairs of which (n-1)
        # are adjacent, so P(adjacent) = 2/n.
        exp_adj = 0.0
        for cls, cnt in by_class.items():
            n = class_sizes.get(cls, 0)
            if n > 1:
                exp_adj += cnt * (2.0 / n)

        return {
            "tag": tag,
            "pairs": len(pairs),
            "same_class_pairs": same_class,
            "cross_class_pairs": cross_class,
            "same_class_with_sequence_gap_1": adjacent,
            "same_class_with_sequence_gap_le_5": within_5,
            "pct_same_class_adjacent": round(100 * adjacent / same_class, 2) if same_class else 0.0,
            "pct_same_class_within_5": round(100 * within_5 / same_class, 2) if same_class else 0.0,
            "expected_adjacent_if_random": round(exp_adj, 3),
            "observed_over_expected": round(adjacent / exp_adj, 1) if exp_adj > 0 else None,
            "gap_histogram": dict(sorted(Counter(gaps).items())[:15]),
            "per_class_same_class_pairs": dict(sorted(by_class.items())),
        }

    result = {
        "operative_phash_threshold": operative,
        "operative": analyze(operative_pairs, f"confirmed pairs with pHash <= {operative}"),
        "full_screen": analyze(confirmed, "all confirmed pairs within the loose screen"),
        "interpretation": (
            "Sequence adjacency is independent of the perceptual-hash threshold: a hash cannot "
            "observe filename order. An adjacency rate far above the random-placement expectation "
            "therefore indicates the flagged pairs are consecutive captures of the same subject, "
            "not coincidental visual similarity between distinct leaves."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key in ("operative", "full_screen"):
        d = result[key]
        print(f"\n=== {d['tag']} ===")
        print(f"  pairs                         : {d['pairs']}")
        print(f"  same-class / cross-class      : {d['same_class_pairs']} / {d['cross_class_pairs']}")
        print(f"  same-class, sequence gap == 1 : {d['same_class_with_sequence_gap_1']} "
              f"({d['pct_same_class_adjacent']}%)")
        print(f"  same-class, sequence gap <= 5 : {d['same_class_with_sequence_gap_le_5']} "
              f"({d['pct_same_class_within_5']}%)")
        print(f"  expected adjacent if random   : {d['expected_adjacent_if_random']}")
        print(f"  observed / expected           : {d['observed_over_expected']}x")
        print(f"  gap histogram                 : {d['gap_histogram']}")

    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

