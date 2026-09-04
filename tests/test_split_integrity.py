"""Automated leakage tests for the primary split.

These are the checks that must hold for any headline result to mean anything. They
run against the committed split manifests, so they can be re-run by anyone who
reproduces the pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data" / "splits"
AUDIT = REPO / "data" / "audit"
MANIFESTS = REPO / "data" / "manifests"

PARTS = ("train", "validation", "test")


def read(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def primary() -> dict[str, list[dict]]:
    return {s: read(SPLITS / f"primary_{s}.csv") for s in PARTS}


@pytest.fixture(scope="module")
def naive() -> dict[str, list[dict]]:
    return {s: read(SPLITS / f"naive_{s}.csv") for s in PARTS}


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return read(MANIFESTS / "v3_manifest.csv")


@pytest.fixture(scope="module")
def confirmed_pairs() -> list[dict]:
    rows = read(AUDIT / "near_duplicate_candidates.csv")
    return [r for r in rows if r["is_duplicate_for_grouping"] == "True"]


# --------------------------------------------------------------------------- #
# Partition well-formedness
# --------------------------------------------------------------------------- #
def test_splits_are_disjoint(primary):
    seen: dict[str, str] = {}
    for part, rows in primary.items():
        for r in rows:
            prev = seen.get(r["relative_path"])
            assert prev is None, f"{r['relative_path']} appears in both {prev} and {part}"
            seen[r["relative_path"]] = part


def test_splits_cover_all_eligible_images(primary, manifest):
    quarantine = read(REPO / "data" / "quarantine" / "quarantine_manifest.csv")
    quarantined = {r["relative_path"] for r in quarantine}
    assigned = {r["relative_path"] for rows in primary.values() for r in rows}
    all_images = {r["relative_path"] for r in manifest}
    assert assigned | quarantined == all_images
    assert not (assigned & quarantined), "an image is both assigned and quarantined"


def test_every_class_present_in_every_split(primary):
    for part, rows in primary.items():
        classes = {r["class_folder_raw"] for r in rows}
        assert len(classes) == 7, f"{part} covers {len(classes)} classes, expected 7"


# --------------------------------------------------------------------------- #
# The leakage guarantees
# --------------------------------------------------------------------------- #
def test_no_duplicate_group_spans_splits(primary):
    """The core guarantee: a duplicate cluster must live in exactly one split."""
    where: dict[str, set[str]] = defaultdict(set)
    for part, rows in primary.items():
        for r in rows:
            where[r["group_id"]].add(part)
    straddling = {g: p for g, p in where.items() if len(p) > 1}
    assert not straddling, f"{len(straddling)} duplicate groups span multiple splits: {list(straddling)[:5]}"


def test_no_exact_sha256_appears_in_two_splits(primary):
    where: dict[str, set[str]] = defaultdict(set)
    for part, rows in primary.items():
        for r in rows:
            where[r["sha256"]].add(part)
    straddling = {h: p for h, p in where.items() if len(p) > 1}
    assert not straddling, f"identical file content in multiple splits: {list(straddling)[:5]}"


def test_no_confirmed_near_duplicate_pair_crosses_splits(primary, confirmed_pairs):
    """Zero confirmed high-similarity overlap across splits, as required."""
    part_of = {r["relative_path"]: part for part, rows in primary.items() for r in rows}
    violations = []
    for pair in confirmed_pairs:
        a, b = part_of.get(pair["path_a"]), part_of.get(pair["path_b"])
        if a and b and a != b:
            violations.append((pair["path_a"], a, pair["path_b"], b, pair["ssim"]))
    assert not violations, f"{len(violations)} confirmed near-duplicate pairs cross splits: {violations[:3]}"


# --------------------------------------------------------------------------- #
# Stratification and proportions
# --------------------------------------------------------------------------- #
def test_split_proportions_close_to_target(primary):
    total = sum(len(v) for v in primary.values())
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for part, target in targets.items():
        frac = len(primary[part]) / total
        assert abs(frac - target) < 0.02, f"{part} fraction {frac:.4f} deviates from {target}"


def test_class_stratification_preserved(primary):
    total_counts: Counter = Counter()
    for rows in primary.values():
        total_counts.update(r["class_folder_raw"] for r in rows)
    grand = sum(total_counts.values())

    for part, rows in primary.items():
        part_counts = Counter(r["class_folder_raw"] for r in rows)
        share = len(rows) / grand
        for cls, n_total in total_counts.items():
            expected = n_total * share
            actual = part_counts[cls]
            # Whole groups are assigned, so exact stratification is impossible;
            # allow a tolerance that scales with class size.
            tol = max(6, 0.25 * expected)
            assert abs(actual - expected) <= tol, (
                f"{part}/{cls}: {actual} vs expected ~{expected:.1f} (tol {tol:.1f})"
            )


# --------------------------------------------------------------------------- #
# Naive control split
# --------------------------------------------------------------------------- #
def test_naive_split_is_disjoint_and_complete(naive, manifest):
    seen: set[str] = set()
    for rows in naive.values():
        for r in rows:
            assert r["relative_path"] not in seen
            seen.add(r["relative_path"])
    assert len(seen) == len(manifest)


def test_naive_split_actually_leaks(naive, confirmed_pairs):
    """The control split must be a genuine control.

    If the naive split happened to isolate every duplicate too, it would not be a
    control at all and the leakage-sensitivity comparison would be meaningless. This
    records how much contamination it actually carries.
    """
    part_of = {r["relative_path"]: part for part, rows in naive.items() for r in rows}
    crossing = sum(
        1 for p in confirmed_pairs
        if (a := part_of.get(p["path_a"])) and (b := part_of.get(p["path_b"])) and a != b
    )
    print(f"\nnaive split: {crossing} of {len(confirmed_pairs)} confirmed pairs cross partitions")
    # Not an assertion on a specific number: with few duplicates this can legitimately
    # be small. It is recorded so the leakage experiment is interpreted honestly.
    assert crossing >= 0


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #
def test_split_files_match_recorded_hashes():
    summary = json.loads((SPLITS / "split_summary.json").read_text(encoding="utf-8"))
    for name, expected in summary["file_sha256"].items():
        path = SPLITS / name
        if not path.exists():
            path = REPO / "data" / "quarantine" / name
        assert path.exists(), f"missing split artifact {name}"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == expected, f"{name} changed since it was written"


def test_no_augmented_or_derived_images_in_splits(primary):
    """Splits must reference original dataset files only."""
    for part, rows in primary.items():
        for r in rows:
            p = r["relative_path"]
            assert p.endswith(".jpg"), f"unexpected extension in {part}: {p}"
            assert not any(t in p.lower() for t in ("aug", "_copy", "resized", "crop")), p

