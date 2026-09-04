"""Construct the leakage-aware primary split and the naive comparison split.

Two splits are produced from the same manifest:

PRIMARY (leakage-aware) — the split every headline result is reported on.
    The unit of assignment is the duplicate/near-duplicate *group*, not the image,
    so every member of a cluster lands in exactly one split. Assignment is greedy
    over groups ordered by decreasing size, each group going to whichever split is
    currently furthest below its target quota for that group's dominant class.
    Ties break deterministically on a seeded permutation, so the split is
    reproducible from the manifest alone.

NAIVE — image-level stratified random assignment, ignoring duplicate structure.
    This exists only to quantify how much a conventional split inflates results
    (RQ1). It is never used as a headline performance claim.

Both splits are built BEFORE any augmentation, and augmentation is applied only to
the training partition at load time.

Cross-label clusters — the same visual content filed under two different class
folders — are quarantined rather than assigned, because there is no defensible way
to pick one of the conflicting labels without expert adjudication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_split(path: Path, rows: list[dict]) -> None:
    fields = ["relative_path", "class_folder_raw", "normalized_class", "group_id", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {r["relative_path"]: r for r in load_csv(Path(a.manifest))}
    groups = load_csv(Path(a.groups))
    clusters = load_csv(Path(a.clusters))

    for g in groups:
        manifest[g["relative_path"]]["group_id"] = g["group_id"]

    # ---- quarantine decisions, made explicit ---- #
    quarantine: dict[str, str] = {}
    for r in manifest.values():
        if r["decode_ok"] != "True":
            quarantine[r["relative_path"]] = f"undecodable:{r['corruption_status']}"

    cross_label_groups = {c["group_id"] for c in clusters if c["cross_label"] in ("True", "true", True)}
    for r in manifest.values():
        if r.get("group_id") in cross_label_groups and r["relative_path"] not in quarantine:
            quarantine[r["relative_path"]] = "cross_label_duplicate_cluster_unadjudicated"

    eligible = [r for r in manifest.values() if r["relative_path"] not in quarantine]
    print(f"images total={len(manifest)} eligible={len(eligible)} quarantined={len(quarantine)}")

    # ---- group-level aggregation ---- #
    gmembers: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        gmembers[r["group_id"]].append(r)

    classes = sorted({r["class_folder_raw"] for r in eligible})
    targets = {"train": a.train, "validation": a.val, "test": a.test}

    class_totals = Counter(r["class_folder_raw"] for r in eligible)
    quota = {s: {c: class_totals[c] * f for c in classes} for s, f in targets.items()}
    filled: dict[str, Counter] = {s: Counter() for s in targets}

    rng = random.Random(a.seed)
    order = sorted(gmembers.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # Seeded jitter only affects ordering among equal-sized groups.
    jitter = {gid: rng.random() for gid, _ in order}
    order.sort(key=lambda kv: (-len(kv[1]), jitter[kv[0]], kv[0]))

    assignment: dict[str, str] = {}
    for gid, members in order:
        dominant = Counter(m["class_folder_raw"] for m in members).most_common(1)[0][0]
        counts = Counter(m["class_folder_raw"] for m in members)
        # Choose the split with the largest remaining deficit in the dominant class;
        # break ties toward the split that is furthest below its overall target.
        best = max(
            targets,
            key=lambda s: (
                quota[s][dominant] - filled[s][dominant],
                sum(quota[s].values()) - sum(filled[s].values()),
                s,
            ),
        )
        assignment[gid] = best
        filled[best].update(counts)

    primary: dict[str, list[dict]] = {s: [] for s in targets}
    for gid, members in gmembers.items():
        primary[assignment[gid]].extend(members)
    for s in primary:
        primary[s].sort(key=lambda r: r["relative_path"])

    # ---- naive image-level stratified split (leakage-sensitivity control) ---- #
    rng2 = random.Random(a.seed)
    naive: dict[str, list[dict]] = {s: [] for s in targets}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        by_class[r["class_folder_raw"]].append(r)
    for c in classes:
        items = sorted(by_class[c], key=lambda r: r["relative_path"])
        rng2.shuffle(items)
        n = len(items)
        n_tr = int(round(n * a.train))
        n_va = int(round(n * a.val))
        naive["train"].extend(items[:n_tr])
        naive["validation"].extend(items[n_tr:n_tr + n_va])
        naive["test"].extend(items[n_tr + n_va:])
    for s in naive:
        naive[s].sort(key=lambda r: r["relative_path"])

    # ---- write ---- #
    files: dict[str, str] = {}
    for name, split in (("primary", primary), ("naive", naive)):
        for s in ("train", "validation", "test"):
            p = outdir / f"{name}_{s}.csv"
            write_split(p, split[s])
            files[p.name] = sha256_file(p)

    qpath = outdir.parent / "quarantine" / "quarantine_manifest.csv"
    qpath.parent.mkdir(parents=True, exist_ok=True)
    with qpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "class_folder_raw", "group_id", "reason"])
        for rp, reason in sorted(quarantine.items()):
            r = manifest[rp]
            w.writerow([rp, r["class_folder_raw"], r.get("group_id", ""), reason])
    files[qpath.name] = sha256_file(qpath)

    def dist(split: dict[str, list[dict]]) -> dict:
        return {
            s: {
                "n": len(rows),
                "fraction": round(len(rows) / max(1, sum(len(v) for v in split.values())), 4),
                "per_class": dict(Counter(r["class_folder_raw"] for r in rows)),
                "n_groups": len({r["group_id"] for r in rows}),
            }
            for s, rows in split.items()
        }

    summary = {
        "seed": a.seed,
        "target_fractions": targets,
        "images_total": len(manifest),
        "images_eligible": len(eligible),
        "images_quarantined": len(quarantine),
        "quarantine_reasons": dict(Counter(quarantine.values())),
        "groups_total": len(gmembers),
        "primary_split": dist(primary),
        "naive_split": dist(naive),
        "file_sha256": files,
        "notes": (
            "Primary split assigns whole duplicate/near-duplicate groups, so achieved "
            "fractions deviate slightly from the 70/15/15 target; the achieved values are "
            "reported above and are the ones cited. The naive split is an image-level "
            "stratified control used solely for the leakage-sensitivity experiment (RQ1)."
        ),
    }
    (outdir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== primary (leakage-aware) ===")
    for s, d in summary["primary_split"].items():
        print(f"  {s:<11} n={d['n']:<5} frac={d['fraction']:<7} groups={d['n_groups']}")
    print("=== naive (control) ===")
    for s, d in summary["naive_split"].items():
        print(f"  {s:<11} n={d['n']:<5} frac={d['fraction']}")
    print(f"\nquarantined: {len(quarantine)} -> {dict(Counter(quarantine.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
