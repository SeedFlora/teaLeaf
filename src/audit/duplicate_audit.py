"""Exact and near-duplicate audit for teaLeafBD, producing leakage-safe grouping keys.

Method, in three stages so that "candidate" and "confirmed duplicate" never get
conflated:

1. EXACT duplicates      — identical SHA-256. No judgement involved.
2. NEAR-duplicate candidates — cheap 256-bit perceptual-hash (pHash) screening
   over all pairs. This stage is deliberately permissive; its job is recall.
3. CONFIRMATION          — every candidate is re-tested with two independent
   measures: an independent 256-bit difference hash (dHash), and structural
   similarity (SSIM) computed on the actual decoded pixels. A pair is only called
   a confirmed near-duplicate when the pixel evidence agrees.

Clusters are then the connected components of the confirmed-duplicate graph. Using
connected components rather than pairs matters: if A~B and B~C but A and C are not
directly similar, all three must still land in the same split, otherwise leakage
re-enters through the transitive chain.

Cross-label clusters (the same visual content appearing under two class folders)
are reported separately and never silently resolved to one label.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Union-Find
# --------------------------------------------------------------------------- #
class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def hex_to_bits(hexes: list[str], nbits: int) -> np.ndarray:
    """Unpack hex perceptual hashes into a uint8 bit matrix (n_images, nbits)."""
    out = np.zeros((len(hexes), nbits), dtype=np.uint8)
    for i, h in enumerate(hexes):
        if not h:
            continue
        v = int(h, 16)
        # Most-significant bit first, so bit ordering is consistent across rows.
        out[i] = np.array([(v >> (nbits - 1 - j)) & 1 for j in range(nbits)], dtype=np.uint8)
    return out


def pairwise_candidates(bits: np.ndarray, threshold: int, block: int = 512):
    """Yield (i, j, hamming) for all pairs within `threshold` bits, i < j.

    Blocked matrix multiplication keeps memory bounded while still using BLAS:
    for 0/1 matrices, hamming(a, b) = popcount(a) + popcount(b) - 2 * a.b
    """
    n, nbits = bits.shape
    f = bits.astype(np.float32)
    pc = bits.sum(axis=1).astype(np.int32)

    for s in range(0, n, block):
        e = min(s + block, n)
        dots = f[s:e] @ f.T                                    # (block, n)
        ham = pc[s:e, None] + pc[None, :] - 2 * dots.astype(np.int32)
        # Only consider j > i to emit each pair once.
        for local_i in range(e - s):
            i = s + local_i
            row = ham[local_i]
            js = np.nonzero((row <= threshold))[0]
            for j in js:
                if j > i:
                    yield i, int(j), int(row[j])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phash-candidate-threshold", type=int, default=32,
                    help="Hamming distance on the 256-bit pHash used for the OPERATIVE grouping.")
    ap.add_argument("--phash-screen-threshold", type=int, default=64,
                    help="Loose screening distance; all pairs within this are scored once so that a "
                         "threshold sensitivity curve can be derived without rescoring.")
    ap.add_argument("--dhash-confirm-threshold", type=int, default=48,
                    help="Independent 256-bit dHash agreement required to confirm.")
    ap.add_argument("--ssim-confirm-threshold", type=float, default=0.90,
                    help="Pixel-level SSIM required to confirm a near-duplicate.")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path(a.image_root)

    with Path(a.manifest).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    print(f"loaded {n} manifest rows")

    paths = [r["relative_path"] for r in rows]
    labels = [r["class_folder_raw"] for r in rows]
    idx_of = {p: i for i, p in enumerate(paths)}
    dsu = DSU(n)

    # ---------------- Stage 1: exact duplicates ---------------- #
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["sha256"]:
            by_sha[r["sha256"]].append(i)
    exact_groups = {s: v for s, v in by_sha.items() if len(v) > 1}

    exact_rows = []
    for gi, (sha, members) in enumerate(sorted(exact_groups.items()), 1):
        member_labels = sorted({labels[m] for m in members})
        for m in members:
            dsu.union(members[0], m)
            exact_rows.append({
                "exact_group_id": f"EX{gi:04d}",
                "sha256": sha,
                "relative_path": paths[m],
                "class_folder_raw": labels[m],
                "group_size": len(members),
                "cross_label": len(member_labels) > 1,
                "labels_in_group": "|".join(member_labels),
            })
    print(f"exact duplicate groups: {len(exact_groups)}  "
          f"files involved: {sum(len(v) for v in exact_groups.values())}")

    # ---------------- Stage 2: near-duplicate candidates ---------------- #
    decodable = [i for i, r in enumerate(rows) if r["decode_ok"] == "True" and r["phash16"]]
    print(f"screening {len(decodable)} decodable images at the loose distance "
          f"(pHash Hamming <= {a.phash_screen_threshold} of 256) so the threshold curve "
          f"can be derived from a single scoring pass")

    sub_phash = hex_to_bits([rows[i]["phash16"] for i in decodable], 256)
    cands = [(decodable[i], decodable[j], d)
             for i, j, d in pairwise_candidates(sub_phash, a.phash_screen_threshold)]
    print(f"screened pairs: {len(cands)}")

    # ---------------- Stage 3: independent confirmation ---------------- #
    sub_dhash = hex_to_bits([rows[i]["dhash16"] for i in decodable], 256)
    pos_in_sub = {g: k for k, g in enumerate(decodable)}

    from skimage.metrics import structural_similarity as ssim
    from PIL import Image

    cache: dict[int, np.ndarray] = {}

    def gray(i: int) -> np.ndarray:
        if i not in cache:
            with Image.open(root / paths[i]) as im:
                cache[i] = np.asarray(
                    im.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32
                )
            if len(cache) > 4000:                       # bound memory on a long run
                cache.pop(next(iter(cache)))
        return cache[i]

    confirmed_rows = []
    n_conf = 0
    for k, (i, j, phd) in enumerate(cands, 1):
        dhd = int(np.count_nonzero(sub_dhash[pos_in_sub[i]] != sub_dhash[pos_in_sub[j]]))
        s = float(ssim(gray(i), gray(j), data_range=255.0))
        is_exact = rows[i]["sha256"] == rows[j]["sha256"]
        pixel_agrees = dhd <= a.dhash_confirm_threshold and s >= a.ssim_confirm_threshold

        # Whole-frame SSIM is a weak confirmer on this dataset: every image is a
        # single detached leaf on a near-uniform studio backdrop, and the leaf
        # occupies only a small fraction of the frame, so the background dominates
        # the score. Visual review of the pairs that pass pixel agreement only at
        # loose perceptual distances confirmed they are different leaves sharing a
        # backdrop. Grouping therefore additionally requires perceptual-hash
        # agreement, which is sensitive to the leaf structure itself.
        is_duplicate = is_exact or (pixel_agrees and phd <= a.phash_candidate_threshold)
        if is_duplicate:
            dsu.union(i, j)
            n_conf += 1
        confirmed_rows.append({
            "path_a": paths[i], "path_b": paths[j],
            "label_a": labels[i], "label_b": labels[j],
            "cross_label": labels[i] != labels[j],
            "phash_hamming": phd, "dhash_hamming": dhd,
            "ssim": round(s, 5),
            "exact_sha_match": is_exact,
            "passes_pixel_confirmation": pixel_agrees,
            "is_duplicate_for_grouping": is_duplicate,
        })
        if k % 2000 == 0:
            print(f"  confirmed {n_conf}/{k} candidates checked", flush=True)

    print(f"confirmed near-duplicate pairs: {n_conf} of {len(cands)} candidates")

    # ---------------- Cluster assignment ---------------- #
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[dsu.find(i)].append(i)

    # Stable, content-derived group ids: singletons keep their own identity so
    # that every image has a group_id and the split logic needs no special case.
    group_of: dict[str, str] = {}
    multi = 0
    cluster_rows = []
    for gi, (_, members) in enumerate(sorted(comp.items(), key=lambda kv: (-len(kv[1]), paths[kv[1][0]])), 1):
        gid = f"G{gi:05d}"
        member_labels = sorted({labels[m] for m in members})
        if len(members) > 1:
            multi += 1
        for m in members:
            group_of[paths[m]] = gid
        cluster_rows.append({
            "group_id": gid,
            "size": len(members),
            "n_distinct_labels": len(member_labels),
            "cross_label": len(member_labels) > 1,
            "labels": "|".join(member_labels),
            "members": "|".join(paths[m] for m in members),
        })

    conflicts = [c for c in cluster_rows if c["cross_label"]]
    sizes = [c["size"] for c in cluster_rows]

    # ---------------- threshold sensitivity curve ---------------- #
    # A reviewer's sharpest objection is that the duplicate grouping -- and hence any
    # leakage-inflation figure derived from it -- is a function of a self-chosen
    # threshold. teaLeafBD ships no leaf id, capture-session id or timestamp, so there
    # is no external key to validate against. The honest answer is to show how the
    # grouping behaves across the whole threshold range rather than defend one point.
    sweep_rows = []
    confirmed_pairs = [(i, j, phd) for (i, j, phd), c in zip(cands, confirmed_rows)
                       if c["passes_pixel_confirmation"] or c["exact_sha_match"]]
    for t in range(0, a.phash_screen_threshold + 1, 2):
        d2 = DSU(n)
        for sha, members in exact_groups.items():
            for m in members:
                d2.union(members[0], m)
        linked = 0
        for i, j, phd in confirmed_pairs:
            if phd <= t:
                d2.union(i, j)
                linked += 1
        comp2: dict[int, int] = defaultdict(int)
        for i in range(n):
            comp2[d2.find(i)] += 1
        szs = list(comp2.values())
        multi_sizes = [s for s in szs if s > 1]
        sweep_rows.append({
            "phash_threshold": t,
            "threshold_fraction_of_256_bits": round(t / 256, 4),
            "confirmed_pairs_linked": linked,
            "total_groups": len(szs),
            "multi_image_groups": len(multi_sizes),
            "images_in_multi_image_groups": sum(multi_sizes),
            "pct_images_grouped": round(100 * sum(multi_sizes) / n, 3),
            "largest_group_size": max(szs) if szs else 0,
        })

    # ---------------- Outputs ---------------- #
    def write(name: str, data: list[dict]) -> None:
        p = outdir / name
        with p.open("w", newline="", encoding="utf-8") as fh:
            if data:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
            else:
                fh.write("")
        print(f"  wrote {p} ({len(data)} rows)")

    write("exact_duplicate_clusters.csv", exact_rows)
    write("near_duplicate_candidates.csv", confirmed_rows)
    write("duplicate_clusters.csv", cluster_rows)
    write("label_conflicts.csv", conflicts)
    write("threshold_sensitivity.csv", sweep_rows)

    with (outdir / "group_assignment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "class_folder_raw", "group_id"])
        for p in paths:
            w.writerow([p, labels[idx_of[p]], group_of[p]])

    summary = {
        "images": n,
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_files": sum(len(v) for v in exact_groups.values()),
        "exact_duplicate_extra_copies": sum(len(v) - 1 for v in exact_groups.values()),
        "pairs_screened_at_loose_distance": len(cands),
        "pairs_passing_pixel_confirmation_only": sum(
            1 for c in confirmed_rows if c["passes_pixel_confirmation"] and not c["is_duplicate_for_grouping"]),
        "duplicate_pairs_used_for_grouping": n_conf,
        "cross_label_pairs_rejected_as_background_similarity": sum(
            1 for c in confirmed_rows
            if c["passes_pixel_confirmation"] and not c["is_duplicate_for_grouping"] and c["cross_label"]),
        "total_groups": len(cluster_rows),
        "multi_image_groups": multi,
        "largest_group_size": max(sizes) if sizes else 0,
        "images_in_multi_image_groups": sum(s for s in sizes if s > 1),
        "cross_label_groups": len(conflicts),
        "thresholds": {
            "phash_operative_hamming_of_256": a.phash_candidate_threshold,
            "phash_screen_hamming_of_256": a.phash_screen_threshold,
            "dhash_confirm_hamming_of_256": a.dhash_confirm_threshold,
            "ssim_confirm": a.ssim_confirm_threshold,
        },
        "group_size_histogram": {str(k): sizes.count(k) for k in sorted(set(sizes))},
        "threshold_sensitivity": sweep_rows,
    }
    (outdir / "duplicate_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "group_size_histogram"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
