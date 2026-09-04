# G2 — Dataset Acquisition, Version Audit, and Forensic Integrity Audit

**Status: PASS**
**Canonical dataset decision: `USE_V3`**
**Executed:** 2026-07-20

---

## 1. Acquisition and provenance

The teaLeafBD archive was supplied locally rather than downloaded in this session, so its provenance could
not be assumed. It was verified cryptographically against the official repository.

| Property | Value |
|---|---|
| Local archive | `d:\tea\teaLeafBD.zip`, 1,334,923,045 bytes |
| Archive SHA-256 | `1d40a9a5db083181220bad5a03c5336a38ae9d6a613f8c3677308cad565fbef0` |
| Entries | 5,276 — all JPG, no README |
| Internal layout | `teaLeafBD/teaLeafBD/<class folder>/` |
| Extracted to | WSL2 ext4 working area (source archive never modified) |

### 1.1 Official metadata (Mendeley Data public API)

| | Version 3 | Version 4 |
|---|---|---|
| DOI | `10.17632/744vznw5k2.3` | `10.17632/744vznw5k2.4` |
| Published | 2025-01-27 | 2025-04-14 |
| Stated images | 5,276 | 5,278 |
| README | none | `read_me.pdf` |
| License | CC BY 4.0 | CC BY 4.0 |

Four versions exist (v1 2024-10-30, v2 2025-01-22, v3, v4). Contributors are identical across versions.

### 1.2 Provenance verification result

The Mendeley API publishes a repository-side SHA-256 for every file, which allows a file-level comparison
without downloading gigabytes of imagery.

| Check | Result |
|---|---|
| Local files matching official v3 by **path and SHA-256** | **4,981 / 4,981** |
| Hash mismatches | **0** |
| Official v3 files absent locally | **0** |
| Local files not in the official listing | 295 |

The 295 apparently unmatched files are **not** discrepancies. The API caps a folder listing at 1,000 entries
and ignored every pagination parameter tested (`limit`, `offset`, `page`, `marker`). The truncation affects
exactly two classes, and the shortfall matches precisely: Gray Blight 1,013 − 1,000 = 13, Green mirid bug
1,282 − 1,000 = 282, total 295.

**Conclusion: the local archive is authentic Mendeley teaLeafBD version 3**, cryptographically verified on
94.4% of files with zero contradictions, and count-reconciled on the remainder.

---

## 2. The 5,276 versus 5,278 discrepancy — resolved

This question is answered from the official manifests, not inferred.

**Version 4 = version 3 plus exactly three files.**

| Status | Count | Files |
|---|---:|---|
| Only in v4 | 3 | `2. Brown Blight/brown_blight_00507.jpg`, `2. Brown Blight/brown_blight_00508.jpg`, `read_me.pdf` |
| Only in v3 | 0 | — |
| Shared paths with changed content | 0 | — |
| Shared paths byte-identical | 4,981 | — |

Corroborating evidence:

- The official v4 `read_me.pdf` (SHA-256 verified against the API) gives a per-class table whose only
  divergence from the local v3 counts is Brown Blight: 508 versus 506. All six other classes match exactly.
- Local Brown Blight filenames run **contiguously 1–506** with no gaps, so v3 is not "v4 minus two arbitrary
  images." Version 4 **appended** two images.
- The two added files were downloaded (hashes match the API exactly): 193,136 and 186,516 bytes, both
  1200×1594 px, both created 2025-04-11 — three days before the v4 release.
- Folder names are identical across both versions. **No label changed.**

**Answers to the required questions:** v3 reports 5,276 because Brown Blight held 506 images; v4 and the
data article report 5,278 because two Brown Blight images were added; the added files are valid JPEGs;
labels did not change; **version 4 is a purely additive correction plus documentation, not a materially
different release.**

---

## 3. Canonical version decision: `USE_V3`

Version 4 is demonstrably the corrected, complete release, which under the project rules would favor it.
It is nonetheless **not** used, for a reason that overrides that preference:

1. **The complete v4 archive could not be obtained.** The S3 bulk-download cache returned HTTP 403 to both
   HEAD and ranged GET, and the API listing is truncated at 1,000 files per folder. Only the two delta files
   could be retrieved individually.
2. **Reconstructing v4 would mean combining files across releases**, which the project rules forbid. Even
   though the shared files are provably byte-identical, the result would be an unciteable hybrid rather than
   a release with a DOI.
3. **The rules explicitly cover this case:** if version 4 cannot be downloaded, use version 3 and document
   that the comparison was blocked. Here only the *download* was blocked — the *comparison* succeeded
   through official hash manifests, and is reported in full above.
4. Version 3 was explicitly requested and is a single immutable, hash-verified, citable release.

**Cost of this decision:** two Brown Blight images, 0.04% of the dataset, confined to one class, fully
identified by filename. This is stated in the manuscript rather than hidden.

The two v4 files are retained under `data/raw/v4/` as **audit evidence only** and are never read by any
training, validation, or test pipeline.

**Citation used:** Alam BMS, et al. teaLeafBD. Version 3. Mendeley Data; 2025. doi:10.17632/744vznw5k2.3

---

## 4. Image integrity

| Check | Result |
|---|---|
| Images scanned | 5,276 |
| Decode failures | **0** |
| Structural corruption | **0** |
| Format | JPEG, 100% |
| Color mode | RGB, 100% |
| Files quarantined for corruption | 0 |

No file was deleted or modified.

---

## 5. Duplicate audit

### 5.1 Method

Three stages, kept deliberately distinct so "candidate" and "duplicate" are never conflated:

1. **Exact** — identical SHA-256.
2. **Screening** — all pairs within pHash Hamming distance 64/256 (recall-oriented, 3,872 pairs).
3. **Confirmation** — independent dHash agreement plus pixel-level SSIM.

Clusters are connected components of the confirmed-duplicate graph, so transitive chains cannot leak.

### 5.2 A correction made during the audit

Whole-frame SSIM initially admitted 141 pairs, of which 51 were **cross-class**. Visual review of the
rendered contact sheets showed these are **different leaves** — the high SSIM comes from the shared studio
backdrop, which is near-constant across the dataset while the leaf occupies only a small fraction of the
frame.

Grouping was therefore tightened to additionally require perceptual-hash agreement (pHash ≤ 32), which is
sensitive to leaf structure rather than background. All 51 cross-class pairs sit at pHash 42–64 with SSIM
0.901–0.950, and **none** falls within the operative threshold. Evidence: `figures/cross_class_review/`.

### 5.3 Results

| Measure | Value |
|---|---:|
| Exact duplicate pairs (byte-identical) | **1** |
| Pairs passing pixel confirmation only (background similarity, not grouped) | 98 |
| **Duplicate pairs used for grouping** | **43** |
| Cross-class pairs rejected as background similarity | 51 |
| Images in multi-image groups | 86 (**1.63%**) |
| Total groups | 5,233 |
| Largest group size | **2** |
| Cross-label conflict groups | **0** |

The exact duplicate is `3. Gray Blight/gray_blight_00493.jpg` ≡ `gray_blight_00494.jpg`, SHA-256
`48f88f50…`, same class.

### 5.4 Threshold sensitivity

Because teaLeafBD ships no leaf ID, session ID, or timestamp, the grouping derives entirely from a chosen
threshold. The full curve is therefore reported rather than a single operating point
(`data/audit/threshold_sensitivity.csv`).

| pHash threshold | Pairs | Images grouped | % of dataset | Largest group |
|---:|---:|---:|---:|---:|
| 0 | 1 | 2 | 0.04% | 2 |
| 10 | 15 | 30 | 0.57% | 2 |
| 20 | 33 | 66 | 1.25% | 2 |
| **30–34** | **43** | **86** | **1.63%** | **2** |
| 44 | 56 | 110 | 2.09% | 3 |
| 54 | 90 | 158 | 3.00% | 6 |
| 64 | 141 | 223 | 4.23% | 12 |

**The operative threshold sits on a plateau**: thresholds 30, 32, and 34 all yield exactly 43 pairs and
1.63% of images. The result is not balanced on a cliff edge.

### 5.5 Threshold-independent corroboration: sequence adjacency

A perceptual hash cannot observe filename order, so sequence adjacency is an independent check.

| Measure (operative threshold) | Value |
|---|---:|
| Confirmed pairs | 43 |
| Same-class / cross-class | **43 / 0** |
| Sequence gap = 1 (consecutive files) | **41 (95.35%)** |
| Sequence gap ≤ 5 | **43 (100%)** |
| Expected adjacent under random placement | 0.127 |
| **Observed / expected** | **323.5×** |

Forty-one of forty-three visually confirmed pairs land on consecutive file numbers. Consecutive numbering
indicates consecutive capture. **These are repeat photographs of the same leaf.**

### 5.6 Bearing on the dataset article's claim

The Data in Brief article states: *"Every image is of a distinct tea leaf. No leaf has been captured twice."*
It is presented as a collection protocol; the article describes no computational verification.

The evidence contradicts it: one byte-identical duplicate pair, plus 43 near-duplicate pairs of which 95%
occupy consecutive sequence positions. **The claim does not hold exactly.**

Proportion matters as much as direction: the affected images are **1.63% of the dataset**, and no cluster
exceeds two images. teaLeafBD is substantially cleaner than the cautionary datasets in the leakage
literature. The finding is a correction to a specific published claim, not a repudiation of the dataset.

---

## 6. Bias audit

| Dimension | Finding |
|---|---|
| Class imbalance | **3.07×** (Green mirid bug 1,282 vs Tea algal leaf spot 418) |
| Resolution | **4 distinct sizes**: 1200×1600 (2,642), 1200×1594 (1,188), 1200×900 (734), 1200×904 (712). The README's claim of a uniform 1200×1600 is inaccurate. |
| Resolution as a shortcut | All four resolutions appear in every class (40–58% at the dominant size), so resolution does not map cleanly onto a label |
| EXIF coverage | **100%** of images |
| Capture devices | realme 6i (1,833), OPPO Reno8 Pro 5G (1,543), Redmi `23053RN02A` (1,279), narzo 50 (621) — empirically confirming the article's stated four devices |
| **Device-label correlation** | **All four devices contribute to all seven classes.** Shares vary (narzo 50 spans 5.0–30.0%) but no class is device-exclusive, so the capture device is a weak cue at most |
| EXIF orientation | **All 5,276 images are orientation 1** — no rotation handling is required, which removes a common desktop/mobile parity hazard |
| Encoding | JPEG/RGB throughout; no screenshots or non-photographic files detected |
| Background | **Every image is a single detached leaf on a near-uniform studio backdrop.** This is a controlled studio dataset, not field imagery — verified visually, not assumed |

The studio-background finding is the most consequential for deployment. A model trained here has never seen
a leaf on a branch against foliage, which is exactly what a phone camera will encounter in a garden. This
grounds the external-validity limitation in direct observation rather than in a generic citation.

A caveat on the blur statistic: the Laplacian-variance figures are computed on 256×256 thumbnails, so the
absolute values and the "blurry/soft/sharp" labels are not calibrated against any external standard. Only
relative comparisons between classes are meaningful, and on that basis Healthy leaf (median 49.2) and
Helopeltis (58.6) are softer than Brown Blight (139.9).

---

## 7. Evidence index

| Artifact | Path |
|---|---|
| Forensic per-image manifest (36 fields × 5,276) | `data/manifests/v3_manifest.csv` |
| Official v3 / v4 manifests | `data/manifests/mendeley_official_manifest_v{3,4}.csv` |
| Version diff | `data/audit/version_diff.csv`, `version_diff_summary.json` |
| Provenance verification | `data/audit/local_vs_official_verification.json` |
| Numbering-gap analysis | `data/audit/v3_numbering_gaps.json` |
| Exact duplicates | `data/audit/exact_duplicate_clusters.csv` |
| All screened pairs with scores | `data/audit/near_duplicate_candidates.csv` |
| Duplicate clusters and group ids | `data/audit/duplicate_clusters.csv`, `group_assignment.csv` |
| Label conflicts | `data/audit/label_conflicts.csv` (empty — none found) |
| Threshold sensitivity | `data/audit/threshold_sensitivity.csv` |
| Sequence-adjacency analysis | `data/audit/sequence_adjacency_analysis.json` |
| Class distribution, quality, bias | `data/audit/class_distribution.csv`, `image_quality.csv`, `bias_audit_summary.json` |
| Contact sheets (duplicates) | `figures/contact_sheets/` |
| Contact sheets (rejected cross-class) | `figures/cross_class_review/` |
| v4 delta files | `data/raw/v4/` |

**G2: PASS. Canonical decision: `USE_V3`.**
