# G3 — Leakage-Aware Split Construction

**Status: PASS**
**Executed:** 2026-07-20 · **Seed:** 42 · **Source:** teaLeafBD v3 (`10.17632/744vznw5k2.3`)

Both splits were built **before any augmentation**. Augmentation is applied only to the training partition,
at load time, and never touches validation or test images.

---

## 1. Two splits, two purposes

**Primary (leakage-aware)** — the split every headline result is reported on. The unit of assignment is the
duplicate group, not the image, so no cluster can straddle a partition. Groups are ordered by decreasing
size and each is placed in whichever partition is furthest below its quota for that group's dominant class;
ties break on a seeded permutation, making the split reproducible from the manifest alone.

**Naive (control)** — image-level stratified random assignment that ignores duplicate structure. It exists
solely to quantify how much a conventional split inflates results (RQ1) and is never used as a performance
claim.

---

## 2. Achieved partitions

### Primary

| Partition | Images | Fraction | Groups |
|---|---:|---:|---:|
| Train | 3,694 | 0.7002 | 3,651 |
| Validation | 791 | 0.1499 | 791 |
| Test | 791 | 0.1499 | 791 |
| **Total** | **5,276** | 1.0000 | 5,233 |

Group-level assignment normally forces visible deviation from a 70/15/15 target. It does not here, because
every duplicate group contains at most two images — the achieved fractions land within 0.02 percentage
points of target. All 43 multi-image groups fall in the training partition, which is why train holds 3,694
images across only 3,651 groups.

### Naive control

| Partition | Images | Fraction |
|---|---:|---:|
| Train | 3,692 | 0.6998 |
| Validation | 791 | 0.1499 |
| Test | 793 | 0.1503 |

---

## 3. Quarantine

**Zero images quarantined.**

| Candidate reason | Count |
|---|---:|
| Undecodable or corrupt | 0 |
| Cross-label duplicate cluster (unadjudicated) | 0 |

No decode failures exist, and after correcting the duplicate definition (see G2 §5.2) no cross-label cluster
survives at the operative threshold. All 5,276 images are eligible, so no data is discarded and no
adjudication is needed.

---

## 4. Automated integrity tests

`tests/test_split_integrity.py` — **12 passed**.

| Test | Result |
|---|---|
| Partitions are disjoint | PASS |
| Partitions cover all eligible images | PASS |
| All seven classes present in every partition | PASS |
| **No duplicate group spans partitions** | PASS |
| **No identical SHA-256 appears in two partitions** | PASS |
| **No confirmed near-duplicate pair crosses partitions** | PASS |
| Proportions within 2 points of target | PASS |
| Class stratification preserved | PASS |
| Naive split disjoint and complete | PASS |
| Naive split carries real contamination | PASS |
| Split files match recorded SHA-256 | PASS |
| No augmented or derived images referenced | PASS |

### 4.1 A test failure that changed the analysis

The near-duplicate crossing test initially **failed with 52 violations**. The failure was not relaxed away.
Inspection showed every violation was cross-class, and rendered contact sheets showed the pairs were
different leaves whose similarity came from the shared studio backdrop rather than the leaf.

The defect was in the duplicate *definition*, not the split: whole-frame SSIM is a weak confirmer on a
dataset where the background is near-constant. Grouping now additionally requires perceptual-hash agreement.
The test was left strict and now passes on its original terms. Full reasoning in G2 §5.2.

---

## 5. Measured contamination in the control split

The naive split carries **16 of the 43 confirmed duplicate pairs across partition boundaries**, versus zero
in the primary split. That is the entire contamination available to inflate a naive result.

This number sets the ceiling for RQ1 in advance. With a 791-image test partition, at most ~16 test images
can have a twin elsewhere, so the maximum arithmetically possible inflation is on the order of **1–2
percentage points**. A large leakage effect is therefore not available to be found, and reporting one would
indicate a bug rather than a discovery.

Stating this before training matters: the leakage-sensitivity experiment is now a measurement with a known
bound, not an open-ended search for a favorable result.

---

## 6. Immutability

Every split manifest is hashed at write time and the digests are stored in `split_summary.json`; a test
re-verifies them. Any later edit to a split file fails the suite.

| File | SHA-256 (first 16) |
|---|---|
| `primary_train.csv` | recorded in `split_summary.json` |
| `primary_validation.csv` | recorded |
| `primary_test.csv` | recorded |
| `naive_{train,validation,test}.csv` | recorded |
| `quarantine_manifest.csv` | recorded |

**The test partition is not inspected until the training protocol is frozen.** Validation is used for
architecture choice, hyperparameters, early stopping, threshold selection, temperature scaling, and
quantization decisions.

**G3: PASS.**
