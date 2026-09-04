"""Score a trained checkpoint on a frozen partition and emit every metric the paper needs.

Raw per-image predictions are always written, not just aggregate metrics. Everything
downstream -- bootstrap intervals, McNemar tests between models, calibration fitting,
selective-risk curves -- operates on those saved predictions, so no analysis ever
requires retraining and every reported number is traceable to a specific image.

Two distinct notions of uncertainty are kept separate:
  * bootstrap intervals resample IMAGES from the fixed prediction set, answering
    "how precisely does a 791-image test partition pin down this metric?"
  * seed-to-seed spread across independent training runs answers a different question,
    "how much does the training procedure itself vary?", and is aggregated elsewhere.
Reporting one as the other would misstate what was measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_ORDER = [
    "1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight", "4. Helopeltis",
    "5. Red spider", "6. Green mirid bug", "7. Healthy leaf",
]
NUM_CLASSES = len(CLASS_ORDER)


# --------------------------------------------------------------------------- #
# Predictive metrics
# --------------------------------------------------------------------------- #
def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def mcc_multiclass(cm: np.ndarray) -> float:
    """Matthews correlation coefficient, multiclass form (Gorodkin)."""
    c = float(np.trace(cm))
    s = float(cm.sum())
    t = cm.sum(axis=1).astype(np.float64)   # true totals per class
    p = cm.sum(axis=0).astype(np.float64)   # predicted totals per class
    num = c * s - float(t @ p)
    den = np.sqrt(max(s * s - float(p @ p), 0.0)) * np.sqrt(max(s * s - float(t @ t), 0.0))
    return float(num / den) if den > 0 else 0.0


def auroc_ovr(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """One-vs-rest AUROC via the rank (Mann-Whitney) identity.

    Only meaningful where a class has both positives and negatives present; classes
    that fail that condition are reported as null rather than silently scored.
    """
    out: dict[str, float | None] = {}
    vals: list[float] = []
    for c in range(NUM_CLASSES):
        pos = probs[y_true == c, c]
        neg = probs[y_true != c, c]
        if len(pos) == 0 or len(neg) == 0:
            out[CLASS_ORDER[c]] = None
            continue
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort().astype(np.float64) + 1
        # Average ranks within ties so the statistic stays unbiased.
        order = np.argsort(allv)
        sorted_v = allv[order]
        i = 0
        while i < len(sorted_v):
            j = i
            while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
            i = j + 1
        r_pos = ranks[:len(pos)].sum()
        a = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        out[CLASS_ORDER[c]] = float(a)
        vals.append(float(a))
    out["macro"] = float(np.mean(vals)) if vals else None
    return out


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    eps = 1e-12
    n = len(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    nll = float(-np.mean(np.log(np.clip(probs[np.arange(n), y_true], eps, 1.0))))

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    # Equal-width binning on top-1 confidence (Naeini/Guo formulation).
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = float(conf[m].mean())
        acc = float(correct[m].mean())
        gap = abs(avg_conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "n": cnt,
                     "avg_confidence": avg_conf, "accuracy": acc})

    return {
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        "overconfidence_gap": float(conf.mean() - correct.mean()),
        "reliability_bins": bins,
    }


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap over images, holding the trained model fixed."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_boot)
    accs = np.empty(n_boot)
    bals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        cm = confusion(yt, yp)
        _, rec, f1 = per_class_prf(cm)
        present = cm.sum(axis=1) > 0
        f1s[b] = f1[present].mean() if present.any() else 0.0
        accs[b] = float((yt == yp).mean())
        bals[b] = rec[present].mean() if present.any() else 0.0

    def ci(v: np.ndarray) -> dict:
        return {"mean": float(v.mean()),
                "ci95_low": float(np.percentile(v, 2.5)),
                "ci95_high": float(np.percentile(v, 97.5))}

    return {"n_bootstrap": n_boot, "macro_f1": ci(f1s),
            "accuracy": ci(accs), "balanced_accuracy": ci(bals)}


# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, probs: np.ndarray, n_boot: int, seed: int) -> dict:
    y_pred = probs.argmax(axis=1)
    cm = confusion(y_true, y_pred)
    prec, rec, f1 = per_class_prf(cm)
    present = cm.sum(axis=1) > 0

    return {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1[present].mean()),
        "macro_precision": float(prec[present].mean()),
        "macro_recall": float(rec[present].mean()),
        "balanced_accuracy": float(rec[present].mean()),
        "mcc": mcc_multiclass(cm),
        "per_class": {
            CLASS_ORDER[c]: {
                "support": int(cm[c].sum()),
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1": float(f1[c]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_ORDER,
        "auroc_ovr": auroc_ovr(y_true, probs),
        "calibration": calibration_metrics(y_true, probs),
        "bootstrap": bootstrap_ci(y_true, y_pred, n_boot, seed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--partitions", default="validation,test")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)

    model = tf.keras.models.load_model(a.checkpoint, compile=False)
    results: dict = {
        "checkpoint": str(a.checkpoint),
        "split_prefix": a.split_prefix,
        "total_params": int(model.count_params()),
        "partitions": {},
    }

    for part in a.partitions.split(","):
        x = np.load(cache / f"{a.split_prefix}_{part}_x.npy").astype("float32")
        y = np.load(cache / f"{a.split_prefix}_{part}_y.npy")
        probs = model.predict(x, batch_size=a.batch_size, verbose=0).astype(np.float64)
        # Guard against a non-normalized head silently corrupting the metrics.
        probs = probs / probs.sum(axis=1, keepdims=True)

        results["partitions"][part] = evaluate(y, probs, a.n_bootstrap, a.bootstrap_seed)

        # Raw per-image predictions: the substrate for every downstream analysis.
        np.savez_compressed(outdir / f"predictions_{a.split_prefix}_{part}.npz",
                            y_true=y, probs=probs.astype(np.float32))

        r = results["partitions"][part]
        c = r["calibration"]
        print(f"\n=== {a.split_prefix}/{part}  (n={r['n']}) ===")
        print(f"  macro-F1        {r['macro_f1']:.4f}   "
              f"[{r['bootstrap']['macro_f1']['ci95_low']:.4f}, "
              f"{r['bootstrap']['macro_f1']['ci95_high']:.4f}]")
        print(f"  accuracy        {r['accuracy']:.4f}")
        print(f"  balanced acc    {r['balanced_accuracy']:.4f}")
        print(f"  MCC             {r['mcc']:.4f}")
        print(f"  AUROC (macro)   {r['auroc_ovr']['macro']:.4f}")
        print(f"  ECE             {c['ece']:.4f}   NLL {c['nll']:.4f}   Brier {c['brier']:.4f}")
        print(f"  mean conf {c['mean_confidence']:.4f} vs acc {c['accuracy']:.4f}"
              f"  (gap {c['overconfidence_gap']:+.4f})")
        print("  per-class F1:")
        for cls, m in r["per_class"].items():
            print(f"    {cls:<24} n={m['support']:<4} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f}")

    (outdir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {outdir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
