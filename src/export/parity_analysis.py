"""Paired statistical comparison of every exported TFLite variant against the Keras reference.

Reviewer 2 observed that dynamic-range INT8 reported a higher macro-F1 (0.8827) than the
FP32 reference (0.8745) despite only 97.2% top-1 agreement, and asked for evidence that
the difference is noise rather than an improvement. Aggregate metrics cannot settle that:
both systems were scored on the SAME 791 images, so the comparison has to be paired.

Three paired quantities are computed on the shared test partition:

  * a paired percentile bootstrap of the macro-F1 DIFFERENCE (variant minus reference),
    where each resample draws one index vector and applies it to both prediction sets;
  * McNemar's exact test on the discordant pairs;
  * per-class recall, F1 and the full confusion matrix, which is what identifies the
    class whose recall collapses under full-integer quantization (Reviewer 1, point 4).

Per-image predictions of every variant are saved so no later analysis needs to re-run an
interpreter, and every .tflite is hashed against export_report.json before use, so the
analysis provably concerns the artifacts that were benchmarked on the handset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))
from export_tflite import CLASS_ORDER, run_tflite  # noqa: E402
from compare_models import mcnemar  # noqa: E402

K = len(CLASS_ORDER)


def confusion_fast(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(K), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(K), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = confusion_fast(y, p)
    present = cm.sum(1) > 0
    return float(prf(cm)[2][present].mean())


def paired_bootstrap(y, pred_a, pred_b, n_boot: int, seed: int) -> dict:
    """Difference (A minus B) resampled over images with ONE index vector per replicate."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_f1 = np.empty(n_boot)
    d_acc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        d_f1[b] = macro_f1(yt, pred_a[idx]) - macro_f1(yt, pred_b[idx])
        d_acc[b] = float((pred_a[idx] == yt).mean() - (pred_b[idx] == yt).mean())

    def summ(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return {"mean_points": round(100 * float(v.mean()), 3),
                "ci95_low_points": round(100 * float(lo), 3),
                "ci95_high_points": round(100 * float(hi), 3),
                "ci_includes_zero": bool(lo <= 0.0 <= hi),
                "fraction_positive": float((v > 0).mean())}

    return {"n_bootstrap": n_boot, "seed": seed,
            "delta_macro_f1": summ(d_f1), "delta_accuracy": summ(d_acc),
            "note": "A single index vector is drawn per replicate and applied to both systems, "
                    "so the interval describes the paired difference on shared images."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exported-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--eval-dir", required=True, help="holds the Keras predictions_*_test.npz")
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    a = ap.parse_args()

    exported, outdir = Path(a.exported_dir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = json.loads((exported / "export_report.json").read_text(encoding="utf-8"))

    xt = np.load(Path(a.cache_dir) / f"{a.split_prefix}_test_x.npy")
    yt = np.load(Path(a.cache_dir) / f"{a.split_prefix}_test_y.npy")
    ref_z = np.load(Path(a.eval_dir) / f"predictions_{a.split_prefix}_test.npz")
    if not np.array_equal(ref_z["y_true"], yt):
        raise SystemExit("reference predictions are not aligned with the test cache; refusing to pair")
    ref = ref_z["probs"].astype(np.float64)
    ref_pred = ref.argmax(1)
    ref_cm = confusion_fast(yt, ref_pred)
    _, ref_rec, ref_f1 = prf(ref_cm)
    ref_macro = macro_f1(yt, ref_pred)
    print(f"Keras reference: macro-F1 {ref_macro:.4f}  acc {(ref_pred == yt).mean():.4f}")

    out: dict = {
        "reference": {
            "source": str(Path(a.eval_dir) / f"predictions_{a.split_prefix}_test.npz"),
            "macro_f1": ref_macro,
            "accuracy": float((ref_pred == yt).mean()),
            "per_class_recall": {c: float(ref_rec[i]) for i, c in enumerate(CLASS_ORDER)},
            "per_class_f1": {c: float(ref_f1[i]) for i, c in enumerate(CLASS_ORDER)},
            "confusion_matrix": ref_cm.tolist(),
        },
        "class_order": CLASS_ORDER,
        "n_test": int(len(yt)),
        "variants": {},
    }

    for name, v in report["variants"].items():
        if not v.get("export_ok"):
            continue
        path = exported / v["file"]
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        if sha != v["sha256"]:
            raise SystemExit(f"{path.name}: digest {sha[:12]} differs from export_report ({v['sha256'][:12]})")
        print(f"\n--- {name} ({path.name}, sha256 verified) ---", flush=True)

        probs, meta = run_tflite(blob, xt)
        np.savez_compressed(outdir / f"predictions_{name}.npz", y_true=yt, probs=probs.astype(np.float32))
        pred = probs.argmax(1)
        cm = confusion_fast(yt, pred)
        prec, rec, f1 = prf(cm)
        mf1 = macro_f1(yt, pred)

        boot = paired_bootstrap(yt, pred, ref_pred, a.n_bootstrap, a.bootstrap_seed)
        mcn = mcnemar(pred == yt, ref_pred == yt)

        # Where do the disagreements go? Counted as (reference class -> variant class).
        dis: dict[str, int] = {}
        for r_, p_ in zip(ref_pred[pred != ref_pred], pred[pred != ref_pred]):
            key = f"{CLASS_ORDER[r_]} -> {CLASS_ORDER[p_]}"
            dis[key] = dis.get(key, 0) + 1
        dis = dict(sorted(dis.items(), key=lambda kv: -kv[1]))

        recall_drop = {c: round(100 * (ref_rec[i] - rec[i]), 3) for i, c in enumerate(CLASS_ORDER)}
        worst_i = int(np.argmax([recall_drop[c] for c in CLASS_ORDER]))
        worst = CLASS_ORDER[worst_i]
        row = cm[worst_i]
        # Misclassification destinations only (largest first); the diagonal is reported apart
        # as `retained`, so the first entry is always where the lost images went.
        destinations = {CLASS_ORDER[j]: int(row[j]) for j in np.argsort(-row)
                        if row[j] > 0 and j != worst_i}

        out["variants"][name] = {
            "file": path.name, "sha256": sha, "mb": v["mb"],
            "interpreter": {k: meta.get(k) for k in ("runtime", "delegate", "input_dtype", "output_dtype")},
            "macro_f1": mf1,
            "accuracy": float((pred == yt).mean()),
            "delta_macro_f1_points_vs_reference": round(100 * (mf1 - ref_macro), 3),
            "top1_agreement_with_reference": float((pred == ref_pred).mean()),
            "n_disagreements": int((pred != ref_pred).sum()),
            "disagreement_transitions": dis,
            "per_class": {c: {"support": int(cm[i].sum()), "precision": float(prec[i]),
                              "recall": float(rec[i]), "f1": float(f1[i]),
                              "recall_drop_points_vs_reference": recall_drop[c]}
                          for i, c in enumerate(CLASS_ORDER)},
            "confusion_matrix": cm.tolist(),
            "paired_bootstrap": boot,
            "mcnemar_vs_reference": {k: mcn[k] for k in
                                     ("a_right_b_wrong", "a_wrong_b_right", "discordant",
                                      "p_value_exact_binomial")},
            "worst_class": {"name": worst, "reference_recall": float(ref_rec[worst_i]),
                            "variant_recall": float(rec[worst_i]),
                            "recall_drop_points": recall_drop[worst],
                            "support": int(row.sum()), "retained": int(row[worst_i]),
                            "where_its_images_went": destinations},
        }
        d = boot["delta_macro_f1"]
        print(f"  macro-F1 {mf1:.4f} ({100*(mf1-ref_macro):+.2f} pts)  agreement {(pred==ref_pred).mean():.4f}")
        print(f"  paired bootstrap dF1 {d['mean_points']:+.2f} pts, 95% CI [{d['ci95_low_points']:+.2f}, "
              f"{d['ci95_high_points']:+.2f}], includes zero: {d['ci_includes_zero']}")
        print(f"  McNemar b={mcn['a_right_b_wrong']} c={mcn['a_wrong_b_right']} p={mcn['p_value_exact_binomial']:.3g}")
        print(f"  worst class: {worst}  recall {ref_rec[worst_i]:.3f} -> {rec[worst_i]:.3f}  -> {destinations}")

    (outdir / "parity_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {outdir/'parity_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
