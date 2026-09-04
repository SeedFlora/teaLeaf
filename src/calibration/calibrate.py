"""Post-hoc temperature scaling and risk-coverage selection of an abstention threshold.

Both are fitted on the VALIDATION partition only. The test partition is scored once
afterwards with the frozen temperature and threshold; tuning either on test would turn
the held-out set into a second validation set and invalidate every headline number.

Two implementation points that are easy to get wrong:

* Temperature scaling operates on LOGITS. Dividing probabilities by T and renormalizing
  is a different, incorrect operation. Logits are reconstructed as log(p), which is exact
  up to an additive constant -- and softmax is invariant to that constant, so nothing is
  lost.
* The abstention threshold is derived from a stated risk target, not chosen as a round
  number. A policy fixes the acceptable selective error rate; the threshold is then the
  lowest one meeting it, which keeps coverage as high as the policy allows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASS_ORDER = [
    "1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight", "4. Helopeltis",
    "5. Red spider", "6. Green mirid bug", "7. Healthy leaf",
]
EPS = 1e-12


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    """Recover logits up to an additive constant. Softmax ignores that constant."""
    return np.log(np.clip(p, EPS, 1.0))


def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Minimize validation NLL over T by golden-section search on a log grid.

    NLL as a function of T is smooth and unimodal for a fixed model, so a bracketed
    scalar search is sufficient and avoids a gradient dependency.
    """
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 400))
    losses = np.array([nll(softmax_T(logits, float(t)), y) for t in grid])
    i = int(losses.argmin())

    lo = float(grid[max(0, i - 1)])
    hi = float(grid[min(len(grid) - 1, i + 1)])
    phi = (np.sqrt(5.0) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(80):
        if nll(softmax_T(logits, c), y) < nll(softmax_T(logits, d), y):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    T = float((a + b) / 2)

    return T, {"grid_best_T": float(grid[i]), "grid_best_nll": float(losses[i]),
               "refined_T": T, "refined_nll": nll(softmax_T(logits, T), y)}


def risk_coverage(probs: np.ndarray, y: np.ndarray, n_points: int = 200) -> list[dict]:
    """Selective risk and coverage as the confidence threshold sweeps."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y)
    curve = []
    for t in np.linspace(0.0, 1.0, n_points):
        keep = conf >= t
        n_keep = int(keep.sum())
        cov = n_keep / len(y)
        if n_keep == 0:
            curve.append({"threshold": float(t), "coverage": 0.0,
                          "selective_accuracy": None, "selective_risk": None, "n_kept": 0})
            continue
        sel_acc = float(correct[keep].mean())
        curve.append({"threshold": float(t), "coverage": float(cov),
                      "selective_accuracy": sel_acc, "selective_risk": float(1 - sel_acc),
                      "n_kept": n_keep})
    return curve


def pick_threshold(curve: list[dict], target_risk: float, min_coverage: float) -> dict:
    """Lowest threshold meeting the risk target while retaining minimum coverage.

    Choosing the LOWEST qualifying threshold maximizes coverage subject to the policy,
    so the model abstains as rarely as the stated risk tolerance permits.
    """
    feasible = [p for p in curve
                if p["selective_risk"] is not None
                and p["selective_risk"] <= target_risk
                and p["coverage"] >= min_coverage]
    if feasible:
        chosen = min(feasible, key=lambda p: p["threshold"])
        return {"threshold": chosen["threshold"], "basis": "risk target met",
                "coverage": chosen["coverage"], "selective_risk": chosen["selective_risk"],
                "selective_accuracy": chosen["selective_accuracy"]}

    # Policy unreachable: fall back to the best achievable risk at the coverage floor,
    # and say so rather than silently relaxing the target.
    at_floor = [p for p in curve if p["coverage"] >= min_coverage and p["selective_risk"] is not None]
    if not at_floor:
        return {"threshold": 0.0, "basis": "no feasible point; abstention disabled",
                "coverage": 1.0, "selective_risk": None, "selective_accuracy": None}
    chosen = min(at_floor, key=lambda p: p["selective_risk"])
    return {"threshold": chosen["threshold"],
            "basis": (f"risk target {target_risk} unreachable at coverage >= {min_coverage}; "
                      f"selected the minimum-risk point satisfying the coverage floor"),
            "coverage": chosen["coverage"], "selective_risk": chosen["selective_risk"],
            "selective_accuracy": chosen["selective_accuracy"]}


def summarize(probs: np.ndarray, y: np.ndarray, tag: str) -> dict:
    return {"tag": tag, "nll": nll(probs, y), "ece": ece(probs, y), "brier": brier(probs, y),
            "accuracy": float((probs.argmax(axis=1) == y).mean()),
            "mean_confidence": float(probs.max(axis=1).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="directory holding predictions_*.npz")
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target-selective-risk", type=float, default=0.05,
                    help="maximum acceptable error rate among non-abstained predictions")
    ap.add_argument("--min-coverage", type=float, default=0.70,
                    help="minimum fraction of inputs the system must still answer")
    a = ap.parse_args()

    ed, outdir = Path(a.eval_dir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    val = np.load(ed / f"predictions_{a.split_prefix}_validation.npz")
    test = np.load(ed / f"predictions_{a.split_prefix}_test.npz")
    pv, yv = val["probs"].astype(np.float64), val["y_true"]
    pt, yt = test["probs"].astype(np.float64), test["y_true"]

    lv, lt = probs_to_logits(pv), probs_to_logits(pt)

    T, fit = fit_temperature(lv, yv)
    pv_cal, pt_cal = softmax_T(lv, T), softmax_T(lt, T)

    curve_val = risk_coverage(pv_cal, yv)
    policy = pick_threshold(curve_val, a.target_selective_risk, a.min_coverage)
    thr = policy["threshold"]

    # Apply the frozen threshold to test exactly once.
    conf_t = pt_cal.max(axis=1)
    keep = conf_t >= thr
    correct_t = (pt_cal.argmax(axis=1) == yt)
    n_keep = int(keep.sum())

    test_selective = {
        "threshold_applied": thr,
        "coverage": float(n_keep / len(yt)),
        "n_answered": n_keep,
        "n_abstained": int(len(yt) - n_keep),
        "selective_accuracy": float(correct_t[keep].mean()) if n_keep else None,
        "selective_risk": float(1 - correct_t[keep].mean()) if n_keep else None,
        "full_coverage_accuracy": float(correct_t.mean()),
        "accuracy_on_abstained": float(correct_t[~keep].mean()) if n_keep < len(yt) else None,
    }

    result = {
        "temperature": T,
        "temperature_fit": fit,
        "fitted_on": "validation partition only",
        "validation_uncalibrated": summarize(pv, yv, "validation uncalibrated"),
        "validation_calibrated": summarize(pv_cal, yv, "validation calibrated"),
        "test_uncalibrated": summarize(pt, yt, "test uncalibrated"),
        "test_calibrated": summarize(pt_cal, yt, "test calibrated"),
        "abstention_policy": {
            "target_selective_risk": a.target_selective_risk,
            "min_coverage": a.min_coverage,
            **policy,
            "selected_on": "validation risk-coverage curve",
        },
        "test_selective_performance": test_selective,
        "risk_coverage_validation": curve_val,
        "risk_coverage_test": risk_coverage(pt_cal, yt),
        "caveats": [
            "Temperature and threshold are fitted on validation and applied unchanged to "
            "test. Neither was tuned on test.",
            "This is a confidence-based abstention rule, not an out-of-distribution "
            "detector: it was not evaluated against OOD inputs and must not be described "
            "as one.",
        ],
    }

    (outdir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outdir / "calibration_params.json").write_text(json.dumps({
        "temperature": T, "abstention_threshold": thr, "class_order": CLASS_ORDER,
        "target_selective_risk": a.target_selective_risk, "min_coverage": a.min_coverage,
    }, indent=2), encoding="utf-8")

    vu, vc = result["validation_uncalibrated"], result["validation_calibrated"]
    tu, tc = result["test_uncalibrated"], result["test_calibrated"]
    print(f"fitted temperature T = {T:.4f}\n")
    print(f"{'':<14}{'NLL':>10}{'ECE':>10}{'Brier':>10}{'meanConf':>10}{'acc':>9}")
    for r in (vu, vc, tu, tc):
        print(f"{r['tag']:<14}{r['nll']:>10.4f}{r['ece']:>10.4f}{r['brier']:>10.4f}"
              f"{r['mean_confidence']:>10.4f}{r['accuracy']:>9.4f}")
    print(f"\nECE change on test: {tc['ece'] - tu['ece']:+.4f}   "
          f"NLL change: {tc['nll'] - tu['nll']:+.4f}")
    print(f"\nabstention threshold {thr:.4f}  ({policy['basis']})")
    print(f"  validation: coverage {policy['coverage']:.4f}  risk {policy['selective_risk']}")
    ts = test_selective
    print(f"  test      : coverage {ts['coverage']:.4f}  "
          f"selective accuracy {ts['selective_accuracy']}  "
          f"vs {ts['full_coverage_accuracy']:.4f} at full coverage")
    print(f"  abstained on {ts['n_abstained']} of {len(yt)}; accuracy on those "
          f"would have been {ts['accuracy_on_abstained']}")
    print(f"\nwritten: {outdir/'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
