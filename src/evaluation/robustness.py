"""Evaluate the frozen test partition under deterministic capture degradations.

Every corruption is applied to the ALREADY-SPLIT test images only. None is ever used
for training, and the severity grid is fixed in code so that a rerun reproduces the
same corrupted pixels rather than a fresh random draw.

The corruptions model failure modes a phone camera in a tea garden actually produces --
poor light, motion during the shutter, aggressive JPEG on a cheap pipeline, an
imperfectly framed or tilted leaf -- rather than the synthetic noise families used in
generic robustness benchmarks. What matters for deployment is whether confidence
degrades gracefully alongside accuracy, because a model that stays confident while
becoming wrong is the dangerous failure mode, and that is exactly what the abstention
mechanism is supposed to catch.

Revision addition (Reviewer 2, point 4). When a calibration file is supplied, the
temperature and abstention threshold that were selected on CLEAN validation data are
applied unchanged to every corrupted condition, and the realised coverage, selective
accuracy and selective risk are recorded together with whether the stated policy
(selective risk <= 5% at coverage >= 70%) still holds. The threshold that WOULD have
been required to satisfy the policy on that corrupted set is also reported, so the
reader can see how far the clean-data threshold is from an appropriate one.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))
sys.path.insert(0, str(HERE.parent / "export"))
from evaluate import evaluate as evaluate_predictions  # noqa: E402
from calibrate import probs_to_logits, softmax_T, risk_coverage, pick_threshold  # noqa: E402
from export_tflite import load_as_float32, verify_against_stored_predictions  # noqa: E402

SEVERITIES = (1, 2, 3)

# The complete severity grid, in one place, so it can be tabulated in the paper exactly
# as it is executed. Each corruption function reads its parameter from here.
PARAMS: dict[str, dict] = {
    "brightness_down": {"parameter": "PIL ImageEnhance.Brightness factor", "values": {1: 0.75, 2: 0.55, 3: 0.35}},
    "brightness_up":   {"parameter": "PIL ImageEnhance.Brightness factor", "values": {1: 1.3, 2: 1.6, 3: 2.0}},
    "contrast_down":   {"parameter": "PIL ImageEnhance.Contrast factor", "values": {1: 0.7, 2: 0.5, 3: 0.3}},
    "gaussian_blur":   {"parameter": "Gaussian blur radius (px, at 224 px)", "values": {1: 1.0, 2: 2.0, 3: 3.5}},
    "motion_blur":     {"parameter": "horizontal box-blur kernel length (px)", "values": {1: 5, 2: 9, 3: 15}},
    "jpeg_compression": {"parameter": "JPEG quality on re-encoding", "values": {1: 40, 2: 20, 3: 10}},
    "rotation":        {"parameter": "rotation (degrees, white fill)", "values": {1: 8, 2: 15, 3: 25}},
    "crop_zoom":       {"parameter": "central crop fraction, rescaled to 224 px", "values": {1: 0.85, 2: 0.70, 3: 0.55}},
}


# --------------------------------------------------------------------------- #
# Corruptions. Each takes a uint8 HxWx3 array and a severity in {1,2,3}.
# --------------------------------------------------------------------------- #
def c_brightness_down(img: np.ndarray, s: int) -> np.ndarray:
    return np.asarray(ImageEnhance.Brightness(Image.fromarray(img)).enhance(PARAMS["brightness_down"]["values"][s]))


def c_brightness_up(img: np.ndarray, s: int) -> np.ndarray:
    return np.asarray(ImageEnhance.Brightness(Image.fromarray(img)).enhance(PARAMS["brightness_up"]["values"][s]))


def c_contrast_down(img: np.ndarray, s: int) -> np.ndarray:
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(img)).enhance(PARAMS["contrast_down"]["values"][s]))


def c_gaussian_blur(img: np.ndarray, s: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(PARAMS["gaussian_blur"]["values"][s])))


def c_motion_blur(img: np.ndarray, s: int) -> np.ndarray:
    """Horizontal box blur, approximating hand shake during exposure."""
    k = PARAMS["motion_blur"]["values"][s]
    f = img.astype(np.float32)
    pad = np.pad(f, ((0, 0), (k // 2, k // 2), (0, 0)), mode="edge")
    out = np.zeros_like(f)
    for i in range(k):
        out += pad[:, i:i + f.shape[1], :]
    return np.clip(out / k, 0, 255).astype(np.uint8)


def c_jpeg(img: np.ndarray, s: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=PARAMS["jpeg_compression"]["values"][s])
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def c_rotation(img: np.ndarray, s: int) -> np.ndarray:
    deg = PARAMS["rotation"]["values"][s]
    return np.asarray(Image.fromarray(img).rotate(deg, resample=Image.BILINEAR, fillcolor=(255, 255, 255)))


def c_crop_zoom(img: np.ndarray, s: int) -> np.ndarray:
    """Partial framing: crop in and rescale, as when the leaf overflows the frame."""
    frac = PARAMS["crop_zoom"]["values"][s]
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = Image.fromarray(img[top:top + ch, left:left + cw])
    return np.asarray(crop.resize((w, h), Image.BILINEAR))


CORRUPTIONS = {
    "brightness_down": c_brightness_down,
    "brightness_up": c_brightness_up,
    "contrast_down": c_contrast_down,
    "gaussian_blur": c_gaussian_blur,
    "motion_blur": c_motion_blur,
    "jpeg_compression": c_jpeg,
    "rotation": c_rotation,
    "crop_zoom": c_crop_zoom,
}


def selective_metrics(probs: np.ndarray, y: np.ndarray, T: float, thr: float,
                      target_risk: float, min_cov: float) -> dict:
    """Apply the clean-validation temperature and threshold unchanged, then audit the policy."""
    cal = softmax_T(probs_to_logits(probs), T)
    conf = cal.max(axis=1)
    correct = cal.argmax(axis=1) == y
    keep = conf >= thr
    n_keep = int(keep.sum())
    sel_acc = float(correct[keep].mean()) if n_keep else None
    risk = (1.0 - sel_acc) if sel_acc is not None else None
    required = pick_threshold(risk_coverage(cal, y), target_risk, min_cov)
    return {
        "threshold_applied": thr,
        "coverage": float(n_keep / len(y)),
        "n_answered": n_keep,
        "n_abstained": int(len(y) - n_keep),
        "selective_accuracy": sel_acc,
        "selective_risk": risk,
        "accuracy_on_abstained": float(correct[~keep].mean()) if n_keep < len(y) else None,
        "calibrated_mean_confidence": float(conf.mean()),
        # The stated policy is a conjunction: selective risk <= target AND coverage >= floor.
        "risk_violated": bool(risk is not None and risk > target_risk),
        "coverage_violated": bool(n_keep / len(y) < min_cov),
        "policy_violated": bool((risk is not None and risk > target_risk) or (n_keep / len(y) < min_cov)),
        "required_threshold_for_policy": required["threshold"],
        "required_threshold_reachable": required["basis"] == "risk target met",
        "coverage_at_required_threshold": required["coverage"],
        "selective_risk_at_required_threshold": required["selective_risk"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--partition", default="test")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--calibration", default=None,
                    help="calibration_params.json; enables abstention auditing under corruption")
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--reference-predictions", default=None,
                    help="predictions_*_test.npz from the evaluation run; the reloaded graph must "
                         "reproduce its top-1 decisions before any corrupted condition is scored")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)

    x = np.load(cache / f"{a.split_prefix}_{a.partition}_x.npy")
    y = np.load(cache / f"{a.split_prefix}_{a.partition}_y.npy")

    # The checkpoint was trained under mixed_float16. The float32 rebuild is the canonical
    # inference graph (it is what was exported and what the handset runs) and, unlike the
    # half-precision original, it executes identically on CPU and GPU.
    model, load_info = load_as_float32(a.checkpoint, probe_tolerance=None)
    verification = None
    if a.reference_predictions:
        ref = np.load(a.reference_predictions)
        if not np.array_equal(ref["y_true"], y):
            raise SystemExit("reference predictions are not aligned with the test cache")
        verification = verify_against_stored_predictions(model, x, ref["probs"], a.batch_size)
        print(f"graph verified against stored predictions: top-1 agreement "
              f"{verification['top1_agreement_with_stored']:.4f}, max |dp| "
              f"{verification['max_abs_probability_deviation']:.2e}")

    calib = None
    if a.calibration:
        calib = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
        print(f"abstention audit enabled: T={calib['temperature']:.4f} "
              f"threshold={calib['abstention_threshold']:.4f} "
              f"policy risk<={calib['target_selective_risk']} coverage>={calib['min_coverage']}")

    def predict(arr: np.ndarray) -> np.ndarray:
        probs = model.predict(arr.astype("float32"), batch_size=a.batch_size, verbose=0).astype(np.float64)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(probs: np.ndarray) -> dict:
        return evaluate_predictions(y, probs, a.n_bootstrap, 12345)

    def audit(probs: np.ndarray) -> dict | None:
        if calib is None:
            return None
        return selective_metrics(probs, y, calib["temperature"], calib["abstention_threshold"],
                                 calib["target_selective_risk"], calib["min_coverage"])

    print(f"clean baseline on {a.split_prefix}/{a.partition} (n={len(y)})", flush=True)
    if a.reference_predictions:
        # The clean baseline IS the evaluation run's stored prediction set, so every drop
        # and the clean abstention row are relative to exactly the numbers the paper reports.
        clean_probs = np.load(a.reference_predictions)["probs"].astype(np.float64)
        clean_probs = clean_probs / clean_probs.sum(axis=1, keepdims=True)
        print("  (clean predictions taken from the stored evaluation run)")
    else:
        clean_probs = predict(x)
    clean = score(clean_probs)
    base_f1, base_acc = clean["macro_f1"], clean["accuracy"]
    base_ece = clean["calibration"]["ece"]
    base_conf = clean["calibration"]["mean_confidence"]
    print(f"  macro-F1 {base_f1:.4f}  acc {base_acc:.4f}  ECE {base_ece:.4f}  "
          f"mean_conf {base_conf:.4f}")
    clean_abst = audit(clean_probs)
    if clean_abst:
        print(f"  abstention: coverage {clean_abst['coverage']:.4f}  selective acc "
              f"{clean_abst['selective_accuracy']:.4f}  risk {clean_abst['selective_risk']:.4f}")

    results: dict = {
        "checkpoint": str(a.checkpoint),
        "checkpoint_load": load_info,
        "graph_verification": verification,
        "partition": f"{a.split_prefix}/{a.partition}",
        "n": int(len(y)),
        "corruption_specs": {
            name: {"parameter": spec["parameter"],
                   **{f"severity_{s}": spec["values"][s] for s in SEVERITIES}}
            for name, spec in PARAMS.items()
        },
        "abstention_policy": None if calib is None else {
            "temperature": calib["temperature"],
            "threshold": calib["abstention_threshold"],
            "target_selective_risk": calib["target_selective_risk"],
            "min_coverage": calib["min_coverage"],
            "selected_on": "clean validation partition; applied unchanged to every condition",
        },
        "clean": clean,
        "clean_abstention": clean_abst,
        "corruptions": {},
        "protocol": (
            "Corruptions are applied only to the frozen test partition, never to training "
            "data. Severity grids are hard-coded, so a rerun reproduces identical pixels. "
            "When --reference-predictions is given, the clean row is the stored evaluation "
            "prediction set (mixed-precision GPU graph) and corrupted rows come from the "
            "float32 rebuild; the two agree on every clean top-1 decision, so drops in "
            "points are unaffected, while probability-level deltas (confidence, ECE) carry "
            "float16 rounding of a few hundredths at most."
        ),
    }
    saved: dict[str, np.ndarray] = {"clean": clean_probs.astype(np.float32)}

    for name, fn in CORRUPTIONS.items():
        results["corruptions"][name] = {}
        for s in SEVERITIES:
            corrupted = np.stack([fn(img, s) for img in x])
            probs = predict(corrupted)
            saved[f"{name}__sev{s}"] = probs.astype(np.float32)
            r = score(probs)
            c = r["calibration"]
            entry = {
                "parameter_value": PARAMS[name]["values"][s],
                "macro_f1": r["macro_f1"],
                "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "mcc": r["mcc"],
                "ece": c["ece"],
                "nll": c["nll"],
                "brier": c["brier"],
                "mean_confidence": c["mean_confidence"],
                "overconfidence_gap": c["overconfidence_gap"],
                "macro_f1_drop_points": round(100 * (base_f1 - r["macro_f1"]), 3),
                "accuracy_drop_points": round(100 * (base_acc - r["accuracy"]), 3),
                "ece_change": round(c["ece"] - base_ece, 5),
                "confidence_change": round(c["mean_confidence"] - base_conf, 5),
                "per_class_recall": {k: v["recall"] for k, v in r["per_class"].items()},
                "abstention": audit(probs),
            }
            results["corruptions"][name][f"severity_{s}"] = entry
            ab = entry["abstention"]
            extra = ""
            if ab:
                extra = (f"  cov {ab['coverage']:.3f} selAcc {ab['selective_accuracy']:.3f} "
                         f"risk {ab['selective_risk']:.3f}{' VIOLATED' if ab['policy_violated'] else ''}")
            print(f"  {name:<18} sev {s}  F1 {r['macro_f1']:.4f} "
                  f"({entry['macro_f1_drop_points']:+.2f} pts)  "
                  f"conf {c['mean_confidence']:.3f} ({entry['confidence_change']:+.3f})  "
                  f"ECE {c['ece']:.4f}{extra}", flush=True)

    # Does confidence fall as accuracy falls? A model whose confidence holds while
    # accuracy collapses is the dangerous case the abstention policy must catch.
    pairs = [(v[f"severity_{s}"]["accuracy"], v[f"severity_{s}"]["mean_confidence"])
             for v in results["corruptions"].values() for s in SEVERITIES]
    accs = np.array([p[0] for p in pairs])
    confs = np.array([p[1] for p in pairs])
    if len(accs) > 2 and accs.std() > 0 and confs.std() > 0:
        r_cal = None
        if calib is not None:
            cal_confs = np.array([v[f"severity_{s}"]["abstention"]["calibrated_mean_confidence"]
                                  for v in results["corruptions"].values() for s in SEVERITIES])
            if cal_confs.std() > 0:
                r_cal = float(np.corrcoef(accs, cal_confs)[0, 1])
        results["confidence_tracks_accuracy"] = {
            "pearson_r": float(np.corrcoef(accs, confs)[0, 1]),
            "pearson_r_calibrated": r_cal,
            "n_conditions": int(len(accs)),
            "interpretation": (
                "A strongly positive correlation means confidence declines together with "
                "accuracy under degradation, which is what makes a confidence threshold a "
                "usable abstention signal. A weak or negative correlation would mean the "
                "model stays confident while becoming wrong."
            ),
        }
        print(f"\nconfidence-accuracy correlation across all corruptions: "
              f"r = {results['confidence_tracks_accuracy']['pearson_r']:.4f}")

    if calib is not None:
        cells = [(n, s, sev[f"severity_{s}"]["abstention"])
                 for n, sev in results["corruptions"].items() for s in SEVERITIES]
        viol = [(n, s) for n, s, ab in cells if ab["policy_violated"]]
        results["policy_summary"] = {
            "conditions": int(len(accs)),
            "conditions_violating_policy": len(viol),
            "risk_violations": sum(ab["risk_violated"] for _, _, ab in cells),
            "coverage_violations": sum(ab["coverage_violated"] for _, _, ab in cells),
            "coverage_violating": [f"{n}:sev{s}" for n, s, ab in cells if ab["coverage_violated"]],
            "violating": [f"{n}:sev{s}" for n, s in viol],
        }
        ps = results["policy_summary"]
        print(f"policy (risk <= {calib['target_selective_risk']}, coverage >= {calib['min_coverage']}) "
              f"violated in {len(viol)} of {len(accs)} corrupted conditions "
              f"(risk {ps['risk_violations']}, coverage floor {ps['coverage_violations']})")

    (outdir / "robustness.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not a.no_save_predictions:
        np.savez_compressed(outdir / "predictions_corrupted.npz", y_true=y, **saved)
    print(f"\nwritten: {outdir/'robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
