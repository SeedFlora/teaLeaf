"""Class-activation-map sanity analysis for the deployed model.

Reviewer 1 asked that the activation-map component be analysed rather than only timed:
visual examples, plus basic checks of whether the model attends to leaf symptoms rather
than to background, scale or framing cues.

What is analysed. The deployed graph computes, for every class c, the map
CAM_c = sum_k W[k, c] * F_k, where F is the final 7x7x960 feature map and W the classifier
weights (see export_with_cam.py). Because the head is global average pooling followed by a
linear layer, Grad-CAM for the class logit reduces to exactly this map -- the gradient of
the logit with respect to F_k is the constant W[k, c] / 49 -- so what is analysed here is
what the application overlays, not a separate attribution method. (The earlier gradcam.py
computed the feature map and the prediction along two disconnected graph branches, so its
gradients were identically zero and its panels showed no heatmap; it is superseded.)

Checks.
  1. Leaf mask. Every image is a detached leaf on a near-uniform light backdrop, so a leaf
     region can be segmented without a learned model: colour distance from the backdrop
     (estimated from the image border) thresholded by Otsu's method, cleaned
     morphologically, largest interior component kept. Mask quality is inspected visually
     (mask_qa.jpg) and images with an implausible mask are flagged and excluded from the
     summaries, never silently included.
  2. Map concentration. The fraction of the ReLU'd, normalised map mass that falls inside
     the leaf mask, divided by the mask's own area fraction. A ratio of 1.0 means the map
     is spread as if the leaf were not there; above 1.0 means it concentrates on the leaf.
     Whether the map's peak lies on the leaf is also recorded. The raw fraction alone would
     mislead: a leaf filling 80% of the frame collects 80% of the mass from a model that
     attends to nothing in particular.
  3. Counterfactuals. Behavioural checks that need no map at all: background-only (leaf
     painted out with the backdrop's median colour), leaf-only (backdrop flattened to its
     median colour), and a tight leaf crop rescaled to 224 px as the scale-and-framing
     control. For each, how often the original prediction survives, accuracy, calibrated
     confidence and coverage under the deployed abstention threshold are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "export"))
sys.path.insert(0, str(HERE.parent / "calibration"))
from export_tflite import CLASS_ORDER, load_as_float32  # noqa: E402
from calibrate import probs_to_logits, softmax_T  # noqa: E402

SHORT = ["algal spot", "brown blight", "gray blight", "Helopeltis", "red spider", "mirid bug", "healthy"]
K = len(CLASS_ORDER)
IMG = 224


# --------------------------------------------------------------------------- #
# Leaf segmentation
# --------------------------------------------------------------------------- #
def leaf_mask(img: np.ndarray, border: int = 8) -> tuple[np.ndarray, dict]:
    """Segment the leaf from a near-uniform backdrop. Returns (mask, diagnostics)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = np.ones((IMG, IMG), dtype=bool)
    ring[border:-border, border:-border] = False
    bg = np.median(lab[ring], axis=0)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    bg_sat = float(np.median(hsv[..., 1][ring]))

    # The backdrop is paper: light and unsaturated. If the border is itself saturated
    # (ground or a hand dominating the edge), colour distance from it is unreliable and
    # saturation is used directly instead.
    if bg_sat < 60:
        score = np.linalg.norm(lab - bg, axis=-1)
        basis = "lab_distance_from_border_median"
    else:
        score = hsv[..., 1].astype(np.float32)
        basis = "saturation"

    t = float(threshold_otsu(score))
    m = score > t
    m = morphology.binary_opening(m, morphology.disk(2))
    m = morphology.binary_closing(m, morphology.disk(3))
    m = ndimage.binary_fill_holes(m)

    labels = measure.label(m, connectivity=2)
    diag = {"basis": basis, "otsu": t, "border_saturation": bg_sat}
    if labels.max() == 0:
        diag.update(status="empty", area_fraction=0.0, touches_border=False)
        return np.zeros((IMG, IMG), dtype=bool), diag

    props = measure.regionprops(labels)

    def touches(p) -> bool:
        r0, c0, r1, c1 = p.bbox
        return r0 == 0 or c0 == 0 or r1 == IMG or c1 == IMG

    interior = [p for p in props if not touches(p) and p.area >= 0.01 * IMG * IMG]
    chosen = max(interior, key=lambda p: p.area) if interior else max(props, key=lambda p: p.area)
    mask = labels == chosen.label
    diag.update(status="ok", area_fraction=float(mask.mean()), touches_border=touches(chosen),
                n_components=int(labels.max()))
    return mask, diag


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=px) if px > 0 else mask


# --------------------------------------------------------------------------- #
# Model pieces
# --------------------------------------------------------------------------- #
def build_feature_model(model: tf.keras.Model):
    """Return (feature extractor, W, b) for the GAP + linear head.

    The classifier graph is input -> backbone -> GAP -> dropout -> dense -> softmax, so
    the backbone's own output IS the final feature map and the backbone can be called
    directly. The layer sequence is asserted rather than assumed, because CAM is only
    valid when nothing but pooling sits between the feature map and the linear layer.
    """
    names = [l.__class__.__name__ for l in model.layers]
    expected = ["InputLayer", "Functional", "GlobalAveragePooling2D", "Dropout", "Dense", "Activation"]
    if names != expected:
        raise SystemExit(f"unexpected head structure {names}; CAM requires {expected}")
    backbone = model.layers[1]
    dense = model.get_layer("logits")
    W, b = dense.get_weights()
    return backbone, W.astype(np.float64), b.astype(np.float64)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    c = cam / cam.max() if cam.max() > 0 else cam
    heat = (matplotlib.colormaps["jet"](c)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * img.astype(np.float32) + alpha * heat, 0, 255).astype(np.uint8)


def draw_contour(ax, mask: np.ndarray, color: str = "cyan") -> None:
    cs = measure.find_contours(mask.astype(np.uint8), 0.5)
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=0.6)


# --------------------------------------------------------------------------- #
def summarise(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        if len(v) == 0:
            out[k] = None
            continue
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--tables-outdir", required=True)
    ap.add_argument("--figures-outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    tdir, fdir = Path(a.tables_outdir), Path(a.figures_outdir)
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    x = np.load(Path(a.cache_dir) / f"{a.split_prefix}_test_x.npy")
    y = np.load(Path(a.cache_dir) / f"{a.split_prefix}_test_y.npy")
    z = np.load(Path(a.eval_dir) / f"predictions_{a.split_prefix}_test.npz")
    if not np.array_equal(z["y_true"], y):
        raise SystemExit("evaluation predictions are not aligned with the test cache")
    cal = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    T, thr = float(cal["temperature"]), float(cal["abstention_threshold"])

    # The float32 rebuild is the graph the application ships (export_with_cam.py builds the
    # CAM head from it), and it executes identically on CPU and GPU, unlike the
    # mixed-precision original. It is verified against the stored GPU predictions below.
    model, load_info = load_as_float32(a.checkpoint, probe_tolerance=None)
    feat, W, b = build_feature_model(model)
    n = len(x)

    # ---- the CAM head must reproduce the classifier exactly, or the maps are not the app's
    xf = x.astype("float32")
    fmap = feat.predict(xf, batch_size=a.batch_size, verbose=0).astype(np.float64)   # n,7,7,K
    probs_head = softmax(fmap.mean(axis=(1, 2)) @ W + b)
    probs_model = model.predict(xf, batch_size=a.batch_size, verbose=0).astype(np.float64)
    head_dev = float(np.abs(probs_head - probs_model).max())
    if head_dev > 1e-3:
        raise SystemExit(f"CAM head deviates from the classifier (max |dp| = {head_dev:.2e})")
    probs_ref = z["probs"].astype(np.float64)
    ref_dev = float(np.abs(probs_model - probs_ref).max())
    ref_agree = float((probs_model.argmax(1) == probs_ref.argmax(1)).mean())
    if ref_agree < 0.995 or ref_dev > 5e-2:
        raise SystemExit(f"reloaded graph does not reproduce stored predictions "
                         f"(agreement {ref_agree:.4f}, max |dp| {ref_dev:.2e})")
    print(f"feature map {fmap.shape[1:]}  classifier weights {W.shape}  "
          f"head reproduces model: max|dp| {head_dev:.2e}; vs stored predictions "
          f"agreement {ref_agree:.4f}, max|dp| {ref_dev:.2e}")

    # Everything below, including the "original image" reference row of the counterfactual
    # table, uses the same float32 graph that produces the maps and the edited images, so
    # rows are comparable. Its top-1 decisions equal the stored evaluation predictions.
    pred = probs_model.argmax(1)
    conf_cal = softmax_T(probs_to_logits(probs_model), T).max(1)
    correct = pred == y
    answered = conf_cal >= thr

    # ---- CAM for the predicted class, exactly as the application computes it
    cams = np.einsum("nhwk,nk->nhw", fmap, W[:, pred].T)                      # n,7,7
    cams = np.maximum(cams, 0.0)
    cams_up = np.stack([cv2.resize(c.astype(np.float32), (IMG, IMG), interpolation=cv2.INTER_LINEAR)
                        for c in cams])
    mass = cams_up.sum(axis=(1, 2))
    zero_map = mass <= 0
    cams_n = np.where(zero_map[:, None, None], 0.0, cams_up / np.maximum(mass, 1e-12)[:, None, None])

    # ---- leaf masks and per-image concentration metrics
    masks = np.zeros((n, IMG, IMG), dtype=bool)
    rows: list[dict] = []
    for i in range(n):
        m, d = leaf_mask(x[i])
        masks[i] = m
        area = float(m.mean())
        # Leaves in this dataset occupy a small fraction of the frame; a mask above 45% or
        # one reaching the border is, on inspection, foliage or a hand behind the paper.
        flagged = ((d["status"] != "ok") or area < 0.02 or area > 0.45
                   or bool(d.get("touches_border", False)))
        m_l = dilate(m, 4)                     # lenient region: the 7x7 map is coarse
        on_leaf = float(cams_n[i][m].sum()) if area > 0 else 0.0
        on_leaf_l = float(cams_n[i][m_l].sum()) if area > 0 else 0.0
        area_l = float(m_l.mean())
        pk = np.unravel_index(int(cams_up[i].argmax()), (IMG, IMG))
        rows.append({
            "test_index": i, "true": CLASS_ORDER[y[i]], "pred": CLASS_ORDER[pred[i]],
            "correct": bool(correct[i]), "confidence_calibrated": float(conf_cal[i]),
            "answered": bool(answered[i]),
            "mask_basis": d["basis"], "mask_status": d["status"],
            "leaf_area_fraction": area, "mask_touches_border": bool(d.get("touches_border", False)),
            "flagged": bool(flagged), "zero_map": bool(zero_map[i]),
            "cam_mass_on_leaf": on_leaf,
            "concentration_ratio": (on_leaf / area) if area > 0 else float("nan"),
            "cam_mass_on_leaf_dilated": on_leaf_l,
            "concentration_ratio_dilated": (on_leaf_l / area_l) if area_l > 0 else float("nan"),
            # The reported statistic uses the leaf mask itself; the 4-px tolerance version is
            # kept alongside it because the 7x7 map is coarse, but it is not what the paper quotes.
            "peak_on_leaf": bool(m[pk]) if area > 0 else False,
            "peak_on_leaf_dilated": bool(m_l[pk]) if area > 0 else False,
        })

    ok = [r for r in rows if not r["flagged"] and not r["zero_map"]]
    print(f"masks: {len(ok)} usable of {n}; flagged {sum(r['flagged'] for r in rows)}, "
          f"zero maps {int(zero_map.sum())}, border-touching {sum(r['mask_touches_border'] for r in rows)}")

    keys = ("leaf_area_fraction", "cam_mass_on_leaf", "concentration_ratio",
            "cam_mass_on_leaf_dilated", "concentration_ratio_dilated")

    def frac(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else None

    summary = {
        "checkpoint": a.checkpoint,
        "checkpoint_load": load_info,
        "method": {
            "map": "CAM_c = sum_k W[k,c] F_k on the final 7x7 feature map, for the predicted "
                   "class; identical to the graph the application ships (export_with_cam.py). "
                   "Equivalent to Grad-CAM here because the head is GAP + linear.",
            "head_reproduces_classifier_max_abs_dev": head_dev,
            "graph_verification": {"top1_agreement_with_stored": ref_agree,
                                   "max_abs_probability_deviation": ref_dev},
            "mask": "Otsu threshold on colour distance from the border-estimated backdrop "
                    "(or on saturation when the border is itself saturated), opening r=2, "
                    "closing r=3, hole filling, largest interior component.",
            "flag_rule": "mask empty, area < 2% or > 45% of the frame, or mask touching the border",
        },
        "n_test": n,
        "n_usable": len(ok),
        "n_flagged": int(sum(r["flagged"] for r in rows)),
        "n_zero_map": int(zero_map.sum()),
        "n_mask_touches_border": int(sum(r["mask_touches_border"] for r in rows)),
        "n_flagged_border_or_large": int(sum(r["flagged"] and (r["mask_touches_border"] or r["leaf_area_fraction"] > 0.45)
                                             for r in rows)),
        "n_flagged_small": int(sum(r["flagged"] and r["leaf_area_fraction"] < 0.02
                                   and not r["mask_touches_border"] for r in rows)),
        "n_mask_saturation_fallback": int(sum(r["mask_basis"] == "saturation" for r in rows)),
        "overall": {**summarise(ok, keys),
                    "fraction_peak_on_leaf": frac(ok, "peak_on_leaf"),
                    "fraction_peak_on_leaf_dilated": frac(ok, "peak_on_leaf_dilated"),
                    "fraction_ratio_above_1": float(np.mean([r["concentration_ratio"] > 1 for r in ok])),
                    "fraction_ratio_above_2": float(np.mean([r["concentration_ratio"] > 2 for r in ok]))},
        "by_correctness": {
            "correct": {**summarise([r for r in ok if r["correct"]], keys),
                        "fraction_peak_on_leaf": frac([r for r in ok if r["correct"]], "peak_on_leaf")},
            "incorrect": {**summarise([r for r in ok if not r["correct"]], keys),
                          "fraction_peak_on_leaf": frac([r for r in ok if not r["correct"]], "peak_on_leaf")},
        },
        "by_abstention": {
            "answered": {**summarise([r for r in ok if r["answered"]], keys),
                         "fraction_peak_on_leaf": frac([r for r in ok if r["answered"]], "peak_on_leaf")},
            "abstained": {**summarise([r for r in ok if not r["answered"]], keys),
                          "fraction_peak_on_leaf": frac([r for r in ok if not r["answered"]], "peak_on_leaf")},
        },
        "by_true_class": {
            c: {**summarise([r for r in ok if r["true"] == c], keys),
                "fraction_peak_on_leaf": frac([r for r in ok if r["true"] == c], "peak_on_leaf")}
            for c in CLASS_ORDER
        },
    }

    # ---- counterfactuals -------------------------------------------------------------
    def bg_colour(i: int) -> np.ndarray:
        outside = ~dilate(masks[i], 8)
        if outside.sum() < 100:
            outside = np.ones((IMG, IMG), dtype=bool)
        return np.median(x[i][outside].reshape(-1, 3), axis=0).astype(np.uint8)

    bg_only = np.empty_like(x)
    leaf_only = np.empty_like(x)
    leaf_crop = np.empty_like(x)
    for i in range(n):
        col = bg_colour(i)
        m8 = dilate(masks[i], 8)
        m3 = dilate(masks[i], 3)
        img = x[i]
        b_ = img.copy(); b_[m8] = col; bg_only[i] = b_
        l_ = np.empty_like(img); l_[:] = col; l_[m3] = img[m3]; leaf_only[i] = l_
        ys, xs = np.where(masks[i])
        if len(ys) == 0:
            leaf_crop[i] = img
            continue
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mr, mc = max(4, int(0.08 * (r1 - r0))), max(4, int(0.08 * (c1 - c0)))
        r0, r1 = max(0, r0 - mr), min(IMG, r1 + mr)
        c0, c1 = max(0, c0 - mc), min(IMG, c1 + mc)
        leaf_crop[i] = np.asarray(Image.fromarray(img[r0:r1, c0:c1]).resize((IMG, IMG), Image.BILINEAR))

    # One exclusion set for the map summaries and the counterfactuals alike.
    usable = np.array([not r["flagged"] and not r["zero_map"] for r in rows])
    counterfactual: dict = {}
    for name, arr, desc in (
        ("background_only", bg_only, "leaf painted out with the backdrop median colour"),
        ("leaf_only", leaf_only, "backdrop replaced by its median colour, leaf untouched"),
        ("leaf_crop", leaf_crop, "tight crop around the leaf (8% margin) rescaled to 224 px"),
    ):
        p = model.predict(arr.astype("float32"), batch_size=a.batch_size, verbose=0).astype(np.float64)
        p = p / p.sum(axis=1, keepdims=True)
        pc = softmax_T(probs_to_logits(p), T)
        cf_pred, cf_conf = pc.argmax(1), pc.max(1)
        u = usable
        hist = np.bincount(cf_pred[u], minlength=K)
        counterfactual[name] = {
            "description": desc,
            "n": int(u.sum()),
            "prediction_survives": float((cf_pred[u] == pred[u]).mean()),
            "accuracy": float((cf_pred[u] == y[u]).mean()),
            "accuracy_original_same_images": float(correct[u].mean()),
            "mean_calibrated_confidence": float(cf_conf[u].mean()),
            "mean_calibrated_confidence_original": float(conf_cal[u].mean()),
            "coverage_at_deployed_threshold": float((cf_conf[u] >= thr).mean()),
            "coverage_original": float(answered[u].mean()),
            "predicted_class_histogram": {CLASS_ORDER[k]: int(hist[k]) for k in range(K)},
        }
        cf = counterfactual[name]
        print(f"{name:<16} survives {cf['prediction_survives']:.3f}  acc {cf['accuracy']:.3f} "
              f"(orig {cf['accuracy_original_same_images']:.3f})  conf {cf['mean_calibrated_confidence']:.3f} "
              f"(orig {cf['mean_calibrated_confidence_original']:.3f})  coverage {cf['coverage_at_deployed_threshold']:.3f}")
        np.savez_compressed(tdir / f"predictions_{name}.npz", y_true=y, probs=p.astype(np.float32))
    summary["counterfactuals"] = counterfactual
    summary["abstention"] = {"temperature": T, "threshold": thr}

    o = summary["overall"]
    print(f"\nCAM mass on leaf: median {o['cam_mass_on_leaf']['median']:.3f} "
          f"(leaf area median {o['leaf_area_fraction']['median']:.3f}); "
          f"concentration ratio median {o['concentration_ratio']['median']:.2f}; "
          f"peak on leaf {o['fraction_peak_on_leaf']:.3f}; ratio>1 in {o['fraction_ratio_above_1']:.3f}")

    (tdir / "cam_sanity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (tdir / "cam_per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- figures ---------------------------------------------------------------------
    # (a) mask QA: random sample plus the smallest and largest masks
    order = np.argsort([r["leaf_area_fraction"] for r in rows])
    qa_idx = list(rng.choice(n, 18, replace=False)) + list(order[:3]) + list(order[-3:])
    fig, axes = plt.subplots(4, 6, figsize=(9, 6.2))
    for ax, i in zip(axes.ravel(), qa_idx):
        ax.imshow(x[i]); draw_contour(ax, masks[i]); ax.set_axis_off()
        ax.set_title(f"#{i} {SHORT[y[i]]} a={rows[i]['leaf_area_fraction']:.2f}"
                     f"{' FLAG' if rows[i]['flagged'] else ''}", fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(fdir / "mask_qa.jpg", dpi=150)
    plt.close(fig)

    # (b) main-text panel: correct examples of four classes, then the four most confident
    #     errors with distinct (true -> predicted) pairs
    def cell(ax, i, title):
        ax.imshow(overlay(x[i], cams_up[i])); draw_contour(ax, masks[i]); ax.set_axis_off()
        ax.set_title(title, fontsize=5.8, pad=2, linespacing=1.1)

    panel_correct = []
    for c in (0, 1, 3, 4):
        pool = np.where(correct & (y == c) & usable)[0]
        panel_correct.append(int(rng.choice(pool)))
    err_pool = np.where(~correct & usable)[0]
    err_pool = err_pool[np.argsort(-conf_cal[err_pool])]
    seen, panel_wrong = set(), []
    for i in err_pool:
        key = (int(y[i]), int(pred[i]))
        if key in seen:
            continue
        seen.add(key); panel_wrong.append(int(i))
        if len(panel_wrong) == 4:
            break
    fig, axes = plt.subplots(2, 4, figsize=(4.9, 3.15))
    for ax, i in zip(axes[0], panel_correct):
        cell(ax, i, f"{SHORT[y[i]]}\ncorrect, p={conf_cal[i]:.2f}")
    for ax, i in zip(axes[1], panel_wrong):
        cell(ax, i, f"true: {SHORT[y[i]]}\npredicted: {SHORT[pred[i]]}, p={conf_cal[i]:.2f}")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.905, bottom=0.005, wspace=0.03, hspace=0.30)
    fig.savefig(fdir / "cam_panel.pdf")
    fig.savefig(fdir / "cam_panel.jpg", dpi=300)
    plt.close(fig)

    # Single-row variant for a page-limited main text: two correct, two confident errors.
    fig, axes = plt.subplots(1, 4, figsize=(4.9, 1.55))
    for ax, i in zip(axes[:2], panel_correct[:2]):
        cell(ax, i, f"{SHORT[y[i]]}\ncorrect, p={conf_cal[i]:.2f}")
    for ax, i in zip(axes[2:], panel_wrong[:2]):
        cell(ax, i, f"true: {SHORT[y[i]]}\npredicted: {SHORT[pred[i]]}, p={conf_cal[i]:.2f}")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.005, wspace=0.03)
    fig.savefig(fdir / "cam_panel_row.pdf")
    fig.savefig(fdir / "cam_panel_row.jpg", dpi=300)
    plt.close(fig)

    # (c) supplementary: per class, a correct example and the most confident error
    fig, axes = plt.subplots(K, 4, figsize=(6.4, 1.62 * K))
    supp_items = []
    for c in range(K):
        pool = np.where(correct & (y == c) & usable)[0]
        ic = int(rng.choice(pool))
        axes[c, 0].imshow(x[ic]); draw_contour(axes[c, 0], masks[ic])
        axes[c, 0].set_title(f"{SHORT[c]}: correct, p={conf_cal[ic]:.2f}", fontsize=6.5, pad=2)
        axes[c, 1].imshow(overlay(x[ic], cams_up[ic])); axes[c, 1].set_title("CAM", fontsize=6.5, pad=2)
        errs = np.where(~correct & (y == c) & usable)[0]
        if len(errs):
            ie = int(errs[np.argmax(conf_cal[errs])])
            axes[c, 2].imshow(x[ie]); draw_contour(axes[c, 2], masks[ie])
            axes[c, 2].set_title(f"→ {SHORT[pred[ie]]}, p={conf_cal[ie]:.2f}", fontsize=6.5, pad=2)
            axes[c, 3].imshow(overlay(x[ie], cams_up[ie])); axes[c, 3].set_title("CAM", fontsize=6.5, pad=2)
            supp_items.append({"class": CLASS_ORDER[c], "correct_index": ic, "error_index": ie})
        else:
            for ax in axes[c, 2:]:
                ax.text(0.5, 0.5, "no test error\nfor this class", ha="center", va="center", fontsize=7)
            supp_items.append({"class": CLASS_ORDER[c], "correct_index": ic, "error_index": None})
        for ax in axes[c]:
            ax.set_axis_off()
    fig.subplots_adjust(left=0.005, right=0.995, top=0.975, bottom=0.005, wspace=0.03, hspace=0.2)
    fig.savefig(fdir / "cam_examples_by_class.jpg", dpi=220)
    plt.close(fig)

    summary["figures"] = {
        "cam_panel": {"correct": panel_correct, "wrong": panel_wrong,
                      "items": [{"test_index": i, "true": CLASS_ORDER[y[i]], "pred": CLASS_ORDER[pred[i]],
                                 "confidence_calibrated": float(conf_cal[i]),
                                 "concentration_ratio": rows[i]["concentration_ratio"]}
                                for i in panel_correct + panel_wrong]},
        "cam_examples_by_class": supp_items,
        "mask_qa": [int(i) for i in qa_idx],
    }
    (tdir / "cam_sanity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten: {tdir/'cam_sanity.json'}, {tdir/'cam_per_image.csv'}, figures in {fdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
