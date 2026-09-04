"""Export FP32/FP16/INT8 TFLite variants and validate each against the Keras reference.

Parity is checked on the FULL frozen test partition, not a handful of samples, and it is
reported at three levels because they answer different questions:

  * max absolute probability deviation -- numerical drift
  * top-1 agreement rate               -- whether the deployed model makes the same decision
  * per-class flip counts              -- whether drift concentrates in one class

The third matters most under a 3.07x class imbalance: an export that loses one point of
aggregate accuracy by destroying the smallest class is not an acceptable trade, and an
aggregate number alone would hide that.

The INT8 representative dataset is drawn from TRAINING images only. Using validation or
test images to calibrate quantization would leak held-out data into the deployed model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_ORDER = [
    "1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight", "4. Helopeltis",
    "5. Red spider", "6. Green mirid bug", "7. Healthy leaf",
]

# Fixed in advance of inspecting any test result, so the pass/fail rule cannot be
# rationalized after the fact.
MATERIAL_DEGRADATION_MACRO_F1_POINTS = 1.5
MATERIAL_PER_CLASS_RECALL_POINTS = 5.0


def to_float32_config(cfg):
    """Rewrite every float16 dtype policy in a serialized Keras config to float32."""
    if isinstance(cfg, dict):
        if cfg.get("class_name") in ("DTypePolicy", "Policy"):
            inner = cfg.get("config") or {}
            if "float16" in str(inner.get("name", "")):
                return {"class_name": cfg["class_name"], "config": {"name": "float32"}}
        out = {}
        for k, v in cfg.items():
            if k == "dtype" and isinstance(v, str) and "float16" in v:
                out[k] = "float32"
            else:
                out[k] = to_float32_config(v)
        return out
    if isinstance(cfg, list):
        return [to_float32_config(v) for v in cfg]
    return cfg


def load_as_float32(checkpoint: str, probe_tolerance: float | None = 1e-2) -> tuple[tf.keras.Model, dict]:
    """Load a checkpoint and return an equivalent pure-float32 graph.

    Training used a mixed_float16 policy for throughput, and Keras serializes that
    policy into every layer's config. Reloading therefore reconstructs a half-precision
    graph, which the TFLite converter cannot lower to a FlatBuffer -- the failure
    surfaces only at export time, long after the checkpoint looks fine.

    Rebuilding from a dtype-rewritten config and copying the weights across is lossless:
    widening float16 values to float32 is exact, and the layer topology is untouched.

    The random-pixel probe compares the rebuilt graph with the half-precision original
    AS EXECUTED ON THE CURRENT DEVICE. On a GPU that is a meaningful check; on a CPU,
    float16 is emulated and the ORIGINAL is the inaccurate side, so callers running on
    CPU pass probe_tolerance=None and verify against stored GPU predictions instead.
    """
    tf.keras.mixed_precision.set_global_policy("float32")
    original = tf.keras.models.load_model(checkpoint, compile=False)

    info = {"rebuilt_as_float32": False,
            "original_policy": str(getattr(original, "dtype_policy", "unknown"))}

    cfg = original.get_config()
    new_cfg = to_float32_config(cfg)
    if new_cfg == cfg:
        return original, info

    rebuilt = tf.keras.Model.from_config(new_cfg)
    rebuilt.set_weights(original.get_weights())

    # Prove the rebuild is behaviourally identical before trusting it.
    probe = np.random.default_rng(0).integers(0, 256, size=(4, 224, 224, 3)).astype("float32")
    a = original.predict(probe, verbose=0).astype(np.float64)
    b = rebuilt.predict(probe, verbose=0).astype(np.float64)
    dev = float(np.abs(a - b).max())
    info.update(rebuilt_as_float32=True,
                rebuild_max_abs_deviation=dev,
                rebuild_top1_agreement=float((a.argmax(1) == b.argmax(1)).mean()))
    if probe_tolerance is not None and dev > probe_tolerance:
        raise SystemExit(f"float32 rebuild diverged from the checkpoint (max |dp| = {dev:.3e})")
    return rebuilt, info


def verify_against_stored_predictions(model: tf.keras.Model, x: np.ndarray, stored_probs: np.ndarray,
                                      batch_size: int = 64, min_agreement: float = 0.995,
                                      max_dev: float = 5e-2) -> dict:
    """Check a reloaded graph against predictions saved from the evaluation run.

    Top-1 agreement is the decisive criterion: every downstream number in the paper is a
    function of the argmax, so a graph that reproduces the stored argmax on the whole test
    partition reproduces the paper's figures. The probability deviation is reported so that
    numerical drift is visible even when it flips nothing; stored predictions come from the
    mixed_float16 graph, so deviations of a few hundredths are float16 rounding (the FP16
    TFLite export shows the same 3.5e-2 against the float32 reference).
    """
    p = model.predict(x.astype("float32"), batch_size=batch_size, verbose=0).astype(np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    s = stored_probs.astype(np.float64)
    agree = float((p.argmax(1) == s.argmax(1)).mean())
    dev = float(np.abs(p - s).max())
    out = {"top1_agreement_with_stored": agree, "max_abs_probability_deviation": dev,
           "n": int(len(x)), "min_agreement_required": min_agreement}
    if agree < min_agreement or dev > max_dev:
        raise SystemExit(f"reloaded graph does not reproduce stored predictions: agreement {agree:.4f}, "
                         f"max |dp| {dev:.2e}")
    return out


def make_interpreter(model_bytes: bytes, num_threads: int = 4) -> tuple[object, dict]:
    """Build a working interpreter, recording which delegate configuration succeeded.

    XNNPACK is applied by default and can fail to prepare on a quantized graph, raising
    only at allocate_tensors(). That is a genuine deployment fact rather than a script
    bug -- the same delegate question arises on Android -- so each configuration is
    tried in turn and the outcome is recorded instead of being swallowed.
    """
    attempts: list[dict] = []

    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRT
        it = LiteRT(model_content=model_bytes, num_threads=num_threads)
        it.allocate_tensors()
        return it, {"runtime": "ai_edge_litert", "delegate": "xnnpack (default)",
                    "attempts": attempts}
    except Exception as exc:  # noqa: BLE001
        attempts.append({"config": "ai_edge_litert + default XNNPACK",
                         "error": f"{type(exc).__name__}: {str(exc)[:180]}"})

    try:
        it = tf.lite.Interpreter(model_content=model_bytes, num_threads=num_threads)
        it.allocate_tensors()
        return it, {"runtime": "tf.lite", "delegate": "xnnpack (default)", "attempts": attempts}
    except Exception as exc:  # noqa: BLE001
        attempts.append({"config": "tf.lite + default XNNPACK",
                         "error": f"{type(exc).__name__}: {str(exc)[:180]}"})

    # Last resort: plain reference kernels, no delegate.
    it = tf.lite.Interpreter(
        model_content=model_bytes,
        num_threads=num_threads,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    return it, {"runtime": "tf.lite", "delegate": "none (XNNPACK disabled)", "attempts": attempts}


def run_tflite(model_bytes: bytes, x: np.ndarray, num_threads: int = 4) -> tuple[np.ndarray, dict]:
    interp, runtime_info = make_interpreter(model_bytes, num_threads)
    runtime = runtime_info["runtime"]
    di, do = interp.get_input_details()[0], interp.get_output_details()[0]

    in_dtype = np.dtype(di["dtype"])
    in_scale, in_zp = di.get("quantization", (0.0, 0))
    out_scale, out_zp = do.get("quantization", (0.0, 0))

    n = len(x)
    out = np.zeros((n, len(CLASS_ORDER)), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        sample = x[i:i + 1].astype("float32")
        if in_dtype in (np.int8, np.uint8) and in_scale:
            q = np.round(sample / in_scale + in_zp)
            lo, hi = (-128, 127) if in_dtype == np.int8 else (0, 255)
            sample = np.clip(q, lo, hi).astype(in_dtype)
        else:
            sample = sample.astype(in_dtype)
        interp.set_tensor(di["index"], sample)
        interp.invoke()
        raw = interp.get_tensor(do["index"])[0].astype(np.float64)
        if np.dtype(do["dtype"]) in (np.int8, np.uint8) and out_scale:
            raw = (raw - out_zp) * out_scale
        out[i] = raw
    elapsed = time.perf_counter() - t0

    # Guard: if the graph ends before softmax, normalize so comparisons are like-for-like.
    s = out.sum(axis=1, keepdims=True)
    if not np.allclose(s, 1.0, atol=1e-2):
        e = np.exp(out - out.max(axis=1, keepdims=True))
        out = e / e.sum(axis=1, keepdims=True)
    else:
        out = out / s

    meta = {
        **runtime_info,
        "input_dtype": str(in_dtype), "input_shape": [int(v) for v in di["shape"]],
        "input_quantization": {"scale": float(in_scale), "zero_point": int(in_zp)},
        "output_dtype": str(np.dtype(do["dtype"])), "output_shape": [int(v) for v in do["shape"]],
        "output_quantization": {"scale": float(out_scale), "zero_point": int(out_zp)},
        "desktop_seconds_total": round(elapsed, 3),
        "desktop_ms_per_image": round(1000 * elapsed / max(n, 1), 3),
        "num_threads": num_threads,
    }
    return out, meta


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(axis=1)
    k = len(CLASS_ORDER)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(k), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(k), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(rec.mean()),
        "per_class_recall": {CLASS_ORDER[i]: float(rec[i]) for i in range(k)},
        "per_class_f1": {CLASS_ORDER[i]: float(f1[i]) for i in range(k)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rep-samples", type=int, default=300)
    ap.add_argument("--rep-seed", type=int, default=42)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)

    model, load_info = load_as_float32(a.checkpoint)
    if load_info["rebuilt_as_float32"]:
        print(f"rebuilt checkpoint as float32 for export "
              f"(max |dp| {load_info['rebuild_max_abs_deviation']:.2e}, "
              f"top-1 agreement {load_info['rebuild_top1_agreement']:.4f})")
    xt = np.load(cache / f"{a.split_prefix}_test_x.npy").astype("float32")
    yt = np.load(cache / f"{a.split_prefix}_test_y.npy")
    xtr = np.load(cache / f"{a.split_prefix}_train_x.npy")

    ref = model.predict(xt, batch_size=64, verbose=0).astype(np.float64)
    ref = ref / ref.sum(axis=1, keepdims=True)
    ref_m = metrics(yt, ref)
    print(f"Keras reference: macro-F1 {ref_m['macro_f1']:.4f}  acc {ref_m['accuracy']:.4f}")

    # Representative data for INT8: training images only, resized to the eval size the
    # deployed graph expects.
    rng = np.random.default_rng(a.rep_seed)
    idx = rng.choice(len(xtr), size=min(a.rep_samples, len(xtr)), replace=False)
    rep = tf.image.resize(xtr[idx].astype("float32"), (xt.shape[1], xt.shape[2]),
                          method="bilinear").numpy()
    np.save(outdir / "representative_indices.npy", idx)

    def rep_gen():
        for i in range(len(rep)):
            yield [rep[i:i + 1]]

    variants: dict[str, tf.lite.TFLiteConverter] = {}

    c32 = tf.lite.TFLiteConverter.from_keras_model(model)
    variants["fp32"] = c32

    c16 = tf.lite.TFLiteConverter.from_keras_model(model)
    c16.optimizations = [tf.lite.Optimize.DEFAULT]
    c16.target_spec.supported_types = [tf.float16]
    variants["fp16"] = c16

    c8 = tf.lite.TFLiteConverter.from_keras_model(model)
    c8.optimizations = [tf.lite.Optimize.DEFAULT]
    c8.representative_dataset = rep_gen
    # Leave the interface float so the Android app feeds raw [0,255] pixels for every
    # variant; only the internal computation is integer.
    variants["int8"] = c8

    # Weights-only (dynamic-range) quantization as a third compression point.
    # Full-integer PTQ quantizes activations too, and MobileNetV3's hard-swish and
    # squeeze-excite blocks have activation ranges that a small calibration set
    # represents poorly. Dynamic-range quantization leaves activations in float, which
    # recovers most of the size benefit without that failure mode -- worth measuring
    # rather than assuming, since it is the variant most likely to be deployable.
    cdr = tf.lite.TFLiteConverter.from_keras_model(model)
    cdr.optimizations = [tf.lite.Optimize.DEFAULT]
    variants["int8_dynamic_range"] = cdr

    results: dict = {
        "checkpoint": str(a.checkpoint),
        "checkpoint_load": load_info,
        "keras_reference": ref_m,
        "representative_dataset": {
            "source": "training partition only",
            "n_samples": int(len(rep)),
            "seed": a.rep_seed,
            "rationale": ("Calibrating quantization on validation or test images would leak "
                          "held-out data into the deployed model."),
        },
        "material_degradation_thresholds": {
            "macro_f1_points": MATERIAL_DEGRADATION_MACRO_F1_POINTS,
            "per_class_recall_points": MATERIAL_PER_CLASS_RECALL_POINTS,
            "fixed_before_inspecting_results": True,
        },
        "variants": {},
    }

    for name, conv in variants.items():
        print(f"\n--- converting {name} ---", flush=True)
        try:
            blob = conv.convert()
        except Exception as exc:  # noqa: BLE001
            results["variants"][name] = {"export_ok": False,
                                         "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            print(f"  CONVERSION FAILED: {exc}")
            continue

        path = outdir / f"tealeaf_{name}.tflite"
        path.write_bytes(blob)

        probs, meta = run_tflite(blob, xt)
        m = metrics(yt, probs)

        dev = np.abs(probs - ref)
        agree = float((probs.argmax(1) == ref.argmax(1)).mean())
        flips = {}
        for c in range(len(CLASS_ORDER)):
            sel = ref.argmax(1) == c
            if sel.sum():
                flips[CLASS_ORDER[c]] = int((probs.argmax(1)[sel] != c).sum())

        recall_drops = {k: round(100 * (ref_m["per_class_recall"][k] - m["per_class_recall"][k]), 3)
                        for k in CLASS_ORDER}
        worst_recall_drop = max(recall_drops.values())
        f1_drop_pts = 100 * (ref_m["macro_f1"] - m["macro_f1"])

        material = (f1_drop_pts > MATERIAL_DEGRADATION_MACRO_F1_POINTS
                    or worst_recall_drop > MATERIAL_PER_CLASS_RECALL_POINTS)

        results["variants"][name] = {
            "export_ok": True,
            "file": path.name,
            "bytes": len(blob),
            "mb": round(len(blob) / 1e6, 3),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "interpreter": meta,
            "metrics": m,
            "parity": {
                "max_abs_probability_deviation": float(dev.max()),
                "mean_abs_probability_deviation": float(dev.mean()),
                "p99_abs_probability_deviation": float(np.percentile(dev, 99)),
                "top1_agreement_with_keras": agree,
                "n_top1_disagreements": int((probs.argmax(1) != ref.argmax(1)).sum()),
                "per_class_flip_counts": flips,
            },
            "degradation": {
                "macro_f1_drop_points": round(f1_drop_pts, 3),
                "accuracy_drop_points": round(100 * (ref_m["accuracy"] - m["accuracy"]), 3),
                "per_class_recall_drop_points": recall_drops,
                "worst_per_class_recall_drop_points": round(worst_recall_drop, 3),
                "material_degradation": bool(material),
            },
        }

        print(f"  {path.name}: {len(blob)/1e6:.2f} MB  runtime={meta['runtime']}  delegate={meta['delegate']}")
        print(f"  macro-F1 {m['macro_f1']:.4f} ({f1_drop_pts:.2f} pts WORSE than Keras)  acc {m['accuracy']:.4f}")
        print(f"  top-1 agreement with Keras {agree:.4f}  "
              f"({int((probs.argmax(1) != ref.argmax(1)).sum())} disagreements)")
        print(f"  max |dp| {dev.max():.2e}   worst per-class recall LOSS {worst_recall_drop:.2f} pts")
        print(f"  MATERIAL DEGRADATION: {material}")

    labels = outdir / "labels.txt"
    labels.write_text("\n".join(CLASS_ORDER), encoding="utf-8")

    (outdir / "export_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {outdir/'export_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



