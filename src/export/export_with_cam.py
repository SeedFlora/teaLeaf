"""Export a deployment model that returns class activation maps alongside probabilities.

Grad-CAM is unavailable on a TFLite runtime because it needs gradients and the runtime
performs inference only. Class Activation Mapping does not: when a global average pooling
layer feeds directly into a linear classifier -- exactly this architecture -- the
classifier's weight for class c IS the per-channel importance for class c, so the map is
a weighted sum computable in a forward pass.

The weighted sum is folded INTO the graph rather than shipped as weights for the app to
apply. Sending a 576x7 matrix through the config would create two sources of truth that
could silently desynchronize when the model is replaced; emitting a 7x7x7 tensor as a
second output means the app only selects a channel.

What the map is: where the evidence for the predicted class is concentrated.
What it is not: an explanation of why, in biological terms. A heatmap over a lesion is
not evidence that the network reasoned about lesion morphology. Its concrete diagnostic
value here is the opposite case -- attention on the plain studio backdrop rather than the
leaf would indicate shortcut learning, which matters because every training image is a
detached leaf on uniform paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_tflite import (  # noqa: E402
    CLASS_ORDER, load_as_float32, make_interpreter, metrics,
)

DISPLAY_EN = [
    "Tea algal leaf spot", "Brown blight", "Gray blight", "Helopeltis damage",
    "Red spider mite damage", "Green mirid bug damage", "Healthy leaf",
]
DISPLAY_ID = [
    "Bercak daun alga", "Hawar coklat", "Hawar kelabu", "Kerusakan Helopeltis",
    "Kerusakan tungau merah", "Kerusakan kepik hijau", "Daun sehat",
]
NORMALIZED = [
    "tea_algal_leaf_spot", "brown_blight", "gray_blight", "helopeltis_damage",
    "red_spider_mite_damage", "green_mirid_bug_damage", "healthy_leaf",
]


def find_pieces(model: tf.keras.Model):
    """Locate the final feature map and the classifier weights.

    CAM is only valid when global average pooling feeds the classifier directly. If the
    head has anything else in between, the classifier weights stop being per-channel
    importances and the map would be meaningless, so that structure is verified here
    rather than assumed.
    """
    backbone = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if backbone is None:
        raise SystemExit("expected a nested backbone model")

    feature_layer = None
    for layer in backbone.layers:
        try:
            shape = layer.output.shape
        except Exception:  # noqa: BLE001
            continue
        if len(shape) == 4 and shape[-1] is not None:
            feature_layer = layer
    if feature_layer is None:
        raise SystemExit("no 4-D feature map found in the backbone")

    dense = next((l for l in model.layers if isinstance(l, tf.keras.layers.Dense)), None)
    if dense is None:
        raise SystemExit("no Dense classifier found")

    has_gap = any(isinstance(l, tf.keras.layers.GlobalAveragePooling2D) for l in model.layers)
    if not has_gap:
        raise SystemExit(
            "CAM requires global average pooling immediately before the classifier; "
            "this model does not have it, so the classifier weights are not per-channel "
            "importances and a CAM would be meaningless"
        )

    return backbone, feature_layer, dense


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--calibration", required=True,
                    help="calibration_params.json holding the temperature and threshold")
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--variant", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--model-version", default="1.0.0")
    a = ap.parse_args()

    outdir, assets = Path(a.outdir), Path(a.assets_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    model, load_info = load_as_float32(a.checkpoint)
    backbone, feature_layer, dense = find_pieces(model)

    W = dense.get_weights()[0]                      # (channels, num_classes)
    print(f"feature layer : {feature_layer.name} {feature_layer.output.shape}")
    print(f"classifier    : {dense.name} weights {W.shape}")

    # Rebuild with two outputs. The CAM head is a single contraction over channels,
    # producing one spatial map per class; the app selects the predicted class's channel.
    inp = model.inputs[0]
    feat_extractor = tf.keras.Model(backbone.inputs, feature_layer.output)
    fmap = feat_extractor(inp)

    cam = tf.keras.layers.Lambda(
        lambda t: tf.einsum("bhwk,kc->bhwc", t, tf.constant(W, dtype=t.dtype)),
        name="class_activation_maps",
    )(fmap)

    dual = tf.keras.Model(inp, [model(inp), cam], name="tealeaf_with_cam")

    conv = tf.lite.TFLiteConverter.from_keras_model(dual)
    if a.variant == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    blob = conv.convert()

    model_name = f"tealeaf_{a.variant}_cam.tflite"
    (outdir / model_name).write_bytes(blob)
    (assets / model_name).write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    print(f"{model_name}: {len(blob)/1e6:.2f} MB  sha256 {sha[:16]}...")

    # Verify the dual-output graph still classifies identically to the single-output one.
    xt = np.load(Path(a.cache_dir) / f"{a.split_prefix}_test_x.npy").astype("float32")
    yt = np.load(Path(a.cache_dir) / f"{a.split_prefix}_test_y.npy")
    ref = model.predict(xt, batch_size=64, verbose=0).astype(np.float64)
    ref = ref / ref.sum(axis=1, keepdims=True)

    interp, runtime_info = make_interpreter(blob, num_threads=4)
    ins = interp.get_input_details()
    outs = interp.get_output_details()
    print(f"runtime {runtime_info['runtime']} delegate {runtime_info['delegate']}")
    for o in outs:
        print(f"  output '{o['name']}' shape {list(o['shape'])} dtype {np.dtype(o['dtype'])}")

    # Identify which output is the probability vector by its rank.
    prob_idx = next(i for i, o in enumerate(outs) if len(o["shape"]) == 2)
    cam_idx = next(i for i, o in enumerate(outs) if len(o["shape"]) == 4)

    probs = np.zeros((len(xt), len(CLASS_ORDER)), dtype=np.float64)
    cam_shape = None
    for i in range(len(xt)):
        interp.set_tensor(ins[0]["index"], xt[i:i + 1])
        interp.invoke()
        probs[i] = interp.get_tensor(outs[prob_idx]["index"])[0]
        if cam_shape is None:
            cam_shape = list(interp.get_tensor(outs[cam_idx]["index"]).shape)
    probs = probs / probs.sum(axis=1, keepdims=True)

    m = metrics(yt, probs)
    agree = float((probs.argmax(1) == ref.argmax(1)).mean())
    dev = float(np.abs(probs - ref).max())
    print(f"\nCAM-enabled export: macro-F1 {m['macro_f1']:.4f}  "
          f"top-1 agreement with Keras {agree:.4f}  max |dp| {dev:.2e}")
    print(f"CAM tensor shape: {cam_shape}")

    if agree < 0.99:
        raise SystemExit(
            f"adding the CAM output changed predictions (agreement {agree:.4f}); refusing to ship"
        )

    cal = json.loads(Path(a.calibration).read_text(encoding="utf-8"))

    config = {
        "model_asset": model_name,
        "model_sha256": sha,
        "variant": a.variant,
        "version": a.model_version,
        "input_size": int(xt.shape[1]),
        "input_convention": "raw [0,255] float32 RGB; normalization embedded in the graph",
        "class_order": NORMALIZED,
        "dataset_class_folders": CLASS_ORDER,
        "display_labels_en": DISPLAY_EN,
        "display_labels_id": DISPLAY_ID,
        "temperature": cal["temperature"],
        "abstention_threshold": cal["abstention_threshold"],
        "target_selective_risk": cal.get("target_selective_risk"),
        "min_coverage": cal.get("min_coverage"),
        "num_threads": 4,
        "use_xnnpack": True,
        "cam": {
            "available": True,
            "output_shape": cam_shape,
            "method": "Class Activation Mapping (Zhou et al. style), computed inside the graph",
            "interpretation": (
                "Shows WHERE evidence for the predicted class is concentrated. It does not "
                "explain WHY in biological terms, and a heatmap over a lesion is not proof "
                "the network reasoned about lesion morphology."
            ),
            "diagnostic_use": (
                "Attention on the plain background rather than the leaf would indicate "
                "shortcut learning, which is a real risk on a studio dataset."
            ),
        },
        "export_verification": {
            "keras_top1_agreement": agree,
            "max_abs_probability_deviation": dev,
            "test_macro_f1": m["macro_f1"],
            "checkpoint_load": load_info,
        },
    }

    for target in (outdir / "model_config.json", assets / "model_config.json"):
        target.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (assets / "labels.txt").write_text("\n".join(NORMALIZED), encoding="utf-8")

    print(f"\nwritten to {outdir} and {assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
