"""Establish which candidate backbones are actually deployable before committing to a benchmark.

The brief forbids selecting an architecture merely because it is newer. For each
candidate this checks the things that decide whether it can ship: a stable
implementation with ImageNet weights, an input convention we can reproduce
byte-for-byte on Android, parameter count, and a successful TFLite conversion.

The input-convention check matters most. Keras backbones disagree about whether they
want [0,1], [-1,1] or ImageNet-normalized input, and a mismatch between training and
the Android app degrades accuracy silently, with no error anywhere. Modern Keras
embeds preprocessing inside the graph, so the model consumes raw [0,255] pixels and
the normalization travels into the .tflite file. That removes the whole bug class --
but only if it is verified per model rather than assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

CANDIDATES = [
    ("MobileNetV3Large", dict(include_preprocessing=True)),
    ("MobileNetV3Small", dict(include_preprocessing=True)),
    ("EfficientNetB0", dict()),
    ("EfficientNetV2B0", dict(include_preprocessing=True)),
    ("ConvNeXtTiny", dict()),
    ("MobileNetV2", dict()),
]

IMG = 224
NUM_CLASSES = 7


def probe(name: str, extra: dict) -> dict:
    row: dict = {"name": name, "requested_kwargs": {k: str(v) for k, v in extra.items()}}
    try:
        ctor = getattr(tf.keras.applications, name)
    except AttributeError:
        row["available"] = False
        row["error"] = "not present in tf.keras.applications"
        return row
    row["available"] = True

    # Build the feature extractor with ImageNet weights.
    try:
        try:
            base = ctor(include_top=False, weights="imagenet",
                        input_shape=(IMG, IMG, 3), **extra)
            row["kwargs_accepted"] = True
        except TypeError:
            base = ctor(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
            row["kwargs_accepted"] = False
            row["note_kwargs"] = "include_preprocessing not supported; falls back to default"
    except Exception as exc:  # noqa: BLE001
        row["weights_loaded"] = False
        row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return row

    row["weights_loaded"] = True
    row["backbone_params"] = int(base.count_params())
    row["output_shape"] = [int(d) if d is not None else None for d in base.output_shape]

    # Probe the expected input range: feed raw [0,255] and see whether activations
    # look sane, then compare against a [0,1] feed. A model with embedded
    # preprocessing responds very differently to the two.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(2, IMG, IMG, 3)).astype("float32")
    try:
        out255 = base(img, training=False).numpy()
        out01 = base(img / 255.0, training=False).numpy()
        row["activation_std_input_0_255"] = round(float(out255.std()), 5)
        row["activation_std_input_0_1"] = round(float(out01.std()), 5)
        row["outputs_differ_between_conventions"] = bool(
            not np.allclose(out255, out01, atol=1e-4))
    except Exception as exc:  # noqa: BLE001
        row["forward_pass_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"

    # Full classifier head, then a TFLite conversion to prove exportability.
    try:
        inp = tf.keras.Input(shape=(IMG, IMG, 3), dtype="float32")
        x = base(inp, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = tf.keras.Model(inp, out)
        row["total_params"] = int(model.count_params())

        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tfl = conv.convert()
        row["tflite_fp32_bytes"] = len(tfl)
        row["tflite_fp32_mb"] = round(len(tfl) / 1e6, 2)
        row["export_ok"] = True

        # Verify the exported graph actually runs and agrees with Keras.
        interp = tf.lite.Interpreter(model_content=tfl)
        interp.allocate_tensors()
        di, do = interp.get_input_details()[0], interp.get_output_details()[0]
        row["tflite_input_shape"] = [int(v) for v in di["shape"]]
        row["tflite_input_dtype"] = str(np.dtype(di["dtype"]))
        row["tflite_output_shape"] = [int(v) for v in do["shape"]]

        one = img[:1]
        interp.set_tensor(di["index"], one)
        interp.invoke()
        tfl_out = interp.get_tensor(do["index"])
        keras_out = model(one, training=False).numpy()
        row["export_max_abs_deviation"] = float(np.abs(tfl_out - keras_out).max())
        row["export_parity_ok"] = bool(row["export_max_abs_deviation"] < 1e-4)
    except Exception as exc:  # noqa: BLE001
        row["export_ok"] = False
        row["export_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    tf.keras.backend.clear_session()
    return row


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("architecture_probe.json")
    print(f"TensorFlow {tf.__version__}\n")

    results = []
    for name, extra in CANDIDATES:
        print(f"--- probing {name} ---", flush=True)
        r = probe(name, extra)
        results.append(r)
        if r.get("export_ok"):
            print(f"    params={r['total_params']:,}  tflite={r['tflite_fp32_mb']} MB  "
                  f"parity_dev={r['export_max_abs_deviation']:.2e}  "
                  f"input={r['tflite_input_dtype']}{r['tflite_input_shape']}")
        else:
            print(f"    FAILED: {r.get('export_error') or r.get('error')}")

    # Is MobileViT reachable at all in this environment?
    mobilevit = {"checked": "keras_cv / keras_hub MobileViT availability"}
    for mod in ("keras_cv", "keras_hub"):
        try:
            __import__(mod)
            mobilevit[mod] = "importable"
        except Exception as exc:  # noqa: BLE001
            mobilevit[mod] = f"not installed ({type(exc).__name__})"
    results.append({"name": "_mobilevit_availability", **mobilevit})

    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
