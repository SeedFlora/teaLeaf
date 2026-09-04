"""Cross-check the on-device probabilities against the desktop runtime.

The instrumented test scores eight deterministic synthetic images and writes its
probabilities to the device. This script regenerates those exact images from the same
linear congruential recurrence, scores them with the desktop LiteRT runtime on the same
.tflite file, and compares.

A shared fixed input is what makes the comparison possible at all: without it there is
nothing to hold constant between the two runtimes, and any difference could be attributed
to the input rather than to the deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def generate_case(seed_idx: int, size: int = 224) -> np.ndarray:
    """Reproduce the Kotlin generator exactly.

    Kotlin: seed = (seed * 1103515245 + 12345) and 0x7FFFFFFF, iterated row-major, with
    pixel = rgb(seed % 256, (seed / 256) % 256, x % 256). Integer division on a positive
    Long matches Python's floor division here because the masked seed is never negative.
    """
    seed = 1000 + seed_idx * 7919
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            img[y, x, 0] = seed % 256
            img[y, x, 1] = (seed // 256) % 256
            img[y, x, 2] = x % 256
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-record", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    a = ap.parse_args()

    device = json.loads(Path(a.device_record).read_text(encoding="utf-8"))
    blob = Path(a.model).read_bytes()

    import hashlib
    desktop_sha = hashlib.sha256(blob).hexdigest()
    if desktop_sha != device["model_sha256"]:
        raise SystemExit(
            "the device and the desktop are not running the same model:\n"
            f"  device : {device['model_sha256']}\n  desktop: {desktop_sha}"
        )
    print(f"model digest matches on both sides: {desktop_sha[:16]}...")

    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_content=blob, num_threads=4)
        runtime = "ai_edge_litert"
    except Exception:
        interp = tf.lite.Interpreter(model_content=blob, num_threads=4)
        runtime = "tf.lite"
    interp.allocate_tensors()
    di = interp.get_input_details()[0]
    outs = interp.get_output_details()
    prob_idx = next(i for i, o in enumerate(outs) if len(o["shape"]) == 2)

    rows = []
    worst = 0.0
    disagreements = 0

    for case in device["cases"]:
        idx = case["case"]
        img = generate_case(idx).astype("float32")[None]
        interp.set_tensor(di["index"], img)
        interp.invoke()
        desk = interp.get_tensor(outs[prob_idx]["index"])[0].astype(np.float64)
        desk = desk / desk.sum()

        dev = np.array(case["probabilities_raw"], dtype=np.float64)
        dev = dev / dev.sum()

        max_dev = float(np.abs(desk - dev).max())
        worst = max(worst, max_dev)
        same_top1 = int(desk.argmax()) == int(case["top1_index"])
        if not same_top1:
            disagreements += 1

        rows.append({
            "case": idx,
            "device_top1": case["top1_index"],
            "desktop_top1": int(desk.argmax()),
            "top1_agrees": same_top1,
            "max_abs_probability_deviation": max_dev,
            "device_probabilities": dev.tolist(),
            "desktop_probabilities": desk.tolist(),
        })
        print(f"  case {idx}: device top1={case['top1_index']} desktop top1={int(desk.argmax())} "
              f"agree={same_top1}  max|dp|={max_dev:.3e}")

    result = {
        "device_runtime": "LiteRT on Android (org.tensorflow.lite.Interpreter, XNNPACK)",
        "desktop_runtime": runtime,
        "model_sha256": desktop_sha,
        "n_cases": len(rows),
        "top1_agreement": (len(rows) - disagreements) / len(rows) if rows else None,
        "top1_disagreements": disagreements,
        "worst_max_abs_probability_deviation": worst,
        "tolerance": a.tolerance,
        "within_tolerance": bool(worst <= a.tolerance),
        "cases": rows,
        "note": (
            "Both sides score byte-identical synthetic inputs regenerated from the same "
            "recurrence, so any deviation is attributable to the runtime rather than to "
            "the data."
        ),
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\ntop-1 agreement          : {result['top1_agreement']:.4f} "
          f"({len(rows) - disagreements}/{len(rows)})")
    print(f"worst probability deviation: {worst:.3e}  (tolerance {a.tolerance})")
    print(f"within tolerance          : {result['within_tolerance']}")
    print(f"\nwritten: {a.out}")
    return 0 if result["within_tolerance"] and disagreements == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
