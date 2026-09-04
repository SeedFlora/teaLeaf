#!/usr/bin/env bash
# Determine whether a mobile vision transformer is available with pretrained weights
# AND a working TFLite export. The brief allows a mobile transformer as a candidate
# but forbids adopting an architecture that cannot be exported reliably, so this
# decides by measurement rather than by reputation.
set -uo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

echo "=== installing keras-hub and the current LiteRT runtime ==="
pip install -q keras-hub ai-edge-litert 2>&1 | tail -3

python - <<'PY'
import json
import numpy as np

result = {}

for mod in ("keras_hub", "ai_edge_litert"):
    try:
        m = __import__(mod)
        result[mod] = getattr(m, "__version__", "importable")
    except Exception as exc:  # noqa: BLE001
        result[mod] = f"FAILED {type(exc).__name__}"
print(json.dumps(result, indent=2))

try:
    import keras_hub
    names = [n for n in dir(keras_hub.models) if "obilevit" in n.lower() or "MobileViT" in n]
    print("MobileViT symbols in keras_hub.models:", names[:10])

    # Enumerate whichever preset names the installed version advertises.
    presets = []
    for attr in ("MobileViTImageClassifier", "MobileViTBackbone"):
        cls = getattr(keras_hub.models, attr, None)
        if cls is not None and hasattr(cls, "presets"):
            presets = sorted(cls.presets.keys())
            print(f"{attr} presets:", presets[:12])
            break
    if not names:
        print("VERDICT: no MobileViT implementation in this keras_hub build")
except Exception as exc:  # noqa: BLE001
    print("keras_hub probe failed:", type(exc).__name__, str(exc)[:200])
PY
