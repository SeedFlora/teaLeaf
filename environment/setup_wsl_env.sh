#!/usr/bin/env bash
# Create the reproducible Python training environment inside WSL2 (Ubuntu).
# TensorFlow is installed with its bundled CUDA wheels so that the RTX 3070 Ti
# is usable; TensorFlow dropped native Windows GPU support after 2.10, which is
# why training runs under WSL2 rather than Windows.
set -euo pipefail

VENV="$HOME/tea_ws/venv"
OUT="$HOME/tea_ws/env_logs"
mkdir -p "$OUT"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel >"$OUT/pip_bootstrap.log" 2>&1

python -m pip install \
  "tensorflow[and-cuda]" \
  numpy pandas scipy scikit-learn scikit-image \
  pillow imagehash opencv-python-headless \
  matplotlib seaborn tqdm statsmodels \
  pyyaml pytest >"$OUT/pip_install.log" 2>&1

python -m pip freeze >"$OUT/requirements-lock.txt"

echo "=== versions ==="
python - <<'PY'
import importlib, platform
print("python", platform.python_version())
for m in ["tensorflow", "numpy", "pandas", "sklearn", "skimage", "PIL", "cv2", "imagehash", "scipy", "statsmodels"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "n/a"))
    except Exception as exc:  # noqa: BLE001
        print(m, "IMPORT_FAILED", exc)
PY

echo "=== GPU visibility ==="
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("tf", tf.__version__, "gpus", gpus)
print("built_with_cuda", tf.test.is_built_with_cuda())
if gpus:
    a = tf.random.normal((2048, 2048))
    print("matmul_ok", float(tf.reduce_sum(tf.matmul(a, a))) == float(tf.reduce_sum(tf.matmul(a, a))))
    print("device_used", tf.matmul(a, a).device)
PY
