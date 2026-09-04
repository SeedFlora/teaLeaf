#!/usr/bin/env bash
# Diagnose why TensorFlow does not register a GPU device inside WSL2.
set -uo pipefail
VENV="$HOME/tea_ws/venv"
source "$VENV/bin/activate"

echo "=== 1. nvidia-smi inside WSL ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3

echo
echo "=== 2. WSL CUDA driver stub present? ==="
ls -la /usr/lib/wsl/lib/libcuda* 2>&1 | head -5

echo
echo "=== 3. nvidia pip packages installed ==="
pip list 2>/dev/null | grep -i nvidia

echo
echo "=== 4. Where the bundled CUDA shared objects actually live ==="
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "site-packages: $SITE"
find "$SITE/nvidia" -name "*.so*" -maxdepth 3 2>/dev/null | sed 's|.*/nvidia/||' | cut -d/ -f1-2 | sort -u | head -25

echo
echo "=== 5. Which libraries TensorFlow fails to dlopen (verbose) ==="
TF_CPP_MIN_LOG_LEVEL=0 python -c "
import tensorflow as tf
print('TF', tf.__version__)
print('built_with_cuda', tf.test.is_built_with_cuda())
print('GPUs', tf.config.list_physical_devices('GPU'))
" 2>&1 | grep -iE "dlopen|cannot|could not|missing|libcud|libnv|successful|GPUs|TF |built_with" | head -30

echo
echo "=== 6. Current LD_LIBRARY_PATH ==="
echo "${LD_LIBRARY_PATH:-<unset>}"

echo
echo "=== 7. cudnn version present ==="
find "$SITE/nvidia" -name "libcudnn*.so*" 2>/dev/null | head -5
