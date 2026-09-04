#!/usr/bin/env bash
# Activate the training environment with CUDA discoverable.
#
# tensorflow[and-cuda] ships the CUDA runtime inside the Python package tree
# (site-packages/nvidia/<component>/lib) rather than installing it system-wide.
# The dynamic loader does not search those directories, so TensorFlow reports
# "Cannot dlopen some GPU libraries" and silently falls back to CPU even though
# every library is present. Putting them on LD_LIBRARY_PATH is the fix.
#
# Usage:  source environment/activate_tf.sh
#   or:   bash -c 'source .../activate_tf.sh && python train.py'

VENV="${TEA_VENV:-$HOME/tea_ws/venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
NV="$SITE/nvidia"

CUDA_PATHS=""
if [ -d "$NV" ]; then
  for d in "$NV"/*/lib; do
    [ -d "$d" ] && CUDA_PATHS="${CUDA_PATHS:+$CUDA_PATHS:}$d"
  done
fi

# /usr/lib/wsl/lib holds the WSL-provided libcuda.so.1 driver stub.
export LD_LIBRARY_PATH="${CUDA_PATHS}:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ptxas lives in the nvcc package; XLA needs it on PATH for JIT compilation.
if [ -d "$NV/cuda_nvcc/bin" ]; then
  export PATH="$NV/cuda_nvcc/bin:$PATH"
fi

# Grow GPU memory on demand instead of pre-allocating the whole 8 GB, so that a
# second process (e.g. an export job) can share the device.
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Quieten the routine oneDNN/absl startup chatter; warnings and errors still show.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
