#!/usr/bin/env bash
# Establish WHY the GPU matmul differs from NumPy, rather than assuming it is benign.
#
# Ampere GPUs default to TF32 for float32 matmul: inputs are rounded to a 10-bit
# mantissa before multiplication. That produces a relative error near 1e-3, which
# fails a tight absolute tolerance on values of magnitude ~16 while being entirely
# correct behaviour. This script distinguishes that from an actual fault by
# measuring the error and then re-running with TF32 disabled.
set -uo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

python - <<'PY'
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
x = rng.standard_normal((256, 256)).astype(np.float32)
ref64 = x.astype(np.float64) @ x.astype(np.float64)      # high-precision reference
cpu32 = x @ x

def report(tag, got):
    err = np.abs(got.astype(np.float64) - ref64)
    rel = err / np.maximum(np.abs(ref64), 1e-6)
    print(f"{tag:<28} max_abs={err.max():.3e}  max_rel={rel.max():.3e}  "
          f"mean_abs={err.mean():.3e}")

print("reference magnitude          :", f"max|ref|={np.abs(ref64).max():.3f}")
report("numpy float32 (CPU)", cpu32)

with tf.device("/GPU:0"):
    gpu_default = tf.matmul(tf.constant(x), tf.constant(x)).numpy()
report("TF GPU (default / TF32)", gpu_default)

# Disable TF32 and repeat. If the error collapses to the CPU float32 level, the
# earlier mismatch was TF32 reduced precision and nothing is wrong.
tf.config.experimental.enable_tensor_float_32_execution(False)
with tf.device("/GPU:0"):
    gpu_fp32 = tf.matmul(tf.constant(x), tf.constant(x)).numpy()
report("TF GPU (TF32 disabled)", gpu_fp32)

cpu_like = np.allclose(gpu_fp32, cpu32, rtol=1e-5, atol=1e-4)
print()
print("TF32-disabled GPU matches CPU float32 :", cpu_like)
print("VERDICT:", "TF32 reduced precision, expected and benign" if cpu_like
      else "UNEXPECTED - GPU numerics differ beyond TF32 explanation")

# Re-enable TF32: it is the right default for training throughput. Export and
# parity checks later run against the TFLite runtime, not against these kernels.
tf.config.experimental.enable_tensor_float_32_execution(True)
PY
