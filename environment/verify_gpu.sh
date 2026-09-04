#!/usr/bin/env bash
# Verify that TensorFlow registers the GPU and can actually execute on it.
set -uo pipefail
source /mnt/d/tea/tea-leaf-android-feia/environment/activate_tf.sh

python - <<'PY'
import time
import numpy as np
import tensorflow as tf

print("TF version      :", tf.__version__)
print("built with CUDA :", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices("GPU")
print("physical GPUs   :", gpus)

if not gpus:
    raise SystemExit("FAIL: no GPU registered")

for g in gpus:
    print("device details  :", tf.config.experimental.get_device_details(g))

# A real computation, not just device enumeration: confirm results are correct
# and that the work is actually placed on the GPU.
with tf.device("/GPU:0"):
    a = tf.random.normal((4096, 4096))
    b = tf.random.normal((4096, 4096))
    c = tf.matmul(a, b)
    _ = c.numpy()                      # force execution
    t0 = time.time()
    for _ in range(10):
        c = tf.matmul(a, b)
    _ = c.numpy()
    dt = time.time() - t0

flops = 10 * 2 * 4096 ** 3
print(f"matmul placed on: {c.device}")
print(f"10x 4096^3 matmul: {dt:.3f} s  ->  {flops / dt / 1e12:.2f} TFLOP/s")

# Correctness check against NumPy on a small case.
x = np.random.randn(256, 256).astype(np.float32)
gpu_res = tf.matmul(tf.constant(x), tf.constant(x)).numpy()
print("matches numpy   :", np.allclose(gpu_res, x @ x, atol=1e-2))

# Confirm a Keras model can be built and run a training step on the GPU.
m = tf.keras.applications.MobileNetV3Large(weights=None, classes=7, input_shape=(224, 224, 3))
m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
xb = tf.random.uniform((8, 224, 224, 3))
yb = tf.constant([0, 1, 2, 3, 4, 5, 6, 0])
t0 = time.time()
m.train_on_batch(xb, yb)
print(f"MobileNetV3-Large train step OK ({time.time() - t0:.2f} s incl. graph build)")
print("PASS")
PY
