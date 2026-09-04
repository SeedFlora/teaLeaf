"""Train one backbone on one split with one seed, recording everything needed to reproduce it.

Design decisions worth stating, because they determine whether the numbers mean anything:

* The network consumes raw [0,255] pixels. Every backbone used here embeds its own
  normalization inside the graph, so that normalization is exported into the .tflite
  file and the Android app can feed camera pixels directly. Preprocessing parity stops
  being something to maintain in two places.
* Augmentation is applied on GPU, to the training partition only, from a 256px cache.
  Validation and test read a 224px cache produced by resizing the original image
  directly -- the exact operation the phone performs.
* Checkpoint selection uses validation macro-F1, matching the primary reported metric.
  Accuracy would over-reward the two large classes under a 3.07x imbalance.
* The test partition is not read during training. It is scored once, afterwards.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_ORDER = [
    "1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight", "4. Helopeltis",
    "5. Red spider", "6. Green mirid bug", "7. Healthy leaf",
]
NUM_CLASSES = len(CLASS_ORDER)

BACKBONES = {
    "mobilenetv3_large": ("MobileNetV3Large", dict(include_preprocessing=True)),
    "mobilenetv3_small": ("MobileNetV3Small", dict(include_preprocessing=True)),
    "mobilenetv2": ("MobileNetV2", dict()),
    "efficientnetb0": ("EfficientNetB0", dict()),
    "efficientnetv2b0": ("EfficientNetV2B0", dict(include_preprocessing=True)),
    "convnext_tiny": ("ConvNeXtTiny", dict()),
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def macro_f1_from_counts(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Select and early-stop on validation macro-F1, the primary reported metric.

    Keras cannot compute macro-F1 for sparse integer labels as a streaming metric, and
    a batch-averaged approximation is biased for rare classes, so it is computed
    exactly over the full validation set each epoch.
    """

    def __init__(self, core, xv, yv, ckpt_path: Path, patience: int, log_path: Path, batch: int = 64):
        super().__init__()
        # Validation is scored on the CORE graph at the evaluation resolution, never on
        # the augmentation-wrapped training graph, and it is the core graph that is
        # checkpointed -- so the saved artifact is exactly what gets exported.
        self.core = core
        self.xv, self.yv = xv, yv
        self.ckpt_path = ckpt_path
        self.patience = patience
        self.log_path = log_path
        self.batch = batch
        self.best = -1.0
        self.best_epoch = -1
        self.wait = 0
        self.history: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        probs = self.core.predict(self.xv, batch_size=self.batch, verbose=0)
        pred = probs.argmax(axis=1)
        f1 = macro_f1_from_counts(self.yv, pred)
        acc = float(np.mean(pred == self.yv))
        logs["val_macro_f1"] = f1
        logs["val_exact_accuracy"] = acc

        row = {"epoch": int(epoch), **{k: float(v) for k, v in logs.items()}}
        self.history.append(row)
        self.log_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

        if f1 > self.best:
            self.best, self.best_epoch, self.wait = f1, epoch, 0
            self.core.save(self.ckpt_path)
            marker = " *best*"
        else:
            self.wait += 1
            marker = ""
        print(f"    epoch {epoch:>3}  val_macroF1={f1:.4f}  val_acc={acc:.4f}"
              f"  loss={logs.get('loss', float('nan')):.4f}{marker}", flush=True)

        if self.wait >= self.patience:
            print(f"    early stopping: no macro-F1 improvement for {self.patience} epochs")
            self.model.stop_training = True


def build_augmenter(train_size: int, eval_size: int, seed: int) -> tf.keras.Sequential:
    """Moderate, biologically plausible augmentation.

    Deliberately excluded: vertical flips (leaves have a consistent tip-to-base
    orientation in this dataset) and strong color jitter (lesion hue carries the
    class signal -- algal leaf spot and brown blight are distinguished partly by
    color, so shifting it would relabel the image).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),
        tf.keras.layers.RandomZoom(0.15, 0.15, fill_mode="reflect", seed=seed),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        tf.keras.layers.RandomContrast(0.15, seed=seed),
        tf.keras.layers.Resizing(eval_size, eval_size, interpolation="bilinear"),
    ], name="augmentation")


def build_model(backbone_key: str, img_size: int, train_size: int, dropout: float, seed: int
                ) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Return (train_model, core_model, backbone).

    `core_model` takes 224px pixels and is the graph that gets exported. `train_model`
    wraps it with a 256px input and the augmentation stack, so augmentation executes on
    the GPU as part of the forward pass instead of on the CPU inside tf.data -- which
    was leaving the GPU idle for most of each step.

    Because `core_model` is a genuine sub-model, the two share weights: training
    `train_model` trains `core_model`, while the exported graph never contains a single
    augmentation op.
    """
    name, extra = BACKBONES[backbone_key]
    ctor = getattr(tf.keras.applications, name)
    try:
        base = ctor(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3), **extra)
    except TypeError:
        base = ctor(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))

    inp = tf.keras.Input(shape=(img_size, img_size, 3), dtype="float32", name="pixels_0_255")
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    # float32 output regardless of mixed-precision policy, so probabilities are exact.
    logits = tf.keras.layers.Dense(NUM_CLASSES, name="logits", dtype="float32")(x)
    probs = tf.keras.layers.Activation("softmax", dtype="float32", name="probs")(logits)
    core = tf.keras.Model(inp, probs, name=f"{backbone_key}_classifier")

    aug_in = tf.keras.Input(shape=(train_size, train_size, 3), dtype="float32", name="aug_pixels")
    aug_out = build_augmenter(train_size, img_size, seed)(aug_in)
    train_model = tf.keras.Model(aug_in, core(aug_out), name=f"{backbone_key}_train")

    return train_model, core, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-prefix", default="primary")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=35)
    ap.add_argument("--warmup-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--mixed-precision", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)

    set_seeds(a.seed)
    if a.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    xtr = np.load(cache / f"{a.split_prefix}_train_x.npy")
    ytr = np.load(cache / f"{a.split_prefix}_train_y.npy")
    xva = np.load(cache / f"{a.split_prefix}_validation_x.npy")
    yva = np.load(cache / f"{a.split_prefix}_validation_y.npy")
    print(f"train {xtr.shape} val {xva.shape}", flush=True)

    # tf.data only casts and batches; augmentation lives inside the model so it runs
    # on the GPU. A bounded shuffle buffer avoids holding a second copy of the 726 MB
    # training array in host memory.
    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr))
          .shuffle(min(4096, len(xtr)), seed=a.seed, reshuffle_each_iteration=True)
          .batch(a.batch_size)
          .map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
          .prefetch(tf.data.AUTOTUNE))

    xva_f = xva.astype("float32")

    class_weight = None
    if a.class_weight == "balanced":
        counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float64)
        w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
        class_weight = {i: float(w[i]) for i in range(NUM_CLASSES)}

    model, core, base = build_model(a.backbone, a.img_size, xtr.shape[1], a.dropout, a.seed)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=a.label_smoothing) \
        if a.label_smoothing > 0 else tf.keras.losses.SparseCategoricalCrossentropy()

    ckpt = outdir / "best.keras"
    log_path = outdir / "epoch_log.json"
    t0 = time.time()

    # Stage 1: train the head with the backbone frozen, so random head gradients do
    # not destroy the pretrained features in the first few steps.
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(a.warmup_lr), loss=loss, metrics=["accuracy"])
    cb = MacroF1Checkpoint(core, xva_f, yva, ckpt, patience=a.patience, log_path=log_path)
    print(f"=== stage 1: head warmup, {a.warmup_epochs} epochs ===", flush=True)
    model.fit(ds, epochs=a.warmup_epochs, callbacks=[cb], class_weight=class_weight, verbose=0)
    warmup_best = cb.best

    # Stage 2: unfreeze and fine-tune the whole network at a lower learning rate.
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(a.finetune_lr), loss=loss, metrics=["accuracy"])
    cb2 = MacroF1Checkpoint(core, xva_f, yva, ckpt, patience=a.patience, log_path=log_path)
    cb2.best = cb.best
    cb2.history = cb.history
    print(f"=== stage 2: fine-tune, up to {a.finetune_epochs} epochs ===", flush=True)
    model.fit(ds, epochs=a.finetune_epochs, callbacks=[cb2], class_weight=class_weight, verbose=0)

    wall = time.time() - t0

    config = {
        "backbone": a.backbone,
        "keras_application": BACKBONES[a.backbone][0],
        "split_prefix": a.split_prefix,
        "seed": a.seed,
        "img_size": a.img_size,
        "batch_size": a.batch_size,
        "warmup_epochs": a.warmup_epochs,
        "finetune_epochs_max": a.finetune_epochs,
        "warmup_lr": a.warmup_lr,
        "finetune_lr": a.finetune_lr,
        "dropout": a.dropout,
        "label_smoothing": a.label_smoothing,
        "early_stopping_patience": a.patience,
        "class_weight": a.class_weight,
        "mixed_precision": a.mixed_precision,
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy",
        "checkpoint_selection": "max validation macro-F1",
        "pretrained_weights": "ImageNet (keras.applications)",
        "input_convention": "raw [0,255] float32; normalization embedded in the backbone graph",
        "augmentation": [
            "RandomFlip horizontal", "RandomRotation 0.05", "RandomZoom 0.15",
            "RandomTranslation 0.08", "RandomBrightness 0.15", "RandomContrast 0.15",
            f"Resizing to {a.img_size} bilinear",
        ],
        "augmentation_excluded": [
            "vertical flip (leaves have consistent tip-to-base orientation)",
            "strong color jitter (lesion hue is class-discriminative)",
        ],
        "best_val_macro_f1": cb2.best,
        "best_epoch": cb2.best_epoch,
        "warmup_stage_best_val_macro_f1": warmup_best,
        "epochs_run": len(cb2.history),
        "wall_clock_seconds": round(wall, 1),
        "total_params": int(core.count_params()),
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "gpus": [d.name for d in gpus],
        "train_n": int(len(xtr)),
        "val_n": int(len(xva)),
        "class_order": CLASS_ORDER,
    }
    (outdir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nbest val macro-F1 = {cb2.best:.4f} at epoch {cb2.best_epoch}")
    print(f"wall clock {wall/60:.1f} min over {len(cb2.history)} epochs")
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

