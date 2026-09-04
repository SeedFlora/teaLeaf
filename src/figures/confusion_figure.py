"""Figure: confusion matrices for the deployed model and for its full-integer INT8 export.

Both matrices come from tables/export/parity_analysis.json, i.e. from per-image
predictions on the same frozen test partition, so the two panels are directly comparable
and the class whose recall collapses under quantization is visible as a row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT = ["Algal", "Brown", "Gray", "Helop.", "R. spider", "Mirid", "Healthy"]


def panel(ax, cm: np.ndarray, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    k = len(SHORT)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5.2,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(SHORT, fontsize=5.2, rotation=45, ha="right")
    ax.set_yticklabels(SHORT, fontsize=5.2)
    ax.set_xlabel("predicted", fontsize=6, labelpad=1)
    ax.set_ylabel("true", fontsize=6, labelpad=1)
    ax.set_title(title, fontsize=6.5, pad=3)
    ax.tick_params(length=1.5, pad=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = json.loads(Path(a.parity).read_text(encoding="utf-8"))

    ref = p["reference"]
    int8 = p["variants"]["int8"]
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.45))
    panel(axes[0], ref["confusion_matrix"],
          f"(a) MobileNetV3-Large, FP32 (macro-F1 {ref['macro_f1']:.4f})")
    panel(axes[1], int8["confusion_matrix"],
          f"(b) full-integer INT8 (macro-F1 {int8['macro_f1']:.4f})")
    fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.21, wspace=0.42)
    fig.savefig(outdir / "fig_confusion.pdf")
    fig.savefig(outdir / "fig_confusion.jpg", dpi=300)
    plt.close(fig)
    print(f"written: {outdir/'fig_confusion.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
