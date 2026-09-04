"""Figure: macro-F1, coverage and selective risk under each corruption as severity rises.

Read from tables/robustness/robustness.json. The coverage and selective-risk panels use the
abstention threshold selected on clean validation data and applied unchanged, which is the
question Reviewer 2 raised: does a clean-data threshold remain appropriate under corruption?
The dashed line marks the 5% selective-risk policy ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {
    "brightness_down": "brightness −", "brightness_up": "brightness +", "contrast_down": "contrast −",
    "gaussian_blur": "Gaussian blur", "motion_blur": "motion blur", "jpeg_compression": "JPEG",
    "rotation": "rotation", "crop_zoom": "crop / zoom",
}
STYLE = {
    "brightness_down": ("tab:orange", "o", "-"), "brightness_up": ("tab:orange", "s", "--"),
    "contrast_down": ("tab:brown", "^", "-"), "gaussian_blur": ("tab:blue", "o", "-"),
    "motion_blur": ("tab:blue", "s", "--"), "jpeg_compression": ("tab:purple", "D", "-"),
    "rotation": ("tab:green", "o", "-"), "crop_zoom": ("tab:green", "s", "--"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    r = json.loads(Path(a.robustness).read_text(encoding="utf-8"))
    pol = r["abstention_policy"]
    clean = r["clean"]
    clean_ab = r["clean_abstention"]

    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.75))
    for name, sev in r["corruptions"].items():
        col, mk, ls = STYLE[name]
        xs = [0, 1, 2, 3]
        f1 = [clean["macro_f1"]] + [sev[f"severity_{s}"]["macro_f1"] for s in (1, 2, 3)]
        cov = [clean_ab["coverage"]] + [sev[f"severity_{s}"]["abstention"]["coverage"] for s in (1, 2, 3)]
        risk = [clean_ab["selective_risk"]] + [sev[f"severity_{s}"]["abstention"]["selective_risk"] for s in (1, 2, 3)]
        for ax, ys in zip(axes, (f1, cov, risk)):
            ax.plot(xs, ys, color=col, marker=mk, linestyle=ls, markersize=2.2, linewidth=0.8,
                    label=LABEL[name])
    axes[2].axhline(pol["target_selective_risk"], color="black", linestyle=":", linewidth=0.7)
    axes[2].text(3.0, pol["target_selective_risk"] + 0.012, "5% policy", fontsize=5, ha="right")
    for ax, title in zip(axes, ("macro-F1", "coverage at deployed threshold", "selective risk")):
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["clean", "1", "2", "3"], fontsize=5.5)
        ax.set_xlabel("severity", fontsize=6, labelpad=1)
        ax.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
        ax.tick_params(axis="x", length=2, pad=1)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_ylim(0.3, 0.92)
    axes[1].set_ylim(0.3, 0.92)
    axes[2].set_ylim(0.0, 0.42)
    axes[0].legend(fontsize=4.6, ncol=2, frameon=False, loc="lower left", handlelength=1.6,
                   columnspacing=0.8, labelspacing=0.25)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.2, wspace=0.32)
    fig.savefig(outdir / "fig_robustness.pdf")
    fig.savefig(outdir / "fig_robustness.jpg", dpi=300)
    plt.close(fig)
    print(f"written: {outdir/'fig_robustness.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
