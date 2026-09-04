"""Dataset bias audit: class balance, resolution, capture metadata and quality confounds.

The question behind each check is the same: could a model reach high accuracy by
reading something other than the leaf? A resolution, EXIF-device or file-size cue
that correlates with the label is a shortcut, and a benchmark that ignores it
reports the shortcut's accuracy rather than the task's.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(Path(a.manifest).open(encoding="utf-8")))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    classes = sorted({r["class_folder_raw"] for r in rows})
    counts = Counter(r["class_folder_raw"] for r in rows)
    n = len(rows)

    # ---- class distribution ---- #
    with (outdir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_folder_raw", "normalized_class", "n_images", "pct_of_dataset"])
        norm = {r["class_folder_raw"]: r["normalized_class"] for r in rows}
        for c in classes:
            w.writerow([c, norm[c], counts[c], round(100 * counts[c] / n, 3)])

    imbalance = max(counts.values()) / min(counts.values())

    # ---- resolution / aspect / size per class ---- #
    per_class: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["class_folder_raw"]
        per_class[c]["w"].append(int(r["width"]))
        per_class[c]["h"].append(int(r["height"]))
        per_class[c]["bytes"].append(int(r["size_bytes"]))
        per_class[c]["blur"].append(float(r["laplacian_var_blur"]))
        per_class[c]["bright"].append(float(r["mean_brightness"]))
        per_class[c]["contrast"].append(float(r["std_contrast"]))

    res_rows = []
    for c in classes:
        d = per_class[c]
        dims = Counter(zip(d["w"], d["h"]))
        res_rows.append({
            "class_folder_raw": c,
            "n": counts[c],
            "distinct_resolutions": len(dims),
            "most_common_resolution": f"{dims.most_common(1)[0][0][0]}x{dims.most_common(1)[0][0][1]}",
            "most_common_resolution_pct": round(100 * dims.most_common(1)[0][1] / counts[c], 2),
            "median_bytes": int(st.median(d["bytes"])),
            "median_blur_laplacian_var": round(st.median(d["blur"]), 1),
            "median_brightness": round(st.median(d["bright"]), 1),
            "median_contrast": round(st.median(d["contrast"]), 1),
        })

    with (outdir / "class_resolution_quality.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res_rows[0]))
        w.writeheader()
        w.writerows(res_rows)

    # ---- global resolution / EXIF picture ---- #
    all_dims = Counter((int(r["width"]), int(r["height"])) for r in rows)
    exif_present = sum(1 for r in rows if r["has_exif"] == "True")
    makes = Counter(r["exif_make"] for r in rows if r["exif_make"])
    models = Counter(r["exif_model"] for r in rows if r["exif_model"])
    orientations = Counter(r["exif_orientation"] for r in rows if r["exif_orientation"])
    modes = Counter(r["color_mode"] for r in rows)
    fmts = Counter(r["detected_format"] for r in rows)
    sharp = Counter(r["sharpness_flag"] for r in rows)

    # Device-label correlation is only meaningful if devices are recorded at all.
    device_by_class: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["exif_model"]:
            device_by_class[r["class_folder_raw"]][r["exif_model"]] += 1

    # ---- image-quality table ---- #
    with (outdir / "image_quality.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "class_folder_raw", "width", "height", "size_bytes",
                    "laplacian_var_blur", "sharpness_flag", "mean_brightness", "std_contrast",
                    "decode_ok", "corruption_status", "has_exif"])
        for r in rows:
            w.writerow([r["relative_path"], r["class_folder_raw"], r["width"], r["height"],
                        r["size_bytes"], r["laplacian_var_blur"], r["sharpness_flag"],
                        r["mean_brightness"], r["std_contrast"], r["decode_ok"],
                        r["corruption_status"], r["has_exif"]])

    summary = {
        "images": n,
        "classes": len(classes),
        "class_counts": {c: counts[c] for c in classes},
        "imbalance_ratio_max_over_min": round(imbalance, 3),
        "resolution": {
            "distinct_resolutions_dataset_wide": len(all_dims),
            "top_resolutions": {f"{w_}x{h_}": c for (w_, h_), c in all_dims.most_common(6)},
            "readme_claimed_resolution": "1200x1600",
            "resolution_is_label_informative": len(all_dims) > 1 and any(
                r["most_common_resolution_pct"] < 99.0 for r in res_rows),
        },
        "exif": {
            "images_with_exif": exif_present,
            "pct_with_exif": round(100 * exif_present / n, 2),
            "distinct_makes": dict(makes.most_common(10)),
            "distinct_models": dict(models.most_common(10)),
            "orientations": dict(orientations),
            "device_label_correlation_testable": bool(models),
        },
        "encoding": {"color_modes": dict(modes), "formats": dict(fmts)},
        "quality": {
            "sharpness_flags": dict(sharp),
            "decode_failures": sum(1 for r in rows if r["decode_ok"] != "True"),
            "corrupt_or_flagged": sum(1 for r in rows if r["corruption_status"] != "ok"),
        },
        "per_class_device_counts": {c: dict(v) for c, v in device_by_class.items()},
        "notes": (
            "A shortcut cue is only a threat if it varies with the label. Resolution, encoding "
            "and EXIF fields are therefore reported per class, not only dataset-wide. Where EXIF "
            "is absent the device-label correlation simply cannot be tested, and that is reported "
            "as untestable rather than as absence of bias."
        ),
    }

    (outdir / "bias_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"images={n}  classes={len(classes)}  imbalance={imbalance:.2f}x")
    print(f"class counts: {dict(counts)}")
    print(f"distinct resolutions dataset-wide: {len(all_dims)}")
    print(f"  top: {dict(all_dims.most_common(5))}")
    print(f"EXIF present: {exif_present}/{n} ({100*exif_present/n:.1f}%)")
    print(f"  makes: {dict(makes.most_common(5))}")
    print(f"  models: {dict(models.most_common(5))}")
    print(f"color modes: {dict(modes)}  formats: {dict(fmts)}")
    print(f"sharpness: {dict(sharp)}")
    print("\nper-class resolution/quality:")
    for r in res_rows:
        print(f"  {r['class_folder_raw']:<24} n={r['n']:<5} res={r['most_common_resolution']:<10}"
              f"({r['most_common_resolution_pct']}%) distinct={r['distinct_resolutions']:<4}"
              f" blur={r['median_blur_laplacian_var']:<8} bright={r['median_brightness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
