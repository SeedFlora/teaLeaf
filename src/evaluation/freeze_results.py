"""Collect every frozen result into one file that the manuscript is written from.

Nothing in the paper is typed from memory or from a terminal scroll-back. Each reported
number is read here from the artifact that produced it, stamped with that file's path, and
written to reports/frozen_results.json. The manuscript then quotes only values present in
that file, so a claim without a source cannot survive into the text.

Revision additions: the three-seed benchmark summary and seed-matched comparisons, the
paired export analysis, the abstention audit under corruption, and the activation-map
sanity checks.
"""

from __future__ import annotations

import json
from pathlib import Path


def load(repo: Path, rel: str):
    p = repo / rel
    return (json.loads(p.read_text(encoding="utf-8")), rel) if p.exists() else (None, rel)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]

    mc, mc_src = load(repo, "tables/model_comparison.json")
    da, da_src = load(repo, "data/audit/duplicate_audit_summary.json")
    seq, seq_src = load(repo, "data/audit/sequence_adjacency_analysis.json")
    vd, vd_src = load(repo, "data/audit/version_diff_summary.json")
    prov, prov_src = load(repo, "data/audit/local_vs_official_verification.json")
    bias, bias_src = load(repo, "data/audit/bias_audit_summary.json")
    sp, sp_src = load(repo, "data/splits/split_summary.json")
    cal, cal_src = load(repo, "tables/calibration/calibration.json")
    rob, rob_src = load(repo, "tables/robustness/robustness.json")
    ex, ex_src = load(repo, "models/exported/export_report.json")
    par, par_src = load(repo, "tables/export/parity_analysis.json")
    cam, cam_src = load(repo, "tables/cam/cam_sanity.json")
    dev, dev_src = load(repo, "reports/device_benchmark/benchmark_report.json")
    dpar, dpar_src = load(repo, "reports/device_benchmark/parity_comparison.json")
    inv, inv_src = load(repo, "reports/device_inventory.json")

    out: dict = {"sources": {}, "dataset": {}, "split": {}, "benchmark": {},
                 "calibration": {}, "robustness": {}, "export": {}, "cam": {}, "device": {}}

    # ---------------- dataset ----------------
    if da:
        out["sources"]["duplicates"] = da_src
        out["dataset"].update(
            images=da["images"],
            exact_duplicate_pairs=da["exact_duplicate_groups"],
            duplicate_pairs_for_grouping=da["duplicate_pairs_used_for_grouping"],
            images_in_duplicate_groups=da["images_in_multi_image_groups"],
            pct_images_in_duplicate_groups=round(
                100 * da["images_in_multi_image_groups"] / da["images"], 3),
            largest_group_size=da["largest_group_size"],
            cross_label_groups=da["cross_label_groups"],
            operative_phash_threshold=da["thresholds"]["phash_operative_hamming_of_256"],
            thresholds=da["thresholds"],
            pairs_screened=da["pairs_screened_at_loose_distance"],
        )
    if seq:
        out["sources"]["sequence_adjacency"] = seq_src
        o = seq["operative"]
        out["dataset"].update(
            adjacent_pairs=o["same_class_with_sequence_gap_1"],
            adjacent_pairs_total=o["pairs"],
            pct_adjacent=o["pct_same_class_adjacent"],
            expected_adjacent_if_random=o["expected_adjacent_if_random"],
            observed_over_expected=o["observed_over_expected"],
            cross_class_pairs_at_operative=o["cross_class_pairs"],
        )
    if vd:
        out["sources"]["version_diff"] = vd_src
        out["dataset"].update(
            v3_files_listed=vd["v3_files_listed"],
            v4_files_listed=vd["v4_files_listed"],
            only_in_v4=vd["only_in_v4_paths"],
            only_in_v3=vd["only_in_v3_paths"],
            content_changed=vd["content_changed_paths"],
        )
    if prov:
        out["sources"]["provenance"] = prov_src
        out["dataset"].update(
            provenance_matched_by_path_and_hash=prov["v3"]["local_matched_by_path_and_hash"],
            provenance_hash_mismatches=prov["v3"]["local_path_present_but_hash_differs"],
            provenance_official_absent_locally=prov["v3"]["official_files_absent_locally"],
        )
    if bias:
        out["sources"]["bias"] = bias_src
        out["dataset"].update(
            class_counts=bias["class_counts"],
            imbalance_ratio=bias["imbalance_ratio_max_over_min"],
            distinct_resolutions=bias["resolution"]["distinct_resolutions_dataset_wide"],
            resolutions=bias["resolution"]["top_resolutions"],
            pct_with_exif=bias["exif"]["pct_with_exif"],
            capture_devices=bias["exif"]["distinct_models"],
        )

    # ---------------- split ----------------
    if sp:
        out["sources"]["split"] = sp_src
        out["split"] = {
            part: {"n": sp["primary_split"][part]["n"],
                   "fraction": sp["primary_split"][part]["fraction"],
                   "groups": sp["primary_split"][part]["n_groups"],
                   "per_class": sp["primary_split"][part]["per_class"]}
            for part in ("train", "validation", "test")
        }
        out["split"]["quarantined"] = sp["images_quarantined"]
        out["split"]["seed"] = sp["seed"]

    # ---------------- benchmark ----------------
    if mc:
        out["sources"]["benchmark"] = mc_src
        out["benchmark"]["runs"] = [{k: v for k, v in r.items() if k != "test_per_class_f1"}
                                    for r in mc["runs_table"]]
        out["benchmark"]["seed_spread"] = mc["seed_spread"]
        out["benchmark"]["multiseed"] = {
            bb: {"params": v["params"], "n_seeds": v["n_seeds"], "seeds": v["seeds"],
                 "test_macro_f1": v["test_macro_f1"], "test_accuracy": v["test_accuracy"],
                 "test_mcc": v["test_mcc"], "test_ece": v["test_ece"], "test_nll": v["test_nll"],
                 "test_per_class_f1": v["test_per_class_f1"]}
            for bb, v in mc["multiseed"].items()
        }
        out["benchmark"]["ranking"] = mc["ranking"]
        out["benchmark"]["leakage_sensitivity"] = mc["leakage_sensitivity"]
        out["benchmark"]["pairwise_mcnemar_basis"] = mc.get("pairwise_mcnemar_basis")
        out["benchmark"]["pairwise_mcnemar"] = {
            k: {"a_right_b_wrong": v["a_right_b_wrong"],
                "a_wrong_b_right": v["a_wrong_b_right"],
                "p_exact": v["p_value_exact_binomial"],
                "p_holm": v.get("holm_bonferroni", {}).get("p_holm")}
            for k, v in mc["pairwise_mcnemar"].items()
        }
        out["benchmark"]["seed_matched_mcnemar"] = {
            k: {"seeds": v["seeds"],
                "n_seeds_significant_raw_0.05": v["n_seeds_significant_raw_0.05"],
                "mean_macro_f1_a_minus_b_points": v["mean_macro_f1_a_minus_b_points"],
                "sign_consistent": v["sign_consistent"],
                "per_seed": v["per_seed"]}
            for k, v in mc["seed_matched_mcnemar"].items()
        }

    # ---------------- calibration ----------------
    if cal:
        out["sources"]["calibration"] = cal_src
        ts = cal["test_selective_performance"]
        out["calibration"] = {
            "temperature": cal["temperature"],
            "test_uncalibrated": {k: cal["test_uncalibrated"][k] for k in ("nll", "ece", "brier", "accuracy", "mean_confidence")},
            "test_calibrated": {k: cal["test_calibrated"][k] for k in ("nll", "ece", "brier", "accuracy", "mean_confidence")},
            "abstention_threshold": cal["abstention_policy"]["threshold"],
            "target_selective_risk": cal["abstention_policy"]["target_selective_risk"],
            "min_coverage": cal["abstention_policy"]["min_coverage"],
            "test_coverage": ts["coverage"],
            "test_selective_accuracy": ts["selective_accuracy"],
            "test_full_coverage_accuracy": ts["full_coverage_accuracy"],
            "n_abstained": ts["n_abstained"],
            "accuracy_on_abstained": ts["accuracy_on_abstained"],
        }

    # ---------------- robustness ----------------
    if rob:
        out["sources"]["robustness"] = rob_src
        out["robustness"] = {
            "clean_macro_f1": rob["clean"]["macro_f1"],
            "clean_mean_confidence": rob["clean"]["calibration"]["mean_confidence"],
            "confidence_accuracy_r": rob.get("confidence_tracks_accuracy", {}).get("pearson_r"),
            "n_conditions": rob.get("confidence_tracks_accuracy", {}).get("n_conditions"),
            "graph_verification": rob.get("graph_verification"),
            "corruption_specs": rob.get("corruption_specs"),
            "drops": {
                name: {f"sev{s}": sev[f"severity_{s}"]["macro_f1_drop_points"] for s in (1, 2, 3)}
                for name, sev in rob["corruptions"].items()
            },
            "confidence_at_sev3": {
                name: sev["severity_3"]["mean_confidence"] for name, sev in rob["corruptions"].items()
            },
            "clean_abstention": rob.get("clean_abstention"),
            "abstention": {
                name: {f"sev{s}": {k: sev[f"severity_{s}"]["abstention"][k] for k in
                                   ("coverage", "selective_accuracy", "selective_risk", "policy_violated",
                                    "required_threshold_for_policy", "required_threshold_reachable",
                                    "coverage_at_required_threshold")}
                       for s in (1, 2, 3)}
                for name, sev in rob["corruptions"].items()
            } if rob.get("abstention_policy") else None,
            "policy_summary": rob.get("policy_summary"),
        }

    # ---------------- export ----------------
    if ex:
        out["sources"]["export"] = ex_src
        out["export"] = {
            "keras_reference_macro_f1": ex["keras_reference"]["macro_f1"],
            "material_thresholds": ex["material_degradation_thresholds"],
            "variants": {
                name: {
                    "mb": v["mb"],
                    "macro_f1": v["metrics"]["macro_f1"],
                    "drop_points": v["degradation"]["macro_f1_drop_points"],
                    "top1_agreement": v["parity"]["top1_agreement_with_keras"],
                    "max_abs_deviation": v["parity"]["max_abs_probability_deviation"],
                    "worst_class_recall_drop": v["degradation"]["worst_per_class_recall_drop_points"],
                    "material": v["degradation"]["material_degradation"],
                    "delegate": v["interpreter"].get("delegate"),
                }
                for name, v in ex["variants"].items() if v.get("export_ok")
            },
        }
    if par:
        out["sources"]["export_paired"] = par_src
        out["export"]["paired"] = {
            name: {
                "macro_f1": v["macro_f1"],
                "delta_points": v["delta_macro_f1_points_vs_reference"],
                "delta_ci95_points": [v["paired_bootstrap"]["delta_macro_f1"]["ci95_low_points"],
                                      v["paired_bootstrap"]["delta_macro_f1"]["ci95_high_points"]],
                "delta_ci_includes_zero": v["paired_bootstrap"]["delta_macro_f1"]["ci_includes_zero"],
                "top1_agreement": v["top1_agreement_with_reference"],
                "n_disagreements": v["n_disagreements"],
                "mcnemar": v["mcnemar_vs_reference"],
                "worst_class": v["worst_class"],
                "per_class_recall": {c: v["per_class"][c]["recall"] for c in par["class_order"]},
                "per_class_precision": {c: v["per_class"][c]["precision"] for c in par["class_order"]},
                "top_transitions": dict(list(v["disagreement_transitions"].items())[:3]),
            }
            for name, v in par["variants"].items()
        }
        out["export"]["reference_per_class_recall"] = par["reference"]["per_class_recall"]

    # ---------------- activation maps ----------------
    if cam:
        out["sources"]["cam"] = cam_src

        def med(d, k):
            return d[k]["median"] if d.get(k) else None

        out["cam"] = {
            "n_test": cam["n_test"], "n_usable": cam["n_usable"], "n_flagged": cam["n_flagged"],
            "head_reproduces_classifier_max_abs_dev": cam["method"]["head_reproduces_classifier_max_abs_dev"],
            "graph_verification": cam["method"].get("graph_verification"),
            "overall": {"leaf_area_median": med(cam["overall"], "leaf_area_fraction"),
                        "mass_on_leaf_median": med(cam["overall"], "cam_mass_on_leaf"),
                        "concentration_ratio_median": med(cam["overall"], "concentration_ratio"),
                        "fraction_peak_on_leaf": cam["overall"]["fraction_peak_on_leaf"],
                        "fraction_ratio_above_1": cam["overall"]["fraction_ratio_above_1"]},
            "by_correctness": {k: {"n": v["n"], "concentration_ratio_median": med(v, "concentration_ratio"),
                                   "mass_on_leaf_median": med(v, "cam_mass_on_leaf"),
                                   "fraction_peak_on_leaf": v["fraction_peak_on_leaf"]}
                               for k, v in cam["by_correctness"].items()},
            "by_abstention": {k: {"n": v["n"], "concentration_ratio_median": med(v, "concentration_ratio"),
                                  "fraction_peak_on_leaf": v["fraction_peak_on_leaf"]}
                              for k, v in cam["by_abstention"].items()},
            "counterfactuals": {k: {kk: v[kk] for kk in ("n", "prediction_survives", "accuracy",
                                                        "accuracy_original_same_images",
                                                        "mean_calibrated_confidence",
                                                        "mean_calibrated_confidence_original",
                                                        "coverage_at_deployed_threshold", "coverage_original",
                                                        "predicted_class_histogram")}
                                for k, v in cam["counterfactuals"].items()},
        }

    # ---------------- device ----------------
    if dev:
        out["sources"]["device_benchmark"] = dev_src
        out["device"]["model"] = dev["device_model"]
        out["device"]["android"] = dev["android_release"]
        out["device"]["api"] = dev["api_level"]
        out["device"]["protocol"] = {
            "warmup": dev["warmup_iterations"],
            "measured_per_repeat": dev["measured_iterations_per_repeat"],
            "repeats": dev["repeats"],
        }
        out["device"]["latency"] = {
            r["model_file"]: {
                "mb": round(r["model_bytes"] / 1e6, 2),
                "delegate": r.get("delegate_used"),
                "cold_start_ms": round(r.get("cold_start_load_ms", -1), 1),
                "median_ms": round(r["pooled"]["median_ms"], 3),
                "p90_ms": round(r["pooled"]["p90_ms"], 3),
                "p95_ms": round(r["pooled"]["p95_ms"], 3),
                "stdev_ms": round(r["pooled"]["stdev_ms"], 3),
                "per_repeat_medians": [round(x["median_ms"], 3) for x in r["repeats"]],
                "thermal_before": r.get("thermal_before"),
                "thermal_after": r.get("thermal_after"),
            }
            for r in dev["results"] if r.get("pooled")
        }
    if dpar:
        out["sources"]["parity"] = dpar_src
        out["device"]["parity"] = {
            "n_cases": dpar["n_cases"],
            "top1_agreement": dpar["top1_agreement"],
            "worst_deviation": dpar["worst_max_abs_probability_deviation"],
            "tolerance": dpar["tolerance"],
            "within_tolerance": dpar["within_tolerance"],
        }
    if inv:
        out["sources"]["device_inventory"] = inv_src
        d = inv["device_identity"]
        out["device"]["soc"] = f"{d['soc_manufacturer']} {d['soc_model']}"
        out["device"]["ram_gb"] = inv["memory_total_gb"]
        out["device"]["cores"] = inv["cpu"]["cores_reported"]
        out["device"]["nnapi_feature_declared"] = inv["acceleration"]["neuralnetworks_feature"]

    dest = repo / "reports" / "frozen_results.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"frozen results written: {dest}")
    print(f"sources recorded: {len(out['sources'])}")
    for section, payload in out.items():
        if section == "sources":
            continue
        print(f"  {section}: {len(payload)} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
