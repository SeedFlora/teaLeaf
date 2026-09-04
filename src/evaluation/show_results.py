"""Print a compact view of the current benchmark, calibration, robustness and export results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    repo = Path(sys.argv[1])
    t = repo / "tables"

    cmp_ = load(t / "model_comparison.json")
    if cmp_:
        print("=" * 96)
        print("BENCHMARK (frozen test partition)")
        print("=" * 96)
        print(f"{'run':<44}{'params':>11}{'testF1':>9}{'CI95':>18}{'acc':>8}{'MCC':>8}{'ECE':>8}")
        for r in cmp_["runs_table"]:
            ci = f"[{r['test_macro_f1_ci_low']:.3f},{r['test_macro_f1_ci_high']:.3f}]"
            print(f"{r['run']:<44}{r['params']:>11,}{r['test_macro_f1']:>9.4f}{ci:>18}"
                  f"{r['test_accuracy']:>8.4f}{r['test_mcc']:>8.4f}{r['test_ece']:>8.4f}")

        if cmp_.get("leakage_sensitivity"):
            print("\nLEAKAGE SENSITIVITY (RQ1)")
            for bb, l in cmp_["leakage_sensitivity"].items():
                print(f"  {bb}: primary {l['primary_mean_macro_f1']:.4f} -> "
                      f"naive {l['naive_mean_macro_f1']:.4f}   "
                      f"inflation {l['inflation_points']:+.2f} points")

        if cmp_.get("pairwise_mcnemar"):
            print("\nPAIRED McNEMAR")
            for k, v in cmp_["pairwise_mcnemar"].items():
                h = v.get("holm_bonferroni", {})
                print(f"  {k}: b={v['a_right_b_wrong']} c={v['a_wrong_b_right']} "
                      f"p={v['p_value_exact_binomial']:.4g} p_holm={h.get('p_holm', float('nan')):.4g}")

    cal = load(t / "calibration" / "calibration.json")
    if cal:
        print("\n" + "=" * 96)
        print("CALIBRATION AND ABSTENTION")
        print("=" * 96)
        print(f"  fitted temperature T = {cal['temperature']:.4f}  (validation only)")
        print(f"{'':<26}{'NLL':>10}{'ECE':>10}{'Brier':>10}{'meanConf':>10}{'acc':>9}")
        for k in ("test_uncalibrated", "test_calibrated"):
            r = cal[k]
            print(f"  {r['tag']:<24}{r['nll']:>10.4f}{r['ece']:>10.4f}{r['brier']:>10.4f}"
                  f"{r['mean_confidence']:>10.4f}{r['accuracy']:>9.4f}")
        tu, tc = cal["test_uncalibrated"], cal["test_calibrated"]
        print(f"  ECE change {tc['ece'] - tu['ece']:+.4f}   NLL change {tc['nll'] - tu['nll']:+.4f}")
        ap_, ts = cal["abstention_policy"], cal["test_selective_performance"]
        print(f"\n  policy: risk target {ap_['target_selective_risk']}, "
              f"min coverage {ap_['min_coverage']}")
        print(f"  threshold {ap_['threshold']:.4f}  ({ap_['basis']})")
        print(f"  TEST coverage {ts['coverage']:.4f}  selective acc {ts['selective_accuracy']}  "
              f"vs {ts['full_coverage_accuracy']:.4f} at full coverage")
        print(f"  abstained on {ts['n_abstained']}; accuracy there would have been "
              f"{ts['accuracy_on_abstained']}")

    rob = load(t / "robustness" / "robustness.json")
    if rob:
        print("\n" + "=" * 96)
        print("ROBUSTNESS (macro-F1 drop in points, by severity)")
        print("=" * 96)
        print(f"  clean macro-F1 {rob['clean']['macro_f1']:.4f}   "
              f"mean confidence {rob['clean']['calibration']['mean_confidence']:.4f}")
        print(f"{'corruption':<20}{'sev1':>10}{'sev2':>10}{'sev3':>10}{'conf@sev3':>12}")
        for name, sev in rob["corruptions"].items():
            d = [sev[f"severity_{s}"]["macro_f1_drop_points"] for s in (1, 2, 3)]
            c3 = sev["severity_3"]["mean_confidence"]
            print(f"{name:<20}{d[0]:>10.2f}{d[1]:>10.2f}{d[2]:>10.2f}{c3:>12.4f}")
        if "confidence_tracks_accuracy" in rob:
            print(f"\n  confidence-accuracy correlation across all corruptions: "
                  f"r = {rob['confidence_tracks_accuracy']['pearson_r']:.4f}")

    exp = load(repo / "models" / "exported" / "export_report.json")
    if exp:
        print("\n" + "=" * 96)
        print("EXPORT AND QUANTIZATION")
        print("=" * 96)
        ref = exp["keras_reference"]
        print(f"  Keras reference: macro-F1 {ref['macro_f1']:.4f}  acc {ref['accuracy']:.4f}")
        print(f"{'variant':<10}{'MB':>8}{'macroF1':>10}{'dF1 pts':>10}{'top1 agree':>12}"
              f"{'max|dp|':>11}{'material':>10}")
        for name, v in exp["variants"].items():
            if not v.get("export_ok"):
                print(f"{name:<10}  FAILED: {v.get('error','')[:60]}")
                continue
            d, p = v["degradation"], v["parity"]
            print(f"{name:<10}{v['mb']:>8.2f}{v['metrics']['macro_f1']:>10.4f}"
                  f"{d['macro_f1_drop_points']:>10.2f}{p['top1_agreement_with_keras']:>12.4f}"
                  f"{p['max_abs_probability_deviation']:>11.2e}"
                  f"{str(d['material_degradation']):>10}")
            worst = max(d["per_class_recall_drop_points"].items(), key=lambda kv: kv[1])
            print(f"{'':>10}  worst per-class recall drop: {worst[0]} {worst[1]:+.2f} pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
