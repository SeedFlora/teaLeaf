"""Aggregate benchmark runs and compare models on the frozen test partition.

Two comparisons that are easy to conflate are kept apart:

SEED SPREAD  — the same architecture trained with different seeds. This measures the
    variability of the training procedure. With three seeds per architecture it is
    reported as mean, standard deviation and observed range; three runs cannot support a
    distributional claim, so the range is always shown alongside the deviation.

PAIRED TEST  — two architectures scored on the SAME images. McNemar's test uses only the
    discordant pairs (right for A and wrong for B, and the reverse), which is the whole
    reason a paired test is appropriate: an unpaired comparison of two macro-F1 values
    throws away the fact that both models saw an identical test partition.

Revision (reviewers asked that rankings rest on mean and variability, not one seed):
every architecture now has the same seeds, the per-architecture summary is mean +/- sd
over them, and McNemar is computed SEED-MATCHED -- architecture A at seed s against
architecture B at the same seed s -- so a pair is judged on three independent
replications rather than on whichever run happened to score highest. The seed-42
family is the one carried into the paper (Holm-Bonferroni corrected across its ten
pairs); the other seeds are reported as replications of each verdict.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_ORDER = [
    "1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight", "4. Helopeltis",
    "5. Red spider", "6. Green mirid bug", "7. Healthy leaf",
]
PRIMARY_SEED = 42


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant counts.

    Used instead of the chi-square approximation because the discordant totals here are
    small; the chi-square form is unreliable below roughly 25 discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    both = int(np.sum(correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    p_exact = binom_two_sided(b, c)

    chi2 = None
    if b + c >= 25:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    return {
        "n": int(len(correct_a)),
        "both_correct": both,
        "both_wrong": neither,
        "a_right_b_wrong": b,
        "a_wrong_b_right": c,
        "discordant": b + c,
        "p_value_exact_binomial": p_exact,
        "chi2_with_continuity_correction": chi2,
        "chi2_applicable": bool(b + c >= 25),
        "note": (
            "The exact binomial test is the primary result. The chi-square approximation "
            "is reported only when at least 25 discordant pairs exist, and is otherwise null."
        ),
    }


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)          # enforce monotonicity
        out[k] = {"p_raw": p, "p_holm": running,
                  "significant_at_0.05": bool(running < 0.05)}
    return out


def load_runs(runs_dir: Path, eval_dir: Path) -> list[dict]:
    out = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg = d / "train_config.json"
        if not cfg.exists():
            continue                          # unfinished run
        c = json.loads(cfg.read_text(encoding="utf-8"))
        ed = eval_dir / d.name
        metrics = ed / "metrics.json"
        rec = {"run": d.name, "config": c, "eval_dir": ed,
               "metrics": json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else None}
        out.append(rec)
    return out


def mean_sd(values: list[float]) -> dict:
    v = np.array(values, dtype=float)
    return {"mean": round(float(v.mean()), 4),
            "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "range_points": round(100 * float(v.max() - v.min()), 2),
            "values": [round(float(x), 4) for x in v]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--partition", default="test")
    a = ap.parse_args()

    runs_dir, eval_dir, outdir = Path(a.runs_dir), Path(a.eval_dir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir, eval_dir)
    scored = [r for r in runs if r["metrics"] and a.partition in r["metrics"]["partitions"]]
    print(f"{len(runs)} completed runs, {len(scored)} with {a.partition} metrics")

    # ---------------- per-run table ---------------- #
    rows = []
    for r in scored:
        c, m = r["config"], r["metrics"]["partitions"][a.partition]
        rows.append({
            "run": r["run"],
            "backbone": c["backbone"],
            "split": c["split_prefix"],
            "seed": c["seed"],
            "params": c["total_params"],
            "epochs": c["epochs_run"],
            "train_minutes": round(c["wall_clock_seconds"] / 60, 1),
            "val_macro_f1": round(c["best_val_macro_f1"], 4),
            "test_macro_f1": round(m["macro_f1"], 4),
            # Unrounded value: every derived statistic (means, spreads, leakage inflation,
            # seed-matched differences) is computed from this, and only the outputs are rounded.
            "test_macro_f1_raw": float(m["macro_f1"]),
            "test_macro_f1_ci_low": round(m["bootstrap"]["macro_f1"]["ci95_low"], 4),
            "test_macro_f1_ci_high": round(m["bootstrap"]["macro_f1"]["ci95_high"], 4),
            "test_accuracy": round(m["accuracy"], 4),
            "test_balanced_accuracy": round(m["balanced_accuracy"], 4),
            "test_mcc": round(m["mcc"], 4),
            "test_ece": round(m["calibration"]["ece"], 4),
            "test_nll": round(m["calibration"]["nll"], 4),
            "test_brier": round(m["calibration"]["brier"], 4),
            "test_per_class_f1": {k: round(v["f1"], 4) for k, v in m["per_class"].items()},
        })
    rows.sort(key=lambda r: (-r["test_macro_f1"]))

    # ---------------- seed spread ---------------- #
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["backbone"], r["split"])].append(r)

    seed_spread = {}
    for (bb, sp), rs in sorted(by_group.items()):
        f1 = [r["test_macro_f1_raw"] for r in rs]
        seed_spread[f"{bb}|{sp}"] = {
            "n_seeds": len(rs),
            "seeds": [r["seed"] for r in rs],
            "test_macro_f1_values": [round(v, 4) for v in f1],
            "mean": round(float(np.mean(f1)), 4),
            "min": round(min(f1), 4),
            "max": round(max(f1), 4),
            "range": round(max(f1) - min(f1), 4),
            "stdev": round(float(np.std(f1, ddof=1)), 4) if len(f1) > 1 else None,
            "note": ("Observed spread across independently seeded runs. With so few runs this "
                     "is descriptive spread, not a confidence interval."),
        }

    # ---------------- multi-seed summary per architecture (primary split) ---------------- #
    multiseed = {}
    for (bb, sp), rs in sorted(by_group.items()):
        if sp != "primary":
            continue
        rs = sorted(rs, key=lambda r: r["seed"])
        entry = {
            "params": rs[0]["params"],
            "n_seeds": len(rs),
            "seeds": [r["seed"] for r in rs],
            "runs": [r["run"] for r in rs],
        }
        for metric in ("test_macro_f1", "test_accuracy", "test_balanced_accuracy", "test_mcc",
                       "test_ece", "test_nll", "test_brier", "val_macro_f1", "epochs", "train_minutes"):
            source = "test_macro_f1_raw" if metric == "test_macro_f1" else metric
            entry[metric] = mean_sd([r[source] for r in rs])
        entry["test_per_class_f1"] = {
            cls: mean_sd([r["test_per_class_f1"][cls] for r in rs]) for cls in CLASS_ORDER
        }
        multiseed[bb] = entry

    # Rank of each architecture within each seed, to show whether the ordering is stable.
    seeds_all = sorted({r["seed"] for r in rows if r["split"] == "primary"})
    rank_by_seed = {}
    for s in seeds_all:
        rs = sorted([r for r in rows if r["split"] == "primary" and r["seed"] == s],
                    key=lambda r: -r["test_macro_f1_raw"])
        rank_by_seed[str(s)] = [{"backbone": r["backbone"], "test_macro_f1": r["test_macro_f1"]} for r in rs]
    by_mean = sorted(multiseed.items(), key=lambda kv: -kv[1]["test_macro_f1"]["mean"])
    rank_summary = {
        "by_mean_macro_f1": [bb for bb, _ in by_mean],
        "by_seed": rank_by_seed,
    }

    # Is the seed spread larger than the gaps between architectures?
    if len(by_mean) >= 2:
        means = [v["test_macro_f1"]["mean"] for _, v in by_mean]
        gaps = [round(100 * (means[i] - means[i + 1]), 2) for i in range(len(means) - 1)]
        rank_summary["adjacent_mean_gaps_points"] = gaps
        rank_summary["max_within_architecture_range_points"] = max(
            v["test_macro_f1"]["range_points"] for _, v in by_mean)

    # ---------------- leakage sensitivity (RQ1) ---------------- #
    leakage = {}
    for bb in {r["backbone"] for r in rows}:
        prim = [r for r in rows if r["backbone"] == bb and r["split"] == "primary"]
        naive = [r for r in rows if r["backbone"] == bb and r["split"] == "naive"]
        if prim and naive:
            pf = [r["test_macro_f1_raw"] for r in prim]
            nf = [r["test_macro_f1_raw"] for r in naive]
            common = sorted({r["seed"] for r in prim} & {r["seed"] for r in naive})
            per_seed = {}
            for s in common:
                p_ = next(r["test_macro_f1_raw"] for r in prim if r["seed"] == s)
                n_ = next(r["test_macro_f1_raw"] for r in naive if r["seed"] == s)
                per_seed[str(s)] = {"primary": round(p_, 4), "naive": round(n_, 4),
                                    "inflation_points": round(100 * (n_ - p_), 3)}
            leakage[bb] = {
                "primary_mean_macro_f1": round(float(np.mean(pf)), 4),
                "naive_mean_macro_f1": round(float(np.mean(nf)), 4),
                "inflation_points": round(100 * (float(np.mean(nf)) - float(np.mean(pf))), 3),
                "primary_seeds": len(pf),
                "naive_seeds": len(nf),
                "seeds": common,
                "primary_range": [round(min(pf), 4), round(max(pf), 4)],
                "naive_range": [round(min(nf), 4), round(max(nf), 4)],
                "primary_sd": round(float(np.std(pf, ddof=1)), 4) if len(pf) > 1 else None,
                "naive_sd": round(float(np.std(nf, ddof=1)), 4) if len(nf) > 1 else None,
                "per_seed": per_seed,
                "interpretation_guard": (
                    "The naive split carries 16 of 43 confirmed duplicate pairs across "
                    "partitions, which bounds the arithmetically possible inflation at roughly "
                    "1-2 points. A materially larger measured difference indicates a bug or a "
                    "confound, not a leakage discovery."
                ),
            }

    # ---------------- seed-matched paired McNemar between architectures ---------------- #
    primary_runs = [r for r in scored if r["config"]["split_prefix"] == "primary"]
    preds: dict[tuple[str, int], np.ndarray] = {}
    truth: np.ndarray | None = None
    for r in primary_runs:
        p = r["eval_dir"] / f"predictions_primary_{a.partition}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        if truth is None:
            truth = z["y_true"]
        elif not np.array_equal(truth, z["y_true"]):
            raise SystemExit(f"{r['run']} was scored on a different label vector; refusing to pair")
        preds[(r["config"]["backbone"], r["config"]["seed"])] = z["probs"].argmax(axis=1)

    backbones = sorted({bb for bb, _ in preds})
    pairwise = {}
    raw_p = {}
    seed_matched = {}
    if truth is not None and len(backbones) >= 2:
        for x_, y_ in itertools.combinations(backbones, 2):
            key = f"{x_}_vs_{y_}"
            per_seed = {}
            for s in seeds_all:
                if (x_, s) in preds and (y_, s) in preds:
                    res = mcnemar(preds[(x_, s)] == truth, preds[(y_, s)] == truth)
                    fx = next(r["test_macro_f1_raw"] for r in rows if r["backbone"] == x_ and r["seed"] == s and r["split"] == "primary")
                    fy = next(r["test_macro_f1_raw"] for r in rows if r["backbone"] == y_ and r["seed"] == s and r["split"] == "primary")
                    per_seed[str(s)] = {
                        "a_right_b_wrong": res["a_right_b_wrong"],
                        "a_wrong_b_right": res["a_wrong_b_right"],
                        "p_exact": res["p_value_exact_binomial"],
                        "macro_f1_a_minus_b_points": round(100 * (fx - fy), 2),
                    }
            if str(PRIMARY_SEED) in per_seed:
                res = mcnemar(preds[(x_, PRIMARY_SEED)] == truth, preds[(y_, PRIMARY_SEED)] == truth)
                pairwise[key] = res
                raw_p[key] = res["p_value_exact_binomial"]
            diffs = [v["macro_f1_a_minus_b_points"] for v in per_seed.values()]
            seed_matched[key] = {
                "seeds": list(per_seed),
                "per_seed": per_seed,
                "n_seeds_significant_raw_0.05": sum(v["p_exact"] < 0.05 for v in per_seed.values()),
                "mean_macro_f1_a_minus_b_points": round(float(np.mean(diffs)), 2) if diffs else None,
                "sign_consistent": bool(diffs) and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)),
            }
        corrected = holm_bonferroni(raw_p)
        for k, v in corrected.items():
            pairwise[k]["holm_bonferroni"] = v

    summary = {
        "partition": a.partition,
        "runs_table": rows,
        "seed_spread": seed_spread,
        "multiseed": multiseed,
        "ranking": rank_summary,
        "leakage_sensitivity": leakage,
        "pairwise_mcnemar": pairwise,
        "pairwise_mcnemar_basis": f"seed-matched at seed {PRIMARY_SEED}; Holm-Bonferroni across all pairs",
        "seed_matched_mcnemar": seed_matched,
        "multiplicity_correction": "Holm-Bonferroni across all architecture pairs tested (seed-42 family)",
        "assumptions": [
            "McNemar assumes paired binary outcomes on identical items; both models were "
            "scored on the same frozen test partition in the same order, verified by "
            "comparing label vectors before pairing.",
            "The exact binomial form is used because discordant counts are small.",
            "Bootstrap intervals quantify test-set sampling uncertainty for a fixed trained "
            "model; they do not capture training variability, which is reported separately "
            "as seed spread.",
            "Seed-matched comparisons pair architecture A at seed s with architecture B at the "
            "same seed s; the seeds are shared initialisations of the head and the data order, "
            "not of the ImageNet backbone weights, which are fixed.",
        ],
    }

    (outdir / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        flat = [{k: v for k, v in r.items() if k not in ("test_per_class_f1", "test_macro_f1_raw")} for r in rows]
        with (outdir / "benchmark_table.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
    if multiseed:
        with (outdir / "multiseed_table.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["backbone", "params", "n_seeds", "macro_f1_mean", "macro_f1_sd", "macro_f1_min",
                        "macro_f1_max", "accuracy_mean", "accuracy_sd", "mcc_mean", "mcc_sd",
                        "ece_mean", "ece_sd", "nll_mean", "nll_sd"])
            for bb, v in by_mean:
                w.writerow([bb, v["params"], v["n_seeds"],
                            v["test_macro_f1"]["mean"], v["test_macro_f1"]["sd"],
                            v["test_macro_f1"]["min"], v["test_macro_f1"]["max"],
                            v["test_accuracy"]["mean"], v["test_accuracy"]["sd"],
                            v["test_mcc"]["mean"], v["test_mcc"]["sd"],
                            v["test_ece"]["mean"], v["test_ece"]["sd"],
                            v["test_nll"]["mean"], v["test_nll"]["sd"]])

    print(f"\n{'run':<46}{'params':>11}{'testF1':>9}{'CI95':>19}{'acc':>8}{'ECE':>8}")
    for r in rows:
        ci = f"[{r['test_macro_f1_ci_low']:.3f},{r['test_macro_f1_ci_high']:.3f}]"
        print(f"{r['run']:<46}{r['params']:>11,}{r['test_macro_f1']:>9.4f}{ci:>19}"
              f"{r['test_accuracy']:>8.4f}{r['test_ece']:>8.4f}")

    if multiseed:
        print(f"\n--- multi-seed summary (primary split, test macro-F1) ---")
        for bb, v in by_mean:
            f = v["test_macro_f1"]
            sd = f"{f['sd']:.4f}" if f["sd"] is not None else "  n/a "
            print(f"  {bb:<20} n={v['n_seeds']}  {f['mean']:.4f} +/- {sd}  "
                  f"[{f['min']:.4f}, {f['max']:.4f}]  range {f['range_points']:.2f} pts  "
                  f"ECE {v['test_ece']['mean']:.4f}")
        print(f"  ranking by seed: " + "; ".join(
            f"seed {s}: " + " > ".join(e['backbone'] for e in v) for s, v in rank_by_seed.items()))

    if leakage:
        print("\n--- leakage sensitivity (naive minus primary, percentage points) ---")
        for bb, l in leakage.items():
            print(f"  {bb:<24} primary {l['primary_mean_macro_f1']:.4f}  "
                  f"naive {l['naive_mean_macro_f1']:.4f}  "
                  f"inflation {l['inflation_points']:+.2f} pts  per seed "
                  + ", ".join(f"{s}:{v['inflation_points']:+.2f}" for s, v in l["per_seed"].items()))

    if pairwise:
        print(f"\n--- paired McNemar, seed-matched at seed {PRIMARY_SEED} (Holm across pairs) ---")
        for k, v in pairwise.items():
            h = v.get("holm_bonferroni", {})
            sm = seed_matched.get(k, {})
            print(f"  {k:<52} b={v['a_right_b_wrong']:<4} c={v['a_wrong_b_right']:<4} "
                  f"p={v['p_value_exact_binomial']:.4g}  p_holm={h.get('p_holm', float('nan')):.4g}  "
                  f"| sig seeds {sm.get('n_seeds_significant_raw_0.05')}/{len(sm.get('seeds', []))}  "
                  f"mean dF1 {sm.get('mean_macro_f1_a_minus_b_points')} pts")

    print(f"\nwritten: {outdir/'model_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
