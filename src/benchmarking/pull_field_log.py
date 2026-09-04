"""Pull the on-device field research log over ADB and summarize what it can support.

Every transfer is verified: each image is re-hashed and checked against the SHA-256 the
app recorded at capture time. A record whose image does not match is quarantined rather
than analyzed, so a truncated or corrupted pull fails loudly instead of quietly shifting
a result.

The summary deliberately reports three tiers separately -- label-free measurements, the
non-expert annotated subset, and the expert-verified subset -- because promoting a figure
from a weaker tier to a stronger one is the specific error this whole design exists to
prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics as st
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REMOTE_TEMPLATE = "/sdcard/Android/data/{pkg}/files/research"


def adb(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout)


def pull(pkg: str, dest: Path) -> tuple[bool, str]:
    remote = REMOTE_TEMPLATE.format(pkg=pkg)
    probe = adb(["shell", "ls", remote])
    if probe.returncode != 0 or "No such file" in (probe.stdout + probe.stderr):
        return False, f"remote directory not found: {remote}\n{probe.stdout}{probe.stderr}"

    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = dest.with_name(f"{dest.name}_previous_{stamp}")
        shutil.move(str(dest), str(backup))
        print(f"existing pull preserved at {backup}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = adb(["pull", remote, str(dest)], timeout=1800)
    if r.returncode != 0:
        return False, f"adb pull failed:\n{r.stdout}{r.stderr}"
    return True, r.stdout.strip()


def verify_and_load(root: Path) -> tuple[list[dict], list[dict]]:
    log = root / "field_log.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"no field_log.jsonl under {root}")

    good: list[dict] = []
    quarantined: list[dict] = []

    for lineno, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            quarantined.append({"line": lineno, "reason": f"unparseable JSON: {exc}"})
            continue

        img = rec.get("image", {})
        rel = img.get("file")
        expected = (img.get("sha256") or "").lower()
        if not rel or not expected:
            quarantined.append({"line": lineno, "record_id": rec.get("record_id"),
                                "reason": "record does not reference an image with a hash"})
            continue

        p = root / rel
        if not p.exists():
            quarantined.append({"line": lineno, "record_id": rec.get("record_id"),
                                "reason": f"image missing after pull: {rel}"})
            continue

        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            quarantined.append({"line": lineno, "record_id": rec.get("record_id"),
                                "reason": "image hash mismatch — transfer corrupted or file altered",
                                "expected": expected, "actual": actual})
            continue

        good.append(rec)

    return good, quarantined


def summarize(records: list[dict]) -> dict:
    consented = [r for r in records
                 if (r.get("consent") or {}).get("research_use_granted") is True]

    def m(r: dict) -> dict:
        return r.get("measurement") or {}

    conf = [m(r).get("top1_confidence_calibrated") for r in consented
            if m(r).get("top1_confidence_calibrated") is not None]
    ent = [m(r).get("predictive_entropy") for r in consented
           if m(r).get("predictive_entropy") is not None]
    abstained = [bool(m(r).get("abstained")) for r in consented]

    def lat(field: str) -> list[float]:
        return [((m(r).get("latency_ms") or {}).get(field)) for r in consented
                if (m(r).get("latency_ms") or {}).get(field) is not None]

    def stats(vals: list[float]) -> dict | None:
        if not vals:
            return None
        s = sorted(vals)
        return {
            "n": len(s),
            "median": round(st.median(s), 3),
            "mean": round(st.fmean(s), 3),
            "p90": round(s[min(len(s) - 1, int(0.90 * len(s)))], 3),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
            "min": round(s[0], 3), "max": round(s[-1], 3),
            "stdev": round(st.pstdev(s), 3) if len(s) > 1 else 0.0,
        }

    # ---- tier 1: label-free, valid regardless of annotator expertise ----
    tier1 = {
        "records": len(consented),
        "top1_calibrated_confidence": stats(conf),
        "predictive_entropy": stats(ent),
        "abstention_rate": round(sum(abstained) / len(abstained), 4) if abstained else None,
        "latency_ms": {k: stats(lat(k)) for k in
                       ("preprocess", "inference", "postprocess", "end_to_end")},
        "delegates": dict(Counter((m(r).get("runtime") or {}).get("delegate") for r in consented)),
        "model_variants": dict(Counter((m(r).get("model") or {}).get("variant") for r in consented)),
        "thermal_status_seen": dict(Counter(
            (m(r).get("device_state") or {}).get("thermal_status") for r in consented)),
        "context_background": dict(Counter((r.get("context") or {}).get("background")
                                           for r in consented)),
        "leaf_attached_to_plant": dict(Counter((r.get("context") or {}).get("leaf_attached_to_plant")
                                               for r in consented)),
    }

    # ---- tier 2: non-expert annotation ----
    annotated = [r for r in consented
                 if (r.get("annotation") or {}).get("verdict") in ("correct", "incorrect", "unsure")]
    verdicts = Counter((r.get("annotation") or {}).get("verdict") for r in annotated)
    expertise = Counter((r.get("annotation") or {}).get("annotator_expertise") for r in annotated)
    decided = verdicts["correct"] + verdicts["incorrect"]

    tier2 = {
        "records_annotated": len(annotated),
        "verdicts": dict(verdicts),
        "annotator_expertise": dict(expertise),
        "apparent_agreement_rate": round(verdicts["correct"] / decided, 4) if decided else None,
        "claimed_true_classes_when_incorrect": dict(Counter(
            (r.get("annotation") or {}).get("claimed_true_class")
            for r in annotated if (r.get("annotation") or {}).get("verdict") == "incorrect")),
        "INTERPRETATION": (
            "This is agreement between the model and a human observer of the stated expertise. "
            "It is NOT accuracy and must never be reported as external validation accuracy "
            "unless every listed annotator is a qualified specialist."
        ),
    }

    # ---- tier 3: expert-verified, the only tier that can carry an accuracy claim ----
    expert = [r for r in consented if (r.get("expert_verification") or {}).get("verified") is True]
    correct = 0
    for r in expert:
        ev = r.get("expert_verification") or {}
        mm = m(r).get("model") or {}
        order = mm.get("class_order") or []
        idx = m(r).get("top1_index")
        pred = order[idx] if isinstance(idx, int) and 0 <= idx < len(order) else None
        if pred is not None and pred == ev.get("expert_class"):
            correct += 1

    tier3 = {
        "records_expert_verified": len(expert),
        "expert_roles": dict(Counter((r.get("expert_verification") or {}).get("expert_role")
                                     for r in expert)),
        "top1_accuracy": round(correct / len(expert), 4) if expert else None,
        "INTERPRETATION": (
            "The only subset on which accuracy, precision, recall or F1 may be reported. "
            "With a small n, report a confidence interval and avoid per-class claims."
        ),
    }

    return {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_total": len(records),
        "records_consented": len(consented),
        "records_without_consent_excluded": len(records) - len(consented),
        "tier1_label_free": tier1,
        "tier2_non_expert_annotation": tier2,
        "tier3_expert_verified": tier3,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="org.tealeafai.screening")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--no-pull", action="store_true",
                    help="analyze an already-pulled directory without touching the device")
    a = ap.parse_args()

    dest = Path(a.dest)

    if not a.no_pull:
        devices = adb(["devices"]).stdout.strip().splitlines()[1:]
        if not [d for d in devices if d.strip()]:
            print("NO DEVICE CONNECTED. Connect the phone, unlock it, and confirm USB debugging.")
            return 2
        ok, msg = pull(a.package, dest)
        print(msg)
        if not ok:
            return 3

    records, quarantined = verify_and_load(dest)
    print(f"\nloaded {len(records)} verified records, {len(quarantined)} quarantined")
    for q in quarantined[:10]:
        print(f"  QUARANTINED line {q.get('line')}: {q['reason']}")

    summary = summarize(records)
    out = dest.parent / f"{dest.name}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if quarantined:
        (dest.parent / f"{dest.name}_quarantined.json").write_text(
            json.dumps(quarantined, indent=2), encoding="utf-8")

    t1, t2, t3 = (summary["tier1_label_free"], summary["tier2_non_expert_annotation"],
                  summary["tier3_expert_verified"])
    print(f"\n--- TIER 1 (no labels needed; valid regardless of annotator) ---")
    print(f"  records                 : {t1['records']}")
    print(f"  abstention rate         : {t1['abstention_rate']}")
    if t1["top1_calibrated_confidence"]:
        c = t1["top1_calibrated_confidence"]
        print(f"  top-1 confidence        : median {c['median']}  p90 {c['p90']}  min {c['min']}")
    if t1["latency_ms"]["end_to_end"]:
        e = t1["latency_ms"]["end_to_end"]
        print(f"  end-to-end latency ms   : median {e['median']}  p95 {e['p95']}")
    if t1["latency_ms"]["inference"]:
        i = t1["latency_ms"]["inference"]
        print(f"  pure inference ms       : median {i['median']}  p95 {i['p95']}")
    print(f"  leaf attached to plant  : {t1['leaf_attached_to_plant']}")

    print(f"\n--- TIER 2 (non-expert annotation; NOT accuracy) ---")
    print(f"  annotated               : {t2['records_annotated']}")
    print(f"  verdicts                : {t2['verdicts']}")
    print(f"  annotator expertise     : {t2['annotator_expertise']}")
    print(f"  apparent agreement rate : {t2['apparent_agreement_rate']}")

    print(f"\n--- TIER 3 (expert-verified; the only accuracy claim) ---")
    print(f"  expert-verified records : {t3['records_expert_verified']}")
    print(f"  top-1 accuracy          : {t3['top1_accuracy']}")

    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
