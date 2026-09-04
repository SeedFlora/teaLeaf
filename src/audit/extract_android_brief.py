"""Save the Android implementation research into committed reference documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    res = raw["result"]
    research = res.get("research", []) or []
    checks = res.get("fact_checks", []) or []
    brief = res.get("implementation_brief", "") or ""

    (outdir / "ANDROID_IMPLEMENTATION_BRIEF.md").write_text(brief, encoding="utf-8")

    lines = ["# Android Technical Research — Verified Findings\n",
             "Each section records what official documentation actually states, the concrete",
             "decision taken, and the silent-failure modes found.\n"]
    for r in research:
        lines += [f"\n---\n\n## {r['area']}\n",
                  "### Findings\n", r.get("findings", ""), "\n### Decisions\n",
                  r.get("concrete_decisions", ""), "\n### Pitfalls\n", r.get("pitfalls", "")]
        if r.get("code_sketch"):
            lines += ["\n### Reference code\n", r["code_sketch"]]
        if r.get("uncertainty"):
            lines += ["\n### Not confirmed\n", r["uncertainty"]]
        if r.get("sources"):
            lines += ["\n### Sources\n"] + [f"- {s}" for s in r["sources"]]
    (outdir / "ANDROID_RESEARCH_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")

    cl = ["# Adversarial Fact-Checks of Load-Bearing Claims\n",
          "Each claim was re-verified against official sources by an agent instructed to",
          "assume it was wrong until proven otherwise.\n"]
    for c in checks:
        cl += [f"\n## [{c.get('verdict')}] {c.get('claim','')}\n",
               f"**Correction:** {c.get('correction','')}\n",
               f"**Source:** {c.get('evidence_url','')}\n"]
    (outdir / "ANDROID_FACT_CHECKS.md").write_text("\n".join(cl), encoding="utf-8")

    print(f"research areas : {len(research)}")
    print(f"fact checks    : {len(checks)}")
    for c in checks:
        print(f"  [{c.get('verdict','?'):<14}] {str(c.get('claim',''))[:88]}")
    print(f"\nbrief length   : {len(brief)} chars")
    print(f"written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
