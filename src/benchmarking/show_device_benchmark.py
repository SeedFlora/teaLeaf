"""Render the on-device latency benchmark report as a readable table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    print(f"device   : {d['device_model']} ({d['device_product']})  hardware={d['hardware']}")
    print(f"android  : {d['android_release']}  API {d['api_level']}  abi={d['supported_abis'][0]}")
    print(f"protocol : {d['warmup_iterations']} warm-up, "
          f"{d['measured_iterations_per_repeat']} measured x {d['repeats']} repeats "
          f"= {d['measured_iterations_per_repeat'] * d['repeats']} timed inferences per variant")
    print(f"clock    : {d['protocol']['clock']}")
    print(f"excludes : {d['protocol']['excludes']}")
    print()

    hdr = (f"{'model':<32}{'MB':>7}{'delegate':>11}{'load ms':>9}"
           f"{'median':>9}{'p90':>8}{'p95':>8}{'sd':>7}{'thermal':>9}")
    print(hdr)
    print("-" * len(hdr))

    for r in d["results"]:
        name = r["model_file"]
        mb = r["model_bytes"] / 1e6
        p = r.get("pooled")
        if not p:
            print(f"{name:<32}{mb:>7.2f}   FAILED: {str(r.get('error', ''))[:58]}")
            continue
        therm = f"{r.get('thermal_before')}->{r.get('thermal_after')}"
        print(f"{name:<32}{mb:>7.2f}{str(r.get('delegate_used', '?'))[:10]:>11}"
              f"{r.get('cold_start_load_ms', -1):>9.1f}"
              f"{p['median_ms']:>9.3f}{p['p90_ms']:>8.3f}{p['p95_ms']:>8.3f}"
              f"{p['stdev_ms']:>7.3f}{therm:>9}")

    print()
    for r in d["results"]:
        if r.get("xnnpack_failed_error"):
            print(f"NOTE {r['model_file']}: XNNPACK could not prepare this graph -- "
                  f"{str(r['xnnpack_failed_error'])[:150]}")

    print("\nper-repeat medians (checks for thermal drift across the run):")
    for r in d["results"]:
        reps = r.get("repeats")
        if not reps:
            continue
        meds = [f"{x['median_ms']:.3f}" for x in reps]
        therms = [str(x.get("thermal_after_repeat")) for x in reps]
        print(f"  {r['model_file']:<32} {' | '.join(meds)}   thermal {' '.join(therms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
