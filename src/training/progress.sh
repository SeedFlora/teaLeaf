#!/usr/bin/env bash
# Report benchmark progress from the per-run epoch logs, which the training script
# rewrites every epoch. This reads state directly rather than tailing the pipeline's
# stdout, whose grep stage block-buffers when writing to a file.
RUNS="$HOME/tea_ws/runs"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null

echo "=== runs ==="
for d in "$RUNS"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  case "$name" in smoke_*) continue ;; esac
  python3 - "$d" "$name" <<'PY'
import json, sys
from pathlib import Path
d, name = Path(sys.argv[1]), sys.argv[2]
log, cfg = d / "epoch_log.json", d / "train_config.json"
if not log.exists():
    print(f"  {name:<46} starting")
    raise SystemExit
h = json.loads(log.read_text())
best = max((r.get("val_macro_f1", 0) for r in h), default=0)
done = "DONE" if cfg.exists() else "running"
mins = ""
if cfg.exists():
    c = json.loads(cfg.read_text())
    mins = f" {c['wall_clock_seconds']/60:.0f}min"
print(f"  {name:<46} {len(h):>3} epochs  best_val_macroF1={best:.4f}  {done}{mins}")
PY
done
