"""Keep only the last N checkpoints (+ the best one by validation hand/ce) of a Stage-1 run to save disk.
Each checkpoint iteration = model/iter_X.pt (~4 GB) + model/iter_X_fused.pt (~4 GB) + optim/ + scheduler/ + trainer/.
usage: python stage1/prune_checkpoints.py <run_dir> [--keep 3] [--metric hand/ce] [--dry-run]
(run_dir = stage1_runs/unidex_stage1/stage1/<job name>; the file latest_checkpoint.txt is never touched)"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--keep", type=int, default=3)
ap.add_argument("--metric", default="hand/ce")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
ck = a.run_dir / "checkpoints"
iters = sorted({int(m.group(1)) for p in (ck / "model").glob("iter_*.pt") if (m := re.match(r"iter_(\d+)", p.name))})
keep = set(iters[-a.keep:])
mf = a.run_dir / "stage1_metrics.jsonl"
if mf.exists():
    val = [json.loads(l) for l in mf.read_text().splitlines() if '"val"' in l]
    val = [v for v in val if a.metric in v and v["iter"] in iters]
    if val:
        best = min(val, key=lambda v: v[a.metric])["iter"]
        keep.add(best)
        print(f"best by val {a.metric}: iter {best} ({min(v[a.metric] for v in val):.4f})")
for it in iters:
    if it in keep:
        continue
    for p in ck.glob(f"*/iter_{it:09d}*"):
        print(("would delete " if a.dry_run else "delete ") + str(p))
        if not a.dry_run:
            p.unlink()
print(f"kept iterations: {sorted(keep)}")
