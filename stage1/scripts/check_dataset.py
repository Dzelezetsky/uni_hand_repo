"""Smoke test + visual check of Stage1Dataset (mimic venv):  python stage1/scripts/check_dataset.py [n]
Prints shapes / timing per dataset and writes stage1_data/check_<dataset>.png: observed frames, future frames at
+0.4 s steps, with the target posture class / mask of each side under every future latent frame."""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage1.dataset import Stage1Dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NAMES = ["pinch", "power", "tool", "two-fing", "flat", "wide", "open"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ds = Stage1Dataset(REPO / "stage1_data", REPO, allow_dummy_t5=True, val_ratio=0.0, seed=0)
by = {}
for i, r in enumerate(ds.rows):
    by.setdefault(r["dataset"], []).append(i)
for d, idx in by.items():
    ts = []
    for k in range(n):
        t0 = time.time()
        s = ds.sample(idx[(k * 97) % len(idx)])
        ts.append(time.time() - t0)
    print(f"{d:22s} {len(idx):5d} eps  sample {np.mean(ts):.2f}s  video {tuple(s['video'].shape)} "
          f"pad {s['padding_mask'].mean():.2f}  fut mask L/R {s['hand/mask_fut'].float().mean(1).tolist()} "
          f"cls_fut R {s['hand/cls_fut'][1].tolist()}")
    # visual: choose a sample whose right-hand class changes inside the window, if any
    best = s
    for k in range(40):
        c = ds.sample(idx[(k * 31) % len(idx)])
        side = int(c["hand/mask_fut"][1].any())
        cf = c["hand/cls_fut"][side][c["hand/mask_fut"][side]]
        if len(cf) and len(set(cf.tolist() + [int(c["hand/cls_cur"][side])])) > 1:
            best = c
            break
    v = best["video"]
    cols = [0, 4] + [4 + 4 * m for m in range(1, 15, 2)]
    fig, ax = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 3))
    for a, f in zip(ax, cols):
        a.imshow(v[:, f].permute(1, 2, 0).numpy()); a.axis("off")
        m = (f - 4) // 4 - 1
        lab = []
        for si, sname in enumerate("LR"):
            if f == 4:
                c, ok = int(best["hand/cls_cur"][si]), bool(best["hand/mask_cur"][si])
            elif m >= 0:
                c, ok = int(best["hand/cls_fut"][si, m]), bool(best["hand/mask_fut"][si, m])
            else:
                continue
            lab.append(f"{sname}:{NAMES[c] if ok else '-'}")
        a.set_title(f"{(f - 4) / 10:+.1f}s\n" + " ".join(lab), fontsize=8)
    fig.suptitle(f"{best['meta/episode']} t={best['meta/t']:.1f}s", fontsize=9)
    fig.tight_layout(); fig.savefig(REPO / "stage1_data" / f"check_{d}.png", dpi=70); plt.close(fig)
