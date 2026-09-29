"""Which hand postures can each verified hand physically reach? (feasibility mask input, 2026-09-29)

Reachable sets are sampled in the NATIVE control space through the verified mappings (unidex/reachability.py).
Outputs unified/validation/reachability/:
  report.txt        per hand: reach of each feature (p1..p99), pinch reach, cross-hand coverage matrix
  coverage.png      cross-hand coverage matrix + 1D reach intervals

Cross-hand coverage C[A, B] = fraction of hand A's REAL frames (verified data) whose posture hand B can reproduce:
box-constrained least-squares fit of B's NATIVE parameters to A's 10 pairwise distances (unidex.reachability.best_fit,
INITS starts from the nearest reachable samples); reproduced = RMS residual < THR palm widths. THR is set from the
self-fit (B = A) error, which must be ~0. A posture that only some hands reach must go into the per-embodiment
feasibility mask, not into a shared class (GRASP_LABELING.md rule 5).

Pad angle features (angle between thumb and finger pad normals) are reported for pinch-like configurations
(thumb-finger distance < PINCH palm widths): if the reachable angle intervals do not overlap, a pad/side attribute
would encode the hand, not the grasp.

Usage: python scripts/analysis/reachability.py
"""
from __future__ import annotations

import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, ".")
from unidex.hands import REPO  # noqa: E402
from unidex.kinematics.canonical import FINGERS  # noqa: E402
from unidex.reachability import NATIVE_RANGES, best_fit, sample  # noqa: E402

OUT = REPO / "unified" / "validation" / "reachability"
PAIRS = list(combinations(range(5), 2))
PINCH = 0.35  # palm widths
THR = 0.05    # RMS palm widths (~4 mm) = worst self-fit error of the solver
INITS = 3
MAX_FRAMES = 1500
SHORT = {"realdex__shadow_e_right": "Shadow", "humanoid_everyday_h1__inspire_rh56dfx_right": "DFX-R",
         "humanoid_everyday_h1__inspire_rh56dfx_left": "DFX-L", "hrdexdb__inspire_rh56f1_right": "F1"}


def pairwise(p):
    return np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in PAIRS], 1)


def angle(n, i, j):
    return np.degrees(np.arccos(np.clip((n[:, i] * n[:, j]).sum(1), -1, 1)))


def real_frames(mapping_id, step_s=0.2):
    hs = pd.read_parquet(REPO / "unified" / "hand_streams.parquet")
    hs = hs[(hs.mapping_id == mapping_id) & (hs.canonicalization_status == "ok")]
    P = []
    for p in hs.path:
        f = pd.read_parquet(REPO / "unified" / p)
        f = f[f.valid]
        keep = np.r_[True, np.diff(np.floor(f.t_s.to_numpy() / step_s)) > 0]
        P.append(np.stack(f.fingertips_palm_norm.to_numpy())[keep])
    return np.concatenate(P).reshape(-1, 5, 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = list(NATIVE_RANGES)
    S = {m: sample(m) for m in ids}
    R = {m: real_frames(m) for m in ids}
    lines = ["Reachability of verified hands (native-space sampling through verified mappings)", ""]
    for m in ids:
        i = S[m]["info"]
        lines.append(f"{SHORT[m]:6s} {m}: kept {i['n_sampled'] * (1 - i['dropped_limit_frac']):.0f} samples "
                     f"(dropped outside URDF limits {i['dropped_limit_frac']:.1%}); range: {i['range_source']}")
    # 1D reach of each pairwise distance
    lines += ["", "Reach of pairwise fingertip distances, palm widths [p1, p99]:",
              f"{'pair':6s} " + " ".join(f"{SHORT[m]:>14s}" for m in ids)]
    D = {m: pairwise(S[m]["tips_norm"]) for m in ids}
    for k, (a, b) in enumerate(PAIRS):
        lines.append(f"{FINGERS[a][0].upper()}-{FINGERS[b][0].upper():4s} " + " ".join(
            f"  [{np.percentile(D[m][:, k], 1):4.2f},{np.percentile(D[m][:, k], 99):4.2f}]" for m in ids))
    # pinch reach and pad angles in pinch configurations
    lines += ["", f"Pinch configurations (thumb-finger distance < {PINCH} pw): share of reachable samples and "
                  "angle(thumb pad normal, finger pad normal) p5/p50/p95 deg (180 = pads face each other)"]
    for f in (1, 2):
        for m in ids:
            d = np.linalg.norm(S[m]["tips_norm"][:, 0] - S[m]["tips_norm"][:, f], axis=1)
            sel = d < PINCH
            if sel.sum() < 20:
                lines.append(f"  thumb-{FINGERS[f]:6s} {SHORT[m]:6s} min distance {d.min():.2f} pw -> no pinch reachable")
                continue
            a = angle(S[m]["normals"][sel], 0, f)
            lines.append(f"  thumb-{FINGERS[f]:6s} {SHORT[m]:6s} share {sel.mean():6.2%}  min d {d.min():.2f}  "
                         f"angle {np.percentile(a, 5):4.0f}/{np.percentile(a, 50):4.0f}/{np.percentile(a, 95):4.0f}  max {a.max():4.0f}")
    # cross-hand coverage of real postures (exact fit, not nearest sample)
    rng = np.random.default_rng(0)
    Dr = {}
    for m in ids:
        d = pairwise(R[m])
        Dr[m] = d[rng.choice(len(d), min(MAX_FRAMES, len(d)), replace=False)]
    C = np.zeros((len(ids), len(ids)))
    miss_by_pair = {}
    for j, b in enumerate(ids):
        tree = cKDTree(D[b])
        for i, a in enumerate(ids):
            nn = tree.query(Dr[a], k=INITS)[1]
            best = None
            for k in range(INITS):
                _, r = best_fit(b, pairwise, Dr[a], S[b]["raw"][nn[:, k]])
                best = r if best is None else np.where(((r ** 2).sum(1) < (best ** 2).sum(1))[:, None], r, best)
            rms = np.sqrt((best ** 2).mean(1))
            C[i, j] = (rms < THR).mean()
            if i != j:
                miss_by_pair[(a, b)] = (np.abs(best[rms >= THR]).mean(0) if (rms >= THR).any() else None, rms)
    lines += ["", f"Cross-hand coverage (exact native-space fit, reproduced = RMS < {THR} pw):",
              "row = hand whose REAL frames are tested, column = hand that must reproduce them",
              f"{'':8s}" + "".join(f"{SHORT[m]:>8s}" for m in ids)]
    for i, a in enumerate(ids):
        lines.append(f"{SHORT[a]:8s}" + "".join(f"{C[i, j]:8.2f}" for j in range(len(ids))) + f"   (n={len(Dr[a])})")
    lines += ["", "Where misses come from: mean |residual| per pairwise distance over NOT reproduced frames (pw)",
              f"{'A -> B':16s}" + "".join(f"{FINGERS[a][0].upper() + '-' + FINGERS[b][0].upper():>6s}" for a, b in PAIRS)]
    for (a, b), (mr, rms) in miss_by_pair.items():
        if mr is not None:
            lines.append(f"{SHORT[a] + ' -> ' + SHORT[b]:16s}" + "".join(f"{v:6.2f}" for v in mr) +
                         f"   median rms {np.median(rms):.3f}")
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[1, 2.2]))
    ax = axes[0]
    ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
    names = [SHORT[m] for m in ids]
    ax.set_xticks(range(len(ids)), names); ax.set_yticks(range(len(ids)), names)
    for i in range(len(ids)):
        for j in range(len(ids)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="w" if C[i, j] < .6 else "k")
    ax.set_xlabel("hand that must reach"); ax.set_ylabel("real frames of"); ax.set_title("cross-hand coverage")
    ax = axes[1]
    for k, m in enumerate(ids):
        lo, hi = np.percentile(D[m], 1, 0), np.percentile(D[m], 99, 0)
        x = np.arange(len(PAIRS)) + (k - 1.5) * 0.18
        ax.vlines(x, lo, hi, lw=5, color=f"C{k}", label=SHORT[m], alpha=.8)
        med = np.median(pairwise(R[m]), 0)
        ax.plot(x, med, "k_", ms=10)
    ax.set_xticks(range(len(PAIRS)), [f"{FINGERS[a][0].upper()}-{FINGERS[b][0].upper()}" for a, b in PAIRS])
    ax.set_ylabel("palm widths"); ax.legend(fontsize=8)
    ax.set_title("reachable interval p1..p99 per pairwise distance (black tick = median of real frames)")
    fig.tight_layout(); fig.savefig(OUT / "coverage.png", dpi=70)


if __name__ == "__main__":
    main()
