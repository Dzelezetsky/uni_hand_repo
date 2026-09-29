"""Gallery of cluster representatives: N per cluster, nearest to the centroid, at most one per episode, families taken
in turn (round robin over families with >= MIN_SHARE of the cluster). Each representative = the real camera frame at
that moment (top; "no local rgb" for RealDex objects downloaded poses-only) + a mesh render of the exact hand pose
(bottom), because the hand is often small or outside the camera view.

usage: python scripts/analysis/cluster_gallery.py [assignments tag, e.g. all_k5] [N]
output: unified/clustering/v2_gallery_<tag>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from unidex.hands import REPO, load_hand  # noqa: E402
from unidex.viz import read_frame  # noqa: E402
from unidex.viz_mesh import draw_mesh, hand_triangles  # noqa: E402

UNI = REPO / "unified"
TAG = sys.argv[1] if len(sys.argv) > 1 else "all_k5"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MIN_SHARE = 0.05
A = pd.read_parquet(UNI / "clustering" / f"v2_assignments_{TAG}.parquet")
cams = pd.read_parquet(UNI / "cameras.parquet")
cams = cams[cams.local]
hs = pd.read_parquet(UNI / "hand_streams.parquet")
hs = hs[hs.canonicalization_status == "ok"].set_index(["dataset_id", "source_episode_id", "side"])
PREF = {"hrdexdb": "22641023"}  # camera used for the visual checks


def image(row):
    c = cams[(cams.dataset_id == row.dataset) & (cams.source_episode_id == row.episode)]
    if not len(c):
        return None
    pref = c[c.camera_id == PREF.get(row.dataset, "")]
    cam = (pref if len(pref) else c).iloc[0]
    ep_dir = UNI / "episodes" / row.dataset / row.episode.replace("/", "__")
    if (ep_dir / "camera_frames.parquet").exists():
        cf = pd.read_parquet(ep_dir / "camera_frames.parquet")
        cf = cf[cf.camera_id == cam.camera_id].dropna(subset=["t_s"])
        if not len(cf):
            return None
        fi = int(cf.frame_index.iloc[np.argmin(np.abs(cf.t_s.to_numpy() - row.t_s))])
    elif row.dataset == "realdex":
        fi = int(row.frame_index)  # hand rows are one-to-one with RGB frames
    else:
        return None
    try:
        return read_frame(cam, fi)
    except Exception:
        return None


def mesh(ax, row):
    r = hs.loc[(row.dataset, row.episode, row.side)]
    f = pd.read_parquet(UNI / r.path, columns=["frame_index", "model_q"])
    q = np.asarray(f.model_q.iloc[int(np.argmin(np.abs(f.frame_index.to_numpy() - row.frame_index)))])[None]
    hand = load_hand(row.hand_model_id)
    draw_mesh(ax, hand_triangles(row.hand_model_id, hand, q, list(r.model_joint_names)),
              np.array([-0.6, 0, -0.8]), np.array([0, 1.0, 0]))
    ax.set_aspect("equal"); ax.autoscale_view()


def representatives(g):
    share = g.family.value_counts(normalize=True)
    fams = [f for f in share.index if share[f] >= MIN_SHARE]
    pools = {f: g[g.family == f].sort_values("dist_to_centroid").drop_duplicates("episode") for f in fams}
    out, i = [], 0
    while len(out) < N and any(len(p) for p in pools.values()):
        f = fams[i % len(fams)]; i += 1
        if len(pools[f]):
            out.append(pools[f].iloc[0]); pools[f] = pools[f].iloc[1:]
    return out


ranks = sorted(A.display_rank.unique())
fig, axes = plt.subplots(2 * len(ranks), N, figsize=(2.9 * N, 5.2 * len(ranks)),
                         gridspec_kw=dict(height_ratios=[1.0, 1.1] * len(ranks)))
for r, rank in enumerate(ranks):
    g = A[A.display_rank == rank]
    comp = g.family.value_counts(normalize=True).round(2).to_dict()
    for j in range(N):
        axes[2 * r, j].axis("off"); axes[2 * r + 1, j].axis("off")
    for j, row in enumerate(representatives(g)):
        ax, axm = axes[2 * r, j], axes[2 * r + 1, j]
        img = image(row)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "no local rgb\n(poses-only download)", ha="center", va="center", fontsize=8,
                    transform=ax.transAxes)
        ax.set_title(f"{row.family} | {row.episode[-24:]}\nt={row.t_s:.1f}s", fontsize=7)
        mesh(axm, row)
    axes[2 * r, 0].text(-0.1, 0.0, f"CLUSTER {rank + 1}\n(id {g.cluster.iloc[0]})\n" +
                        "\n".join(f"{k} {v}" for k, v in comp.items()), transform=axes[2 * r, 0].transAxes,
                        ha="right", va="center", fontsize=11, fontweight="bold")
fig.suptitle(f"{TAG}: {N} representatives per cluster (nearest to centroid, one per episode, families in turn); "
             "top = camera frame, bottom = exact hand pose (mesh, oblique palm view)", fontsize=13)
fig.tight_layout(rect=(0.06, 0, 1, 0.98))
out = UNI / "clustering" / f"v2_gallery_{TAG}.png"
fig.savefig(out, dpi=55)
print(out)
