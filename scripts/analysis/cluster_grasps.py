"""Cluster GRASP moments (grasp_moments_v1) of the verified hands. Unit of analysis = one hold (a contiguous
in_interaction segment, gaps < 0.3 s merged, >= 0.5 s long); its posture = median of 10 pairwise fingertip
distances (palm widths). Hierarchical clustering (Ward) + dendrogram; one representative RGB frame per hold.
Outputs -> unified/clustering/grasps_*"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from unidex.viz import read_frame  # noqa: E402

UNI = Path("unified")
OUT = UNI / "clustering"
mom = pd.read_parquet(UNI / "derived/grasp_moments_v1.parquet")
hs = pd.read_parquet(UNI / "hand_streams.parquet")
cams = pd.read_parquet(UNI / "cameras.parquet")
PAIRS = list(combinations(range(5), 2))
FN = "TIMRP"
FEAT = [f"d_{FN[i]}{FN[j]}" for i, j in PAIRS]

holds = []
for (ds, ep, side), m in mom.groupby(["dataset_id", "source_episode_id", "side"]):
    r = hs[(hs.dataset_id == ds) & (hs.source_episode_id == ep) & (hs.side == side)].iloc[0]
    fr = pd.read_parquet(UNI / r.path).set_index("frame_index")
    m = m.sort_values("t_s")
    on = m.in_interaction.to_numpy(); t = m.t_s.to_numpy(); fi = m.frame_index.to_numpy()
    seg, start = [], None
    for k in range(len(on)):
        if on[k] and start is None:
            start = k
        if start is not None and (not on[k] or k == len(on) - 1):
            end = k if on[k] else k - 1
            seg.append([start, end]); start = None
    merged = []
    for s in seg:  # merge gaps < 0.3 s
        if merged and t[s[0]] - t[merged[-1][1]] < 0.3:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    for a, b in merged:
        if t[b] - t[a] < 0.5:
            continue
        P = np.stack(fr.loc[fi[a:b + 1]].fingertips_palm_norm.to_numpy()).reshape(-1, 5, 3)
        D = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in PAIRS], 1)
        mid = fi[(a + b) // 2]
        holds.append(dict(dataset=ds, episode=ep, hand=r.hand_model_id, t0=t[a], t1=t[b], dur=t[b] - t[a],
                          mid_frame=int(mid), mid_t=float(t[(a + b) // 2]), posture=np.median(D, 0),
                          within_std=float(D.std(0).mean()), tips=np.median(P, 0)))
H = pd.DataFrame(holds)
import yaml  # noqa: E402
CAT = {}
for cat, v in yaml.safe_load(open("config/object_categories.yaml"))["categories"].items():
    for ds, objs in v.items():
        for o in objs:
            CAT[(ds, o)] = cat
H["object"] = [e.split("/")[1] if d == "hrdexdb" else e.split("/")[0] for d, e in zip(H.dataset, H.episode)]
H["category"] = [CAT.get((d, o), "other") for d, o in zip(H.dataset, H.object)]
H["label"] = [f"{'Shadow' if 'shadow' in h else 'F1'}|{c}|{o}@{a:.0f}s"
              for h, c, o, a in zip(H.hand, H.category, H.object, H.t0)]
X = np.stack(H.posture.to_numpy())
print(f"holds: {len(H)}  ({H.hand.value_counts().to_dict()})  mean within-hold std {H.within_std.mean():.3f} vs "
      f"between-hold std {X.std(0).mean():.3f} (palm widths)")

# ---- how much posture variance is explained by object category vs by hand (holds weighted so that every
# episode has total weight 1, otherwise long RealDex sequences with many re-grasps dominate)
w = 1.0 / H.groupby("episode").episode.transform("size").to_numpy()
both = H[H.category.isin(H.groupby("category").hand.nunique().loc[lambda s: s > 1].index)]


def r2(factor, df):
    Xf = np.stack(df.posture.to_numpy()); ww = 1.0 / df.groupby("episode").episode.transform("size").to_numpy()
    mu = np.average(Xf, 0, ww); tot = (ww[:, None] * (Xf - mu) ** 2).sum()
    g = df[factor].to_numpy(); expl = 0.0
    for v in np.unique(g):
        m = g == v; mv = np.average(Xf[m], 0, ww[m]); expl += ww[m].sum() * ((mv - mu) ** 2).sum()
    return expl / tot


if len(both):
    both = both.assign(cell=both.category + "|" + both.hand)
    print(f"\nmatched categories with both hands: {sorted(both.category.unique())}, holds {len(both)}")
    print(f"R2 category = {r2('category', both):.2f}   R2 hand = {r2('hand', both):.2f}   "
          f"R2 category x hand cells = {r2('cell', both):.2f}")
    per = both.groupby(["category", "hand"]).apply(lambda d: pd.Series(np.average(np.stack(d.posture), 0,
                                                   1.0 / d.groupby("episode").episode.transform("size")), index=FEAT))
    print("\nmean posture per category x hand (palm widths):\n", per.round(2).to_string())
    per.to_csv(OUT / "grasps_category_hand_means.csv")

Z = linkage(X, "ward")
res = []
for k in range(2, min(8, len(H) - 1)):
    lab = fcluster(Z, k, "maxclust")
    res.append((k, silhouette_score(X, lab), [pd.Series(H.hand[lab == c]).value_counts().to_dict() for c in np.unique(lab)]))
for k, s, comp in res:
    print(f"k={k} silhouette={s:.2f} composition={comp}")

fig, ax = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw=dict(width_ratios=[1.3, 1]))
dendrogram(Z, labels=H.label.tolist(), orientation="left", ax=ax[0], leaf_font_size=8)
CC = dict(zip(sorted(H.category.unique()), plt.cm.tab10.colors))
for tl in ax[0].get_ymajorticklabels():
    tl.set_color(CC[tl.get_text().split("|")[1]])
ax[0].set_title("Ward dendrogram of holds: label = hand|category|object, color = category")
order = dendrogram(Z, no_plot=True)["leaves"]
im = ax[1].imshow(X[order], aspect="auto", cmap="viridis")
ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(H.label.iloc[order], fontsize=7)
ax[1].set_xticks(range(10)); ax[1].set_xticklabels(FEAT, rotation=60)
plt.colorbar(im, ax=ax[1], label="palm widths"); ax[1].set_title("hold posture = pairwise fingertip distances")
fig.tight_layout(); fig.savefig(OUT / "grasps_dendrogram.png", dpi=60)

# gallery: RGB frame at hold middle + fingertip rays; <= 3 holds per (hand, object), grouped by category
gal = H.assign(i=range(len(H))).sort_values(["category", "hand", "object", "t0"]).groupby(["hand", "object"]).head(3)
order = gal.i.tolist()
n = len(order); cols = 6; rows_ = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows_ * 2, cols, figsize=(3.3 * cols, 5.6 * rows_))
COL = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
for j, i in enumerate(order):
    h = H.iloc[i]; r0, c0 = 2 * (j // cols), j % cols
    c = cams[(cams.dataset_id == h.dataset) & (cams.source_episode_id == h.episode) & cams.local]
    c = c[c.camera_id == "22641023"].iloc[0] if (c.camera_id == "22641023").any() else c.iloc[0]
    fi = h.mid_frame
    if h.dataset == "hrdexdb":
        cf = pd.read_parquet(UNI / "episodes/hrdexdb" / h.episode.replace("/", "__") / "camera_frames.parquet")
        cf = cf[cf.camera_id == c.camera_id].dropna()
        fi = int(cf.frame_index.iloc[np.argmin(np.abs(cf.t_s - h.mid_t))])
    img = read_frame(c, fi)
    if img is not None:
        axs[r0, c0].imshow(img)
    axs[r0, c0].set_title(h.label, fontsize=8); axs[r0, c0].axis("off")
    for k in range(5):
        axs[r0 + 1, c0].plot([0, h.tips[k, 0]], [0, h.tips[k, 1]], "-o", color=COL[k], ms=3)
        axs[r0 + 1, c0].plot([0, h.tips[k, 2] + 2.2], [0, h.tips[k, 1]], "-o", color=COL[k], ms=3, alpha=.6)
    axs[r0 + 1, c0].set_xlim(-1.2, 3.8); axs[r0 + 1, c0].set_ylim(-1.8, 1.8); axs[r0 + 1, c0].set_aspect("equal")
    axs[r0 + 1, c0].set_title("front X/Y | side Z/Y (shifted)", fontsize=7)
for a in axs.ravel()[2 * n:]:
    a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "grasps_gallery.png", dpi=55)
H.drop(columns=["posture", "tips"]).assign(**{f: X[:, k] for k, f in enumerate(FEAT)}).to_csv(OUT / "grasps_holds.csv", index=False)
print("saved")
