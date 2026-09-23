"""Exploratory posture clustering on the VERIFIED canonical hand geometry (GRASP_LABELING.md rules).

- one sample per DT seconds per stream (removes temporal autocorrelation)
- balanced: at most CAP samples per hand model
- features: 10 pairwise fingertip distances (palm widths)  [+ raw 15D for comparison]
- embedding: PCA + UMAP (visual only), clustering: HDBSCAN + KMeans/GMM sweep
- checks: silhouette, bootstrap stability (ARI), per-cluster embodiment composition
Outputs -> unified/clustering/
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
DT, CAP, SEED = 0.2, 1500, 0
OUT = Path("unified/clustering"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

hs = pd.read_parquet("unified/hand_streams.parquet")
hs = hs[hs.canonicalization_status == "ok"]
rows = []
for _, r in hs.iterrows():
    fr = pd.read_parquet("unified/" + r.path)
    keep = np.r_[True, np.diff(np.floor(fr.t_s.to_numpy() / DT)) > 0]
    fr = fr[keep & fr.valid.to_numpy()]
    for _, x in fr.iterrows():
        rows.append(dict(hand=r.hand_model_id.replace("_right", "").replace("_left", "") + "/" + r.side,
                         dataset=r.dataset_id, episode=r.source_episode_id, side=r.side, t_s=x.t_s,
                         frame_index=int(x.frame_index), tips=np.asarray(x.fingertips_palm_norm, float)))
df = pd.DataFrame(rows)
df = pd.concat([g.sample(min(len(g), CAP), random_state=SEED) for _, g in df.groupby("hand")], ignore_index=True)
P = np.stack(df.tips.to_numpy()).reshape(-1, 5, 3)
pairs = list(combinations(range(5), 2))
FN = "TIMRP"
feat_names = [f"d_{FN[i]}{FN[j]}" for i, j in pairs]
Xd = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in pairs], 1)
X = StandardScaler().fit_transform(Xd)
print("samples per hand:", df.hand.value_counts().to_dict())

# ---------- embeddings
import umap  # noqa: E402
emb_pca = X @ np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:2].T
emb_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(X)

# ---------- HDBSCAN
hdb = HDBSCAN(min_cluster_size=max(30, len(X) // 100), min_samples=10).fit(X)
lab = hdb.labels_
print(f"HDBSCAN: {lab.max() + 1} clusters, noise {np.mean(lab < 0):.0%}")

# ---------- KMeans / GMM sweep
sweep = []
for k in range(2, 13):
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    gm = GaussianMixture(k, covariance_type="full", random_state=SEED).fit(X)
    # stability: KMeans on bootstrap resamples, ARI vs full-data labels on the shared points
    aris = []
    for b in range(5):
        idx = rng.choice(len(X), len(X), replace=True)
        kb = KMeans(k, n_init=5, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(X)))
    sweep.append(dict(k=k, silhouette=silhouette_score(X, km.labels_, sample_size=3000, random_state=SEED),
                      gmm_bic=gm.bic(X), stability_ari=np.mean(aris)))
sweep = pd.DataFrame(sweep)
print(sweep.round(3).to_string(index=False))

# null model: same marginals, dependencies destroyed (shuffle each feature independently)
Xn = np.column_stack([rng.permutation(c) for c in X.T])
null_sil = {k: silhouette_score(Xn, KMeans(k, n_init=10, random_state=SEED).fit_predict(Xn), sample_size=3000,
                                random_state=SEED) for k in (3, 5, 8)}
print("null-model silhouette:", {k: round(v, 3) for k, v in null_sil.items()})

best_k = int(sweep.sort_values("silhouette", ascending=False).k.iloc[0])
km = KMeans(best_k, n_init=20, random_state=SEED).fit(X)
df["kmeans"] = km.labels_
df["hdbscan"] = lab
df["umap_x"], df["umap_y"] = emb_umap[:, 0], emb_umap[:, 1]


def composition(col):
    t = pd.crosstab(df[col], df.hand, normalize="index").round(2)
    t["n"] = df[col].value_counts().sort_index()
    t["n_hands_ge10pct"] = (t.drop(columns="n") >= 0.10).sum(1)
    return t


comp_k, comp_h = composition("kmeans"), composition("hdbscan")
print(f"\nKMeans k={best_k} composition:\n", comp_k.to_string())
print("\nHDBSCAN composition:\n", comp_h.to_string())

# cluster centroids in interpretable units (palm widths)
cent = pd.DataFrame(Xd, columns=feat_names).groupby(df.kmeans).median().round(2)
print("\nKMeans cluster medians (palm widths):\n", cent.to_string())

df.drop(columns="tips").to_parquet(OUT / "assignments.parquet")
np.save(OUT / "tips_norm.npy", P)
json.dump(dict(best_k=best_k, sweep=sweep.to_dict("records"), null_silhouette=null_sil,
               hdbscan_clusters=int(lab.max() + 1), hdbscan_noise=float(np.mean(lab < 0)),
               features=feat_names, dt=DT, cap=CAP), open(OUT / "summary.json", "w"), indent=1, default=float)

# ---------- figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(2, 3, figsize=(20, 12))
for h in sorted(df.hand.unique()):
    m = df.hand == h
    ax[0, 0].scatter(emb_pca[m, 0], emb_pca[m, 1], s=3, alpha=.4, label=h)
    ax[0, 1].scatter(emb_umap[m, 0], emb_umap[m, 1], s=3, alpha=.4, label=h)
ax[0, 0].set_title("PCA, colored by hand"); ax[0, 1].set_title("UMAP, colored by hand")
ax[0, 0].legend(markerscale=4, fontsize=8)
ax[0, 2].scatter(emb_umap[:, 0], emb_umap[:, 1], c=np.where(lab < 0, -1, lab), s=3, cmap="tab20")
ax[0, 2].set_title(f"UMAP, HDBSCAN ({lab.max() + 1} clusters, grey/-1 = noise {np.mean(lab < 0):.0%})")
ax[1, 0].scatter(emb_umap[:, 0], emb_umap[:, 1], c=km.labels_, s=3, cmap="tab10")
for c in range(best_k):
    xy = emb_umap[km.labels_ == c].mean(0); ax[1, 0].text(*xy, str(c), fontsize=14, weight="bold")
ax[1, 0].set_title(f"UMAP, KMeans k={best_k}")
ax[1, 1].plot(sweep.k, sweep.silhouette, "o-", label="silhouette")
ax[1, 1].plot(sweep.k, sweep.stability_ari, "s-", label="bootstrap ARI")
ax[1, 1].axhline(null_sil[5], ls="--", c="grey", label="null silhouette (k=5)")
ax[1, 1].legend(); ax[1, 1].set_xlabel("k"); ax[1, 1].set_title("KMeans model selection")
comp_k.drop(columns=["n", "n_hands_ge10pct"]).plot.bar(stacked=True, ax=ax[1, 2])
ax[1, 2].set_title("hand composition per KMeans cluster"); ax[1, 2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "overview.png", dpi=60)

# representative posture per cluster (medoid per hand), skeleton-free: fingertips + palm knuckle line
COL = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
fig, ax = plt.subplots(2, best_k, figsize=(3.2 * best_k, 7))
for c in range(best_k):
    m = km.labels_ == c
    med = P[m][np.argmin(np.linalg.norm(X[m] - X[m].mean(0), axis=1))]
    for r, (a, b, lab_) in enumerate([(0, 1, "front X/Y"), (2, 1, "side Z/Y")]):
        for k in range(5):
            ax[r, c].plot([0, med[k, a]], [0, med[k, b]], "-o", color=COL[k], ms=4)
        ax[r, c].set_xlim(-1.5, 2.5) if a == 0 else ax[r, c].set_xlim(-1.5, 2.5)
        ax[r, c].set_ylim(-2, 2.5); ax[r, c].set_aspect("equal"); ax[r, c].grid(alpha=.3)
        ax[r, c].set_title(f"cluster {c} (n={m.sum()}) {lab_}", fontsize=8)
fig.suptitle("KMeans cluster medoids: rays from knuckle centroid to fingertips (T red, I orange, M green, R blue, P purple)")
fig.tight_layout(); fig.savefig(OUT / "medoids.png", dpi=60)
print("saved", OUT)
