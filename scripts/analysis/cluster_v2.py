"""Posture / grasp clustering on the scaled verified corpus (2026-09-29). GRASP_LABELING.md rules A-C apply:
features = 10 pairwise fingertip distances (palm widths), hand-balanced, no contact/object inputs.

Populations
  ALL    every valid sample on a 0.2 s grid; hand FAMILIES balanced (Shadow, Inspire DFX L+R, Inspire F1)
  GRASP  interaction samples only (grasp_moments_v1: F1 object lifted > 3 cm, Shadow authors' contact.txt);
         H1 has no interaction signal and is absent. Used only to INTERPRET clusters (object categories), never as
         a clustering input for ALL.

Per k (KMeans, hand-balanced fit): silhouette, bootstrap stability (mean ARI of 10 refits on resamples),
NMI(cluster, family), single-family clusters. For the chosen k: per-cluster family composition, interaction rate
(share of the cluster's F1/Shadow samples that are grasps), feasibility (can each family reproduce the cluster
centroid? native-space fit, RMS < 0.05 pw), objects most over-represented among grasp samples, and a figure with the
sample nearest to the centroid of each family rendered as a mesh.

Outputs unified/clustering/v2_*
usage: python scripts/analysis/cluster_v2.py [k_all] [k_grasp]
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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score  # noqa: E402

sys.path.insert(0, ".")
from unidex.hands import REPO, load_hand  # noqa: E402
from unidex.reachability import best_fit, sample  # noqa: E402
from unidex.viz_mesh import draw_mesh, hand_triangles  # noqa: E402

UNI, OUT = REPO / "unified", REPO / "unified" / "clustering"
DT, CAP, SEED, KS, NBOOT = 0.2, 4000, 0, range(2, 13), 10
PAIRS = list(combinations(range(5), 2))
FN = "TIMRP"
FEAT = [f"{FN[i]}{FN[j]}" for i, j in PAIRS]
FAMILY = {"shadow_e_right_realdex": "Shadow", "inspire_rh56dfx_right_unitree": "DFX",
          "inspire_rh56dfx_left_unitree": "DFX", "inspire_rh56f1_right_hrdexdb": "F1",
          "xhand1_right": "XHand", "xhand1_left": "XHand", "sharpa_wave_right": "Sharpa", "sharpa_wave_left": "Sharpa",
          "ruiyan_ryh2_right_egosteer": "Ruiyan", "ruiyan_ryh2_left_egosteer": "Ruiyan"}
FAMS = ["Shadow", "DFX", "F1", "XHand", "Sharpa", "Ruiyan"]
FIT_MAPPING = {"Shadow": "realdex__shadow_e_right", "DFX": "humanoid_everyday_h1__inspire_rh56dfx_right",
               "F1": "hrdexdb__inspire_rh56f1_right", "XHand": "dexora__xhand1_right",
               "Sharpa": "trex__sharpa_wave_right", "Ruiyan": "egosteer__ruiyan_ryh2_right"}
POOL = 20000  # samples drawn per family before balancing (per-stream budget = POOL / #streams of that family)
rng = np.random.default_rng(SEED)


def dists(P):
    return np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in PAIRS], 1)


# ------------------------------------------------------------------ load
hs = pd.read_parquet(UNI / "hand_streams.parquet")
hs = hs[hs.canonicalization_status == "ok"]
mom = pd.read_parquet(UNI / "derived/grasp_moments_v1.parquet")
mom = mom[~mom.moment_source.str.startswith("excluded")]
mkey = mom.set_index(["source_episode_id", "side", "frame_index"]).in_interaction
hs["family"] = hs.hand_model_id.map(FAMILY)
n_streams = hs.family.value_counts()
# large families: read a random subset of streams (>= 3 samples each) instead of all ~15-23 k streams
hs = pd.concat([g.sample(min(len(g), max(POOL // 3, 1)), random_state=SEED) for _, g in hs.groupby("family")])
budget = {f: max(3, int(np.ceil(POOL / min(n, POOL // 3)))) for f, n in n_streams.items()}
trex_meta = None
if (hs.dataset_id == "trex").any():
    from unidex.adapters.trex import _meta as _trex_meta
    trex_meta = _trex_meta()[["motor_primitive", "object"]]
man = pd.read_parquet(UNI / "manifest.parquet", columns=["dataset_id", "source_episode_id", "instruction_original"])
ego_task = man[man.dataset_id == "egosteer"].set_index("source_episode_id").instruction_original
rows = []
for _, r in hs.iterrows():
    fr = pd.read_parquet(UNI / r.path, columns=["frame_index", "t_s", "valid", "fingertips_palm_norm", "model_q"])
    keep = np.r_[True, np.diff(np.floor(fr.t_s.to_numpy() / DT)) > 0] & fr.valid.to_numpy()
    fr = fr[keep]
    if len(fr) > budget[r.family]:
        fr = fr.iloc[np.sort(rng.choice(len(fr), budget[r.family], replace=False))]
    has_sig = r.dataset_id in ("hrdexdb", "realdex") and \
        (r.source_episode_id, r.side, int(fr.frame_index.iloc[0])) in mkey.index if len(fr) else False
    inter = [bool(mkey.get((r.source_episode_id, r.side, int(f)), False)) for f in fr.frame_index] if has_sig \
        else [None] * len(fr)
    rows.append(pd.DataFrame({"family": r.family, "hand_model_id": r.hand_model_id,
                              "dataset": r.dataset_id, "episode": r.source_episode_id, "side": r.side,
                              "frame_index": fr.frame_index.to_numpy(), "t_s": fr.t_s.to_numpy(),
                              "tips": list(np.stack(fr.fingertips_palm_norm.to_numpy())),
                              "model_q": list(np.stack(fr.model_q.to_numpy())),
                              "names": [tuple(r.model_joint_names)] * len(fr), "inter": inter,
                              "primitive": (trex_meta.loc[int(r.source_episode_id.split("_")[-1]), "motor_primitive"]
                                            if r.dataset_id == "trex" else None),
                              "trex_object": (trex_meta.loc[int(r.source_episode_id.split("_")[-1]), "object"]
                                              if r.dataset_id == "trex" else None),
                              "ego_task": ego_task.get(r.source_episode_id) if r.dataset_id == "egosteer" else None}))
df = pd.concat(rows, ignore_index=True)
df["object"] = np.where(df.dataset == "hrdexdb", df.episode.str.split("/").str[1],
                        np.where(df.dataset == "realdex", df.episode.str.split("/").str[0], None))
print("samples per family:", df.family.value_counts().to_dict())


def balanced(d, cap=CAP):
    return pd.concat([g.sample(min(len(g), cap), random_state=SEED) for _, g in d.groupby("family")],
                     ignore_index=True)


def sweep(X, fam, name, lines):
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    lines.append(f"\n[{name}] n={len(X)} per family {pd.Series(fam).value_counts().to_dict()}")
    lines.append(f"{'k':>3s} {'silh':>6s} {'stabARI':>8s} {'NMIfam':>7s} {'1-fam':>6s} {'min%':>6s}")
    res = {}
    for k in KS:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        lab = km.labels_
        aris = []
        for b in range(NBOOT):
            idx = rng.choice(len(Z), len(Z), replace=True)
            aris.append(adjusted_rand_score(lab, KMeans(k, n_init=3, random_state=b).fit(Z[idx]).predict(Z)))
        comp = pd.crosstab(lab, fam, normalize="columns")  # share of each family's samples per cluster
        compn = comp.div(comp.sum(1), axis=0)              # family-balanced composition
        res[k] = (km, lab)
        lines.append(f"{k:3d} {silhouette_score(Z, lab, sample_size=4000, random_state=0):6.3f} "
                     f"{np.mean(aris):8.3f} {normalized_mutual_info_score(fam, lab):7.3f} "
                     f"{int((compn.max(1) > 0.8).sum()):6d} {np.bincount(lab).min() / len(lab):6.1%}")
    return res, (mu, sd)


lines = ["Posture / grasp clustering v2 (10 pairwise distances, palm widths; KMeans, family-balanced)"]
ALL = balanced(df)
Xa = dists(np.stack(ALL.tips.to_numpy()).reshape(-1, 5, 3))
res_all, norm_all = sweep(Xa, ALL.family.to_numpy(), "ALL postures", lines)
GR = df[df.inter == True]  # noqa: E712
GR = balanced(GR, cap=3000)
Xg = dists(np.stack(GR.tips.to_numpy()).reshape(-1, 5, 3))
res_gr, norm_gr = sweep(Xg, GR.family.to_numpy(), "GRASP samples (F1 lifted, Shadow contact)", lines)
print("\n".join(lines))
if len(sys.argv) < 3:
    (OUT / "v2_sweep.txt").write_text("\n".join(lines) + "\n")
    sys.exit(0)

K_ALL, K_GR = int(sys.argv[1]), int(sys.argv[2])
S = {f: sample(FIT_MAPPING[f], 60_000) for f in FAMS}
SD = {f: dists(S[f]["tips_norm"]) for f in FAMS}
TREES = {f: cKDTree(SD[f]) for f in FAMS}


def feasible(centroids):
    out = {}
    for f in FAMS:
        nn = TREES[f].query(centroids, k=3)[1]
        best = None
        for k in range(3):
            _, r = best_fit(FIT_MAPPING[f], dists, centroids, S[f]["raw"][nn[:, k]])
            best = r if best is None else np.where(((r ** 2).sum(1) < (best ** 2).sum(1))[:, None], r, best)
        out[f] = np.sqrt((best ** 2).mean(1))
    return out


def describe(D, X, km, norm, name, k):
    mu, sd = norm
    lab = km.labels_
    cent = km.cluster_centers_ * sd + mu
    fam = D.family.to_numpy()
    comp = pd.crosstab(lab, fam, normalize="columns").reindex(columns=[f for f in FAMS if f in set(fam)])
    compn = comp.div(comp.sum(1), axis=0)
    feas = feasible(cent)
    order = np.argsort(cent[:, 0])  # sort clusters by thumb-index distance (pinch first)
    L = [f"\n=== {name}, k={k}  (clusters sorted by thumb-index distance; distances in palm widths)",
         f"{'c':>2s} " + " ".join(f"{n:>5s}" for n in FEAT) + "  | family share (balanced) | interaction rate | "
         "reachable (RMS pw)"]
    for c in order:
        m = lab == c
        inter = D.inter[m]
        ir = {}
        for f in ("Shadow", "F1"):
            x = inter[(fam[m] == f)].dropna()
            if len(x):
                ir[f] = x.astype(float).mean()
        L.append(f"{c:2d} " + " ".join(f"{v:5.2f}" for v in cent[c]) + "  | " +
                 " ".join(f"{f}:{compn.loc[c, f]:.2f}" for f in compn.columns) + " | " +
                 " ".join(f"{f}:{v:.2f}" for f, v in ir.items()) + " | " +
                 " ".join(f"{f}:{feas[f][c]:.3f}{'' if feas[f][c] < 0.05 else '!'}" for f in FAMS))
    # objects: in which cluster do an object's GRASP samples fall? (majority cluster + purity, objects with >= 10)
    g = D[(D.inter == True).to_numpy()].assign(c=lab[(D.inter == True).to_numpy()])  # noqa: E712
    if len(g):
        L.append("objects by majority cluster of their GRASP samples (purity = share in that cluster; >= 10 samples):")
        tab = pd.crosstab(g.object, g.c)
        tab = tab[tab.sum(1) >= 10]
        maj, pur = tab.idxmax(1), tab.max(1) / tab.sum(1)
        for c in order:
            objs = pur[maj == c].sort_values(ascending=False)
            fams_c = g[g.object.isin(objs.index)].groupby("object").family.first()
            L.append(f"  c{c}: {len(objs)} objects, purity median {objs.median() if len(objs) else float('nan'):.2f}: " +
                     ", ".join(f"{o}({fams_c[o][0]}) {v:.2f}" for o, v in objs.head(12).items()))
    # T-Rex: motor primitive / object composition per cluster (labels from the dataset, not used for clustering)
    tr = D.assign(c=lab)[D.dataset.to_numpy() == "trex"]
    if len(tr):
        L.append("T-Rex samples per cluster: over-represented motor primitives / objects (share in cluster / overall):")
        for key in ("primitive", "trex_object"):
            base = tr[key].value_counts(normalize=True)
            for c in order:
                tc = tr[tr.c == c]
                if len(tc) < 20:
                    continue
                vc = tc[key].value_counts()
                enr = (vc / len(tc) / base[vc.index]).where(vc >= 10).dropna().sort_values(ascending=False)
                L.append(f"  c{c} {key} (n={len(tc)}): " + ", ".join(f"{o} x{e:.1f}" for o, e in enr.head(6).items()))
    # EgoSteer: task names per cluster (dataset labels, not used for clustering)
    eg = D.assign(c=lab)[D.dataset.to_numpy() == "egosteer"]
    if len(eg):
        L.append("EgoSteer samples per cluster: over-represented tasks (share in cluster / overall, >= 10 samples):")
        base = eg.ego_task.value_counts(normalize=True)
        for c in order:
            ec = eg[eg.c == c]
            if len(ec) < 20:
                continue
            vc = ec.ego_task.value_counts()
            enr = (vc / len(ec) / base[vc.index]).where(vc >= 10).dropna().sort_values(ascending=False)
            L.append(f"  c{c} (n={len(ec)}): " + ", ".join(f"{o} x{e:.1f}" for o, e in enr.head(6).items()))
    return L, lab, cent, order


def figure(D, X, lab, cent, order, name, fname):
    fams = [f for f in FAMS if f in set(D.family)]
    fig, axes = plt.subplots(len(order), 2 * len(fams), figsize=(4.2 * len(fams), 2.9 * len(order)))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(order):
        for j, f in enumerate(fams):
            m = np.where((lab == c) & (D.family.to_numpy() == f))[0]
            for v in range(2):
                ax = axes[r, 2 * j + v]; ax.set_xticks([]); ax.set_yticks([])
                if v == 0:
                    ax.set_ylabel(f"c{c}", fontsize=10) if j == 0 else None
                if not len(m):
                    ax.text(.5, .5, "no samples", ha="center", transform=ax.transAxes); continue
                i = m[np.argmin(np.linalg.norm(X[m] - cent[c], axis=1))]
                row = D.iloc[i]
                hand = load_hand(row.hand_model_id)
                view, up = (np.array([0, 0, -1.0]), np.array([0, 1.0, 0])) if v == 0 else \
                    (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]))
                draw_mesh(ax, hand_triangles(row.hand_model_id, hand, np.asarray(row.model_q)[None],
                                             list(row.names)), view, up)
                ax.set_aspect("equal"); ax.autoscale_view()
                if r == 0:
                    ax.set_title(f"{f} {'palm view' if v == 0 else 'side view'}", fontsize=9)
                if v == 1:
                    ax.text(0.02, 0.02, f"{row.episode[:28]}\nt={row.t_s:.1f}s", fontsize=6, transform=ax.transAxes)
    fig.suptitle(f"{name}: sample nearest to each cluster centroid, per hand family", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=60); plt.close(fig)


La, lab_a, cent_a, ord_a = describe(ALL, Xa, res_all[K_ALL][0], norm_all, "ALL postures", K_ALL)
Lg, lab_g, cent_g, ord_g = describe(GR, Xg, res_gr[K_GR][0], norm_gr, "GRASP samples", K_GR)
for D, X, lab, cent, order, tag in [(ALL, Xa, lab_a, cent_a, ord_a, f"all_k{K_ALL}"),
                                    (GR, Xg, lab_g, cent_g, ord_g, f"grasp_k{K_GR}")]:
    rank = {c: i for i, c in enumerate(order)}  # display rank = order by thumb-index distance
    D.drop(columns=["tips", "model_q", "names"]).assign(
        cluster=lab, display_rank=[rank[c] for c in lab],
        dist_to_centroid=np.linalg.norm((X - cent[lab]) / norm_all[1] if tag.startswith("all") else
                                        (X - cent[lab]) / norm_gr[1], axis=1)
    ).to_parquet(OUT / f"v2_assignments_{tag}.parquet")
txt = "\n".join(lines + La + Lg)
(OUT / "v2_report.txt").write_text(txt + "\n")
print("\n".join(La + Lg))
figure(ALL, (Xa - norm_all[0]) / norm_all[1] * norm_all[1] + norm_all[0], lab_a, cent_a, ord_a,
       f"ALL postures k={K_ALL}", "v2_all_medoids.png")
figure(GR, Xg, lab_g, cent_g, ord_g, f"GRASP samples k={K_GR}", "v2_grasp_medoids.png")
print("saved")
