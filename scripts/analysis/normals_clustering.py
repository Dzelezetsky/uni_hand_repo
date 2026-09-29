"""Do pad normals help the shared posture vocabulary? Rule C admission test (GRASP_LABELING.md), 2026-09-29.

Feature sets (per frame, one sample per 0.2 s, verified streams with pad normals):
  D      10 pairwise fingertip distances (palm widths) — current vocabulary space
  D+cos  D + 10 pairwise cosines between pad normals (relative pad orientation, frame independent)
  D+th   D + thumb pad normal in the palm frame (3)
  D+N    D + all 5 pad normals in the palm frame (15)
  cos    10 pairwise normal cosines only
Every set is z-scored globally, so each feature has equal weight.

Checks:
  1 mixing ratio per hand (as embodiment_gap.py): ~1 = hands mixed, >>1 = features encode the hand
  2 KMeans k=5 (hand-balanced fit): NMI(cluster, hand) and number of single-hand clusters (one hand > 80 % of the
    hand-balanced mass)
  3 future information (as choose_k.py, D=1 s, event windows): held-out gain over copy-current when the FUTURE class
    is known; x = current D+N (same for all sets), targets: future distances and future normals separately
  4 reachability (reachability.py): share of each hand's real frames the other hands reproduce in D+cos
    (distance RMS < 0.05 pw AND cosine RMS < 0.1)

Output: unified/clustering/normals_rule_c.txt
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, ".")
from unidex.hands import REPO  # noqa: E402
from unidex.reachability import NATIVE_RANGES, best_fit, sample  # noqa: E402

UNI = REPO / "unified"
PAIRS = list(combinations(range(5), 2))
STEP, HOR, K, EVENT_Q, SEED = 0.2, 1.0, 5, 0.8, 0
SHORT = {"shadow_e_right_realdex": "Shadow", "inspire_rh56dfx_right_unitree": "DFX-R",
         "inspire_rh56dfx_left_unitree": "DFX-L", "inspire_rh56f1_right_hrdexdb": "F1"}
rng = np.random.default_rng(SEED)


def dists(P):
    return np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in PAIRS], 1)


def cosines(N):
    return np.stack([(N[:, i] * N[:, j]).sum(1) for i, j in PAIRS], 1)


def feature_sets(P, N):
    d, c = dists(P), cosines(N)
    return {"D": d, "D+cos": np.c_[d, c], "D+th": np.c_[d, N[:, 0]], "D+N": np.c_[d, N.reshape(len(N), 15)],
            "cos": c}


# ---------------------------------------------------------------- load (t, P, N) per stream
hs = pd.read_parquet(UNI / "hand_streams.parquet")
hs = hs[(hs.canonicalization_status == "ok") & (hs.pad_normal_status == "verified") & hs.has_time]
streams = []
for _, r in hs.iterrows():
    f = pd.read_parquet(UNI / r.path)
    f = f[f.valid].sort_values("t_s")
    if len(f) < 10:
        continue
    streams.append((r.hand_model_id, r.mapping_id, r.trajectory_group_id + "|" + r.side, f.t_s.to_numpy(),
                    np.stack(f.fingertips_palm_norm.to_numpy()).reshape(-1, 5, 3).astype(float),
                    np.stack(f.pad_normals_palm.to_numpy()).reshape(-1, 5, 3).astype(float)))


def resample(t, A, tq):
    flat = A.reshape(len(A), -1)
    return np.stack([np.interp(tq, t, flat[:, j]) for j in range(flat.shape[1])], 1).reshape(len(tq), *A.shape[1:])


def unit(N):
    return N / np.linalg.norm(N, axis=-1, keepdims=True)


# frames on a 0.2 s grid (mixing, clustering) and (t, t+1 s) anchor pairs (future information)
P, N, H, E = [], [], [], []
X0, X1, Y0, Y1, AH, AE = [], [], [], [], [], []
for hid, mid, ep, t, Pt, Nt in streams:
    g = np.arange(t[0], t[-1], STEP)
    P.append(resample(t, Pt, g)); N.append(unit(resample(t, Nt, g))); H += [hid] * len(g); E += [ep] * len(g)
    a = np.arange(t[0], t[-1] - HOR, STEP)
    if len(a) >= 3:
        X0.append(resample(t, Pt, a)); X1.append(unit(resample(t, Nt, a)))
        Y0.append(resample(t, Pt, a + HOR)); Y1.append(unit(resample(t, Nt, a + HOR)))
        AH += [hid] * len(a); AE += [ep] * len(a)
P, N, H, E = np.concatenate(P), np.concatenate(N), np.array(H), np.array(E)
X0, X1, Y0, Y1 = (np.concatenate(v) for v in (X0, X1, Y0, Y1))
AH, AE = np.array(AH), np.array(AE)
hands = sorted(np.unique(H))
lines = [f"frames {len(P)} (" + ", ".join(f"{SHORT[h]} {np.sum(H == h)}" for h in hands) + f"), anchors {len(X0)}", ""]


def z(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-9)


# ---------------------------------------------------------------- 1 mixing ratio
def mixing(Y):
    Y = z(Y)
    out = {}
    for h in hands:
        m = H == h
        d_o = cKDTree(Y[~m]).query(Y[m])[0]
        d_s = np.full(m.sum(), np.nan)
        idx = np.where(m)[0]
        for e in np.unique(E[m]):
            me, mo = m & (E == e), m & (E != e)
            if mo.sum():
                d_s[np.searchsorted(idx, np.where(me)[0])] = cKDTree(Y[mo]).query(Y[me])[0]
        out[SHORT[h]] = np.nanmedian(d_o / np.maximum(d_s, 1e-6))
    return out


FS = feature_sets(P, N)
w_frame = 1.0 / pd.Series(H).map(pd.Series(H).value_counts()).to_numpy()
rows = []
for name, F in FS.items():
    mix = mixing(F)
    # ------------------------------------------------------------ 2 clustering composition
    Zf = z(F)
    sub = rng.choice(len(F), 8000, replace=True, p=w_frame / w_frame.sum())
    lab = KMeans(K, n_init=10, random_state=SEED).fit(Zf[sub]).predict(Zf)
    comp = np.zeros((K, len(hands)))
    for i, h in enumerate(hands):
        comp[:, i] = np.bincount(lab[H == h], minlength=K) / np.sum(H == h)  # share of the hand's frames
    comp_n = comp / comp.sum(1, keepdims=True)                              # hand-balanced cluster composition
    single = int((comp_n.max(1) > 0.8).sum())
    nmi = normalized_mutual_info_score(H[sub], lab[sub])
    rows.append(dict(features=name, dim=F.shape[1], mixing_median=np.median(list(mix.values())),
                     **{f"mix_{k}": v for k, v in mix.items()}, NMI_hand=nmi, single_hand_clusters=single))
    lines.append(f"{name:6s} clusters (hand-balanced composition, columns {', '.join(SHORT[h] for h in hands)}):")
    for c in range(K):
        lines.append("   " + " ".join(f"{v:5.2f}" for v in comp_n[c]))
mix_df = pd.DataFrame(rows).set_index("features")

# ---------------------------------------------------------------- 3 future information
w_anc = 1.0 / pd.Series(AH).map(pd.Series(AH).value_counts()).to_numpy()
Xcur = z(feature_sets(X0, X1)["D+N"])
targets = {"future distances": dists(Y0), "future normals": Y1.reshape(len(Y1), 15)}
chg = {k: np.linalg.norm(v - (dists(X0) if k == "future distances" else X1.reshape(len(X1), 15)), axis=1)
       for k, v in targets.items()}
FY = feature_sets(Y0, Y1)
gains = {}
for name in ["none (k=1)"] + list(FS):
    for tname, Y in targets.items():
        cur = dists(X0) if tname == "future distances" else X1.reshape(len(X1), 15)
        event = chg[tname] >= np.quantile(chg[tname], EVENT_Q)
        se = np.zeros(len(Y))
        for tr, te in GroupKFold(5).split(Xcur, Y, AE):
            if name == "none (k=1)":
                ltr, lte = np.zeros(len(tr), int), np.zeros(len(te), int)
            else:
                C = FY[name]
                mu, sd = C[tr].mean(0), C[tr].std(0) + 1e-9
                s = rng.choice(tr, min(6000, len(tr)), replace=True, p=w_anc[tr] / w_anc[tr].sum())
                km = KMeans(K, n_init=5, random_state=SEED).fit((C[s] - mu) / sd)
                ltr, lte = km.predict((C[tr] - mu) / sd), km.predict((C[te] - mu) / sd)
            pred = np.zeros_like(Y[te])
            for c in np.unique(lte):
                mtr, mte = ltr == c, lte == c
                if mtr.sum() < 20:
                    pred[mte] = Y[tr][mtr].mean(0) if mtr.any() else cur[te][mte]
                    continue
                pred[mte] = Ridge(alpha=1e-2).fit(Xcur[tr][mtr], Y[tr][mtr], sample_weight=w_anc[tr][mtr]) \
                    .predict(Xcur[te][mte])
            se[te] = ((Y[te] - pred) ** 2).sum(1)
        se_copy = ((Y - cur) ** 2).sum(1)
        gains[(name, tname)] = 1 - np.average(se[event], weights=w_anc[event]) / \
            np.average(se_copy[event], weights=w_anc[event])
gain_df = pd.Series(gains).unstack()

# ---------------------------------------------------------------- 4 reachability with normals (D+cos)
def dcos(Pb, Nb):
    return np.c_[dists(Pb), cosines(Nb)]


mids = {hid: mid for hid, mid, *_ in streams}
S = {h: sample(mids[h], 100_000) for h in hands if mids[h] in NATIVE_RANGES}
cov = pd.DataFrame(index=[SHORT[h] for h in hands], columns=[SHORT[h] for h in hands], dtype=float)
for b in hands:
    Fb = dcos(S[b]["tips_norm"], S[b]["normals"])
    tree = cKDTree(Fb)
    for a in hands:
        m = H == a
        idx = rng.choice(np.where(m)[0], min(800, m.sum()), replace=False)
        T = dcos(P[idx], N[idx])
        nn = tree.query(T, k=3)[1]
        best = None
        for k in range(3):
            _, r = best_fit(mids[b], dcos, T, S[b]["raw"][nn[:, k]], with_normals=True)
            best = r if best is None else np.where(((r ** 2).sum(1) < (best ** 2).sum(1))[:, None], r, best)
        ok = (np.sqrt((best[:, :10] ** 2).mean(1)) < 0.05) & (np.sqrt((best[:, 10:] ** 2).mean(1)) < 0.1)
        cov.loc[SHORT[a], SHORT[b]] = ok.mean()

pd.set_option("display.width", 200)
txt = "\n".join([
    "Rule C admission test for pad-normal features (hand-balanced, verified hands)", "",
    "1-2  mixing ratio (~1 mixed, >>1 hand-specific), NMI(cluster, hand) and single-hand clusters at k=5:",
    mix_df.round(2).to_string(), "",
    f"3    held-out gain over copy-current, event windows, D={HOR}s, x = current D+N for all rows,",
    "     classes = k=5 clusters of the FUTURE in the given feature set:",
    gain_df.round(3).to_string(), "",
    "4    reachability in D+cos: row = real frames of, column = hand that must reproduce (dist RMS<0.05 pw, cos RMS<0.1)",
    cov.round(2).to_string(), "", *lines])
(UNI / "clustering" / "normals_rule_c.txt").write_text(txt + "\n")
print(txt)
