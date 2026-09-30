"""Is the future hand posture multimodal WITHIN a posture class, given the current posture? (STAGE1_TRAINING.md,
open check: decides head option 1 "class + residual" vs option 2 generative / flow matching.)

Space: the 10 pairwise fingertip distances, z-scored with the frozen hand_posture_class_v1 normalization.
Anchors every STEP s on verified timed streams (<= 400 streams per hand family, as choose_k.py):
  x = posture(t), y = posture(t + D), c = hand_posture_class_v1(y).
Neighbourhood test (per hand family, horizon D, condition):
  A "x"      : N nearest anchors in x (other episodes, one per episode) -> conditional p(y | x)
  B "x + c"  : same, restricted to the same FUTURE class c           -> p(y | x, c) = what option 1's residual faces
  2-means on the N future postures; Ashman separation Dsep = |m1-m2| / sqrt((s1^2+s2^2)/2) along the centre line,
  minor mode share >= MIN_SHARE. Null: N draws from a Gaussian fitted to the same neighbourhood (Ledoit-Wolf),
  NULL_REPS per neighbourhood, pooled -> 95th percentile threshold per (family, D, condition).
Effect sizes (palm widths, pw; mm via each family's palm scale): mode separation, and the MEAN-REGRESSION error =
distance from the neighbourhood mean (what an L2 head predicts) to the nearest mode, vs the within-mode spread.
Caveat: conditions on hand proprioception only; video + language can only reduce the multimodality -> upper bound.
Output: unified/clustering/within_class_multimodality.txt
usage: within_class_multimodality.py [--selftest | --resid]   (--resid: D = 1 s, futures residualized on the
  top-3 principal directions of x inside each neighbourhood — control for x variation within the radius)
"""
from __future__ import annotations

import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.covariance import LedoitWolf

sys.path.insert(0, ".")
from unidex.hands import REPO, load_hand  # noqa: E402

UNI, OUT = REPO / "unified", REPO / "unified" / "clustering"
HOR, STEP, SEED, MAX_STREAMS = [0.5, 1.0, 2.0], 0.2, 0, 400
RESID = "--resid" in sys.argv
if RESID:
    HOR = [1.0]
N_NB, K_CAND, R_MAX, N_QUERY, NULL_REPS, MIN_SHARE, EVENT_Q = 50, 600, 1.0, 500, 5, 0.2, 0.8
PAIRS = list(combinations(range(5), 2))
MODEL = json.loads((REPO / "config/hand_posture_class_v1.json").read_text())
MU, SD = np.array(MODEL["mu"]), np.array(MODEL["sd"])
CZ = (np.array(MODEL["centroids"]) - MU) / SD
rng = np.random.default_rng(SEED)


def two_means(Y, iters=25, restarts=4):
    best = None
    for _ in range(restarts):
        i = rng.choice(len(Y))
        j = np.argmax(((Y - Y[i]) ** 2).sum(1) * rng.random(len(Y)) ** 0.2)  # far point, randomized
        c = Y[[i, j]].astype(float)
        for _ in range(iters):
            lab = ((Y[:, None] - c[None]) ** 2).sum(2).argmin(1)
            if lab.min() == lab.max():
                break
            c = np.stack([Y[lab == k].mean(0) for k in (0, 1)])
        sse = ((Y - c[lab]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, lab, c)
    return best[1], best[2]


def separation(Y):
    """-> (Ashman D of the best 2-split along the neighbourhood's first principal axis, minor share, 10-D centres of
    the two groups, labels). The split is searched in 1-D (PC1) — 2-means in 10-D on ~50 points overfits (selftest)."""
    Yc = Y - Y.mean(0)
    u = np.linalg.svd(Yc, full_matrices=False)[2][0]
    p = Yc @ u
    o = np.sort(p)
    best, cut = -1, None
    for i in range(int(MIN_SHARE * len(p)), int((1 - MIN_SHARE) * len(p)) + 1):  # exact 1-D 2-means over split points
        a, b = o[:i], o[i:]
        score = len(a) * len(b) * (b.mean() - a.mean()) ** 2
        if score > best:
            best, cut = score, (o[i - 1] + o[i]) / 2
    lab = (p > cut).astype(int)
    c = np.stack([Y[lab == k].mean(0) for k in (0, 1)])
    p0, p1 = p[lab == 0], p[lab == 1]
    s = np.sqrt((p0.var() + p1.var()) / 2) + 1e-9
    return abs(p1.mean() - p0.mean()) / s, min(lab.mean(), 1 - lab.mean()), c, lab


def null_seps(Y):
    lw = LedoitWolf().fit(Y)
    return [separation(rng.multivariate_normal(lw.location_, lw.covariance_, len(Y)))[0] for _ in range(NULL_REPS)]


def selftest():
    d, n = 10, N_NB
    g = [separation(rng.standard_normal((n, d)))[0] for _ in range(300)]
    thr = np.quantile(sum((null_seps(rng.standard_normal((n, d))) for _ in range(60)), []), 0.95)
    for delta in (0, 2, 3, 4):
        det = 0
        for _ in range(200):
            Y = rng.standard_normal((n, d))
            Y[: n // 2, 0] += delta
            s, share, *_ = separation(Y)
            det += s > thr and share >= MIN_SHARE
        print(f"planted 50/50 bimodal, mode gap {delta} sd: detected {det / 200:.0%}")
    print(f"null threshold {thr:.2f}; Gaussian Dsep median {np.median(g):.2f}")


def load():
    hs = pd.read_parquet(UNI / "hand_streams.parquet")
    hs = hs[(hs.canonicalization_status == "ok") & hs.has_time]
    hs["family"] = hs.hand_model_id.map(MODEL["family_of_hand_model"])
    hs = pd.concat([g.sample(min(len(g), MAX_STREAMS), random_state=SEED) for _, g in hs.groupby("family")])
    scale = {f: np.mean([load_hand(h).frame.scale for h in g.hand_model_id.unique()]) for f, g in hs.groupby("family")}
    streams = []
    for _, r in hs.iterrows():
        fr = pd.read_parquet(UNI / r.path, columns=["t_s", "valid", "fingertips_palm_norm"])
        fr = fr[fr.valid].sort_values("t_s")
        if len(fr) < 10:
            continue
        P = np.stack(fr.fingertips_palm_norm.to_numpy()).reshape(-1, 5, 3).astype(float)
        F = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in PAIRS], 1)
        streams.append((r.dataset_id, r.family, r.trajectory_group_id + "|" + r.side, fr.t_s.to_numpy(), (F - MU) / SD))
    return streams, scale


def anchors(streams, D):
    X, Y, G, FAM, DS = [], [], [], [], []
    for ds, fam, ep, t, Z in streams:
        a = np.arange(t[0], t[-1] - D, STEP)
        if len(a) < 3:
            continue
        X.append(np.stack([np.interp(a, t, Z[:, j]) for j in range(10)], 1))
        Y.append(np.stack([np.interp(a + D, t, Z[:, j]) for j in range(10)], 1))
        G += [ep] * len(a); FAM += [fam] * len(a); DS += [ds] * len(a)
    X, Y = np.concatenate(X), np.concatenate(Y)
    G, FAM, DS = np.array(G), np.array(FAM), np.array(DS)
    C = ((Y[:, None] - CZ[None]) ** 2).sum(2).argmin(1)
    chg = np.linalg.norm((Y - X) * SD, axis=1)
    ev = np.zeros(len(X), bool)
    for ds in np.unique(DS):
        m = DS == ds
        ev[m] = chg[m] >= np.quantile(chg[m], EVENT_Q)
    return X, Y, G, FAM, C, ev


def neighbourhood(q, idx, tree, X, G):
    d, nn = tree.query(X[q], k=min(K_CAND, len(idx)))
    nn, d = idx[np.atleast_1d(nn)], np.atleast_1d(d)
    keep = (G[nn] != G[q])
    nn, d = nn[keep], d[keep]
    _, first = np.unique(G[nn], return_index=True)
    first = np.sort(first)[:N_NB]
    if len(first) < N_NB or d[first[-1]] > R_MAX:
        return None
    return nn[first]


def main():
    streams, scale = load()
    names = [c["name"] for c in MODEL["classes"]]
    cz_pw = np.array(MODEL["centroids"])
    between = np.array([min(np.linalg.norm(cz_pw[a] - cz_pw[b]) for b in range(len(cz_pw)) if b != a)
                        for a in range(len(cz_pw))])
    L = [__doc__.split("Output:")[0], f"N_NB={N_NB} R_MAX={R_MAX} z  N_QUERY={N_QUERY}/family (event anchors)  "
         f"null reps {NULL_REPS}  min share {MIN_SHARE}",
         f"nearest-class-centroid distance per class (pw): " +
         ", ".join(f"{n} {b:.2f}" for n, b in zip(names, between))]
    rows = []
    for D in HOR:
        X, Y, G, FAM, C, ev = anchors(streams, D)
        for fam in sorted(np.unique(FAM)):
            fm = FAM == fam
            qs = rng.choice(np.where(fm & ev)[0], min(N_QUERY, (fm & ev).sum()), replace=False)
            trees = {"A": {None: (np.where(fm)[0],)}, "B": {c: (np.where(fm & (C == c))[0],) for c in range(len(CZ))}}
            for cond, groups in trees.items():
                for key, (idx,) in groups.items():
                    groups[key] = (idx, cKDTree(X[idx]) if len(idx) >= N_NB else None)
                res, nulls = [], []
                for q in qs:
                    idx, tree = groups[None if cond == "A" else C[q]]
                    if tree is None:
                        continue
                    nb = neighbourhood(q, idx, tree, X, G)
                    if nb is None:
                        continue
                    Yn = Y[nb]
                    if RESID:  # remove the part of the future explained linearly by the remaining variation of x
                        Xc = X[nb] - X[nb].mean(0)
                        V = np.linalg.svd(Xc, full_matrices=False)[2][:3]
                        A = np.c_[Xc @ V.T, np.ones(len(nb))]
                        Yn = Yn - A[:, :3] @ np.linalg.lstsq(A, Yn, rcond=None)[0][:3]
                    s, share, cen, lab = separation(Yn)
                    nulls += null_seps(Yn)
                    Ypw, cpw = Yn * SD, cen * SD
                    mean = Ypw.mean(0)
                    spread = np.sqrt(np.mean([((Ypw[lab == k] - cpw[k]) ** 2).sum(1).mean() for k in (0, 1)
                                              if (lab == k).any()]))
                    res.append(dict(q=q, c=C[q], sep=s, share=share, gap_pw=np.linalg.norm(cpw[1] - cpw[0]),
                                    mean_err_pw=np.linalg.norm(cpw - mean, axis=1).min(), spread_pw=spread,
                                    total_pw=np.sqrt(((Ypw - mean) ** 2).sum(1).mean())))
                if not res:
                    continue
                R = pd.DataFrame(res)
                thr = np.quantile(nulls, 0.95)
                R["bimodal"] = (R.sep > thr) & (R.share >= MIN_SHARE)
                b = R[R.bimodal]
                mm = scale[fam] * 1000
                rows.append(dict(D=D, family=fam, cond=cond, n=len(R), local=len(R) / len(qs), thr=thr,
                                 frac_bimodal=R.bimodal.mean(),
                                 gap_mm=b.gap_pw.median() * mm if len(b) else np.nan,
                                 mean_err_mm=b.mean_err_pw.median() * mm if len(b) else np.nan,
                                 spread_mm=b.spread_pw.median() * mm if len(b) else np.nan,
                                 total_mm=R.total_pw.median() * mm,
                                 gap_vs_class=(b.gap_pw / between[b.c]).median() if len(b) else np.nan))
                if cond == "B" and D == 1.0:
                    per = R.groupby("c").bimodal.agg(["mean", "size"])
                    L.append(f"  D=1 {fam:7s} B per future class (frac bimodal / n): " +
                             ", ".join(f"{names[c][:12]} {m:.0%}/{s}" for c, (m, s) in per.iterrows() if s >= 15))
            print(f"D={D} done", flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    L.append("\nA = p(y|x), B = p(y|x, future class). local = share of queries with a neighbourhood within R_MAX.\n"
             "frac_bimodal = significant 2-mode split (vs Gaussian null) with minor mode >= 20 %.\n"
             "For bimodal neighbourhoods (medians): gap = distance between modes, mean_err = L2-head error "
             "(mean -> nearest mode), spread = within-mode RMS; total = RMS spread of all futures; "
             "gap_vs_class = gap / distance to the nearest other class centroid.")
    L.append(T.round(3).to_string(index=False))
    for cond in "AB":
        s = T[T.cond == cond].groupby("D")[["frac_bimodal", "gap_mm", "mean_err_mm", "spread_mm", "gap_vs_class"]]
        L.append(f"\nfamily-mean, condition {cond}:\n" + s.mean().round(3).to_string())
    txt = "\n".join(L)
    (OUT / f"within_class_multimodality{'_resid' if RESID else ''}.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
