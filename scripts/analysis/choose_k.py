"""How many posture classes? Choose k by how much information the FUTURE class label adds about the future hand
posture beyond what the current posture already gives (verified streams, timestamps required).

Space: 10 pairwise fingertip distances (palm widths) — embodiment-invariant (GRASP_LABELING.md).
For horizon D, anchors every STEP s:  x = posture(t), y = posture(t + D), c = class(y) from KMeans(k) on train y.
Predictor given (x, c): per-class ridge  y ≈ A_c x + b_c  (k = 1 == "present only").
  gain(k) = 1 - MSE_k / MSE_copy      (MSE_copy = |y - x|^2, the copy-current baseline)
on held-out episodes (GroupKFold by episode), for all windows and for event windows (|y - x| large).
Cost side: label entropy (bits), smallest class share, cluster switch rate c(t) != c(t+D).
Outputs unified/clustering/choose_k.*
"""
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UNI, OUT = Path("unified"), Path("unified/clustering")
HOR, KS, STEP, FOLDS, SEED = [0.5, 1.0, 2.0], [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24], 0.2, 5, 0
EVENT_Q = 0.8  # event window = top 20% of |y - x| within its dataset (dataset-relative, rates differ a lot)
PAIRS = list(combinations(range(5), 2))
rng = np.random.default_rng(SEED)

hs = pd.read_parquet(UNI / "hand_streams.parquet")
hs = hs[(hs.canonicalization_status == "ok") & hs.has_time]


def feats(P):
    return np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in PAIRS], 1)


def interp(t, F, tq):
    return np.stack([np.interp(tq, t, F[:, j]) for j in range(F.shape[1])], 1)


streams = []
for _, r in hs.iterrows():
    fr = pd.read_parquet(UNI / r.path)
    fr = fr[fr.valid].sort_values("t_s")
    if len(fr) < 10:
        continue
    P = np.stack(fr.fingertips_palm_norm.to_numpy()).reshape(-1, 5, 3).astype(float)
    streams.append((r.dataset_id, r.hand_model_id, r.trajectory_group_id + "|" + r.side, fr.t_s.to_numpy(), feats(P)))

rows, curves = [], []
for D in HOR:
    X, Y, G, DS, HD = [], [], [], [], []
    for ds, hand, ep, t, F in streams:
        a = np.arange(t[0], t[-1] - D, STEP)
        if len(a) < 3:
            continue
        X.append(interp(t, F, a)); Y.append(interp(t, F, a + D))
        G += [ep] * len(a); DS += [ds] * len(a); HD += [hand] * len(a)
    X, Y, G, DS, HD = np.concatenate(X), np.concatenate(Y), np.array(G), np.array(DS), np.array(HD)
    chg = np.linalg.norm(Y - X, axis=1)
    event = np.zeros(len(X), bool)
    for ds in np.unique(DS):
        m = DS == ds
        event[m] = chg[m] >= np.quantile(chg[m], EVENT_Q)
    # balance hands in the loss/metrics: weight = 1 / (#samples of that hand)
    w = 1.0 / pd.Series(HD).map(pd.Series(HD).value_counts()).to_numpy()
    for k in KS:
        se = np.zeros(len(X)); lab_all = np.zeros(len(X), int); switch = np.zeros(len(X), bool)
        for tr, te in GroupKFold(FOLDS).split(X, Y, G):
            if k == 1:
                lab_tr, lab_te, lab_x = np.zeros(len(tr), int), np.zeros(len(te), int), np.zeros(len(te), int)
            else:
                # fit clusters on hand-balanced resample of TRAIN future postures
                p = w[tr] / w[tr].sum()
                sub = rng.choice(tr, size=min(len(tr), 6000), replace=True, p=p)
                km = KMeans(k, n_init=5, random_state=SEED).fit(Y[sub])
                lab_tr, lab_te, lab_x = km.predict(Y[tr]), km.predict(Y[te]), km.predict(X[te])
            pred = np.zeros_like(Y[te])
            for c in range(k):
                mtr, mte = lab_tr == c, lab_te == c
                if not mte.any():
                    continue
                if mtr.sum() < 20:  # too few train samples in class: fall back to class mean
                    pred[mte] = Y[tr][mtr].mean(0) if mtr.any() else X[te][mte]
                    continue
                reg = Ridge(alpha=1e-2).fit(X[tr][mtr], Y[tr][mtr], sample_weight=w[tr][mtr])
                pred[mte] = reg.predict(X[te][mte])
            se[te] = ((Y[te] - pred) ** 2).sum(1)
            lab_all[te] = lab_te; switch[te] = lab_te != lab_x
        se_copy = ((Y - X) ** 2).sum(1)
        for scope, m in [("all", np.ones(len(X), bool)), ("event", event)]:
            for ds in ["ALL"] + sorted(np.unique(DS)):
                mm = m & ((DS == ds) if ds != "ALL" else True)
                ww = w[mm] if ds == "ALL" else np.ones(mm.sum())
                gain = 1 - np.average(se[mm], weights=ww) / np.average(se_copy[mm], weights=ww)
                rows.append(dict(D=D, k=k, scope=scope, dataset=ds, gain=gain))
        pk = np.bincount(lab_all, weights=w, minlength=k); pk = pk / pk.sum()
        curves.append(dict(D=D, k=k, entropy_bits=float(-(pk[pk > 0] * np.log2(pk[pk > 0])).sum()),
                           min_class_share=float(pk.min()), switch=float(np.average(switch, weights=w))))
    print(f"D={D}: anchors {len(X)}, episodes {len(np.unique(G))}", flush=True)

R = pd.DataFrame(rows); C = pd.DataFrame(curves)
R.to_csv(OUT / "choose_k_gain.csv", index=False); C.to_csv(OUT / "choose_k_cost.csv", index=False)
piv = R[R.dataset == "ALL"].pivot_table(index="k", columns=["scope", "D"], values="gain").round(3)
print("\nheld-out gain over copy-current, hand-balanced (rows k; k=1 = present posture only):\n", piv.to_string())
print("\nper-dataset gain, event windows, D=1s:\n",
      R[(R.scope == "event") & (R.D == 1.0)].pivot_table(index="k", columns="dataset", values="gain").round(3).to_string())
print("\nlabel cost:\n", C.pivot_table(index="k", columns="D", values=["entropy_bits", "min_class_share", "switch"]).round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(19, 5))
for i, scope in enumerate(["all", "event"]):
    for D in HOR:
        g = R[(R.dataset == "ALL") & (R.scope == scope) & (R.D == D)]
        ax[i].plot(g.k, g.gain, "o-", label=f"D={D}s")
    ax[i].set_xscale("log", base=2); ax[i].set_xlabel("k (log2)"); ax[i].grid(alpha=.3); ax[i].legend()
    ax[i].set_title(f"held-out gain over copy-current, {scope} windows")
for D in HOR:
    c = C[C.D == D]
    ax[2].plot(c.k, c.entropy_bits, "o-", label=f"entropy D={D}s")
ax[2].plot(C[C.D == 1.0].k, np.log2(C[C.D == 1.0].k.clip(lower=1)), "k:", label="log2 k (uniform)")
ax[2].set_xscale("log", base=2); ax[2].set_title("label entropy (bits)"); ax[2].legend(); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "choose_k.png", dpi=65)
print("saved")
