"""Is future hand geometry a non-trivial prediction target? (verified streams only)

For horizons D and every anchor time t (one anchor per 0.1 s):
  change      mean fingertip displacement |H(t+D) - H(t)| in mm (metric canonical frame)
  persist     relative MSE of "copy current": E|H(t+D)-H(t)|^2 / E|H(t+D)-mean|^2    (~0 = trivial target)
  constvel    same for constant-velocity extrapolation from H(t-0.1 s)                (what own motion explains)
  moving      fraction of anchors whose change exceeds 10 mm
  switch      fraction of anchors whose posture cluster (KMeans k=6 on 10 pairwise distances) differs at t+D
Per dataset (tasks differ strongly). Outputs unified/clustering/future_*.
"""
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UNI, OUT = Path("unified"), Path("unified/clustering")
HOR = [0.25, 0.5, 1.0, 2.0, 3.0]
STEP, VEL_DT, MOVE_MM, K = 0.1, 0.1, 10.0, 6
PAIRS = list(combinations(range(5), 2))

hs = pd.read_parquet(UNI / "hand_streams.parquet")
hs = hs[(hs.canonicalization_status == "ok") & hs.has_time]
streams = []
for _, r in hs.iterrows():
    fr = pd.read_parquet(UNI / r.path)
    fr = fr[fr.valid].sort_values("t_s")
    t = fr.t_s.to_numpy()
    if len(t) < 10:
        continue
    M = np.stack(fr.fingertips_palm_m.to_numpy()).reshape(-1, 5, 3).astype(np.float64)
    N = np.stack(fr.fingertips_palm_norm.to_numpy()).reshape(-1, 5, 3).astype(np.float64)
    streams.append(dict(dataset=r.dataset_id, hand=r.hand_model_id, ep=r.source_episode_id, t=t, M=M, N=N))


def interp(t, X, tq):  # linear interpolation of (T,5,3) at query times (inside range only)
    flat = X.reshape(len(X), -1)
    return np.stack([np.interp(tq, t, flat[:, j]) for j in range(flat.shape[1])], 1).reshape(len(tq), 5, 3)


# posture clusters on all verified data (pairwise distances, palm widths)
allN = np.concatenate([s["N"][:: max(1, len(s["N"]) // 400)] for s in streams])
feat = lambda P: np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in PAIRS], 1)  # noqa: E731
sc = StandardScaler().fit(feat(allN))
km = KMeans(K, n_init=10, random_state=0).fit(sc.transform(feat(allN)))
lab = lambda P: km.predict(sc.transform(feat(P)))  # noqa: E731

rows = []
for s in streams:
    t, M, N = s["t"], s["M"], s["N"]
    for D in HOR:
        a = np.arange(t[0] + VEL_DT, t[-1] - D, STEP)
        if len(a) < 5:
            continue
        cur, fut, prev = interp(t, M, a), interp(t, M, a + D), interp(t, M, a - VEL_DT)
        cv = cur + (cur - prev) * (D / VEL_DT)
        disp = np.linalg.norm(fut - cur, axis=2).mean(1) * 1000
        lc, lf = lab(interp(t, N, a)), lab(interp(t, N, a + D))
        rows.append(dict(dataset=s["dataset"], hand=s["hand"], ep=s["ep"], D=D, n=len(a),
                         se_persist=((fut - cur) ** 2).sum((1, 2)).sum(), se_cv=((fut - cv) ** 2).sum((1, 2)).sum(),
                         fut_sum=fut.reshape(len(a), -1).sum(0), fut_sq=(fut ** 2).reshape(len(a), -1).sum(0),
                         disp_med=np.median(disp), disp_p90=np.percentile(disp, 90),
                         moving=np.mean(disp > MOVE_MM), switch=np.mean(lc != lf), disp_all=disp))
R = pd.DataFrame(rows)


def summarize(g):
    n = g.n.sum()
    mean = np.stack(g.fut_sum).sum(0) / n
    var = (np.stack(g.fut_sq).sum(0) / n - mean ** 2).sum()  # total variance of future geometry
    disp = np.concatenate(g.disp_all.to_list())
    return pd.Series(dict(episodes=g.ep.nunique(), anchors=n,
                          change_med_mm=np.median(disp), change_p90_mm=np.percentile(disp, 90),
                          moving_frac=np.mean(disp > MOVE_MM),
                          persist_relMSE=g.se_persist.sum() / n / var, constvel_relMSE=g.se_cv.sum() / n / var,
                          cluster_switch=np.average(g.switch, weights=g.n)))


S = R.groupby(["dataset", "D"]).apply(summarize).reset_index()
pd.set_option("display.width", 200)
print(S.round(3).to_string(index=False))
S.to_csv(OUT / "future_predictability.csv", index=False)

fig, ax = plt.subplots(1, 4, figsize=(22, 5))
for ds, g in S.groupby("dataset"):
    ax[0].plot(g.D, g.change_med_mm, "o-", label=ds)
    ax[1].plot(g.D, g.persist_relMSE, "o-", label=ds)
    ax[1].plot(g.D, g.constvel_relMSE, "x--", color=ax[1].lines[-1].get_color(), alpha=.6)
    ax[2].plot(g.D, g.moving_frac, "o-", label=ds)
    ax[3].plot(g.D, g.cluster_switch, "o-", label=ds)
ax[0].set_title("median fingertip change over horizon (mm)")
ax[1].set_title("relative MSE: copy-current (solid) / const-velocity (dashed)\n1.0 = no better than predicting the mean")
ax[1].axhline(1, c="grey", ls=":")
ax[2].set_title(f"fraction of anchors with change > {MOVE_MM:.0f} mm")
ax[3].set_title(f"fraction of anchors whose posture cluster changes (k={K})")
for a in ax:
    a.set_xlabel("horizon D (s)"); a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "future_predictability.png", dpi=65)
print("saved")
