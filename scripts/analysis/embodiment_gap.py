"""How embodiment-specific is each candidate hand feature? (thumb gap study, 2026-09-23)

Mixing ratio per hand = median[ d(nearest sample of ANOTHER hand) / d(nearest sample of the SAME hand, other episode) ],
one sample per 0.5 s. ~1 = hands indistinguishable in that feature space, >>1 = features encode the embodiment.
Features are z-scored globally before distances so that feature sets are comparable.
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from unidex.hands import load_hand  # noqa: E402

hs = pd.read_parquet("unified/hand_streams.parquet")
hs = hs[hs.canonicalization_status == "ok"]
X, H, E = [], [], []
for _, r in hs.iterrows():
    fr = pd.read_parquet("unified/" + r.path)
    keep = np.r_[True, np.diff(np.floor(fr.t_s.to_numpy() / 0.5)) > 0]
    n = np.stack(fr.fingertips_palm_norm.to_numpy())[keep]
    ok = np.isfinite(n).all(1)
    X.append(n[ok].reshape(-1, 5, 3)); H += [r.hand_model_id] * ok.sum(); E += [r.trajectory_group_id + r.side] * ok.sum()
P = np.concatenate(X); H = np.array(H); E = np.array(E)

# kinematic reach of every fingertip coordinate per hand, from the URDF alone (data independent)
rng = np.random.default_rng(0)
reach = {}
for hid in np.unique(H):
    h = load_hand(hid); fk = h.fk; nm = fk.actuated_joints
    lo = np.array([fk.joints[j].lower if fk.joints[j].lower is not None else -1 for j in nm])
    hi = np.array([fk.joints[j].upper if fk.joints[j].upper is not None else 1 for j in nm])
    s = h.canonical(lo + rng.random((20000, len(nm))) * (hi - lo), nm)[1]
    reach[hid] = (np.percentile(s, 2, 0), np.percentile(s, 98, 0))


def pairwise(p):
    return np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in combinations(range(5), 2)], 1)


def reach_norm(p, hid):
    lo, hi = reach[hid]
    return (p - lo) / np.maximum(hi - lo, 1e-3)


feats = {
    "A raw 15D fingertips (current)": P.reshape(len(P), -1),
    "B 4 fingers only (12D, drop thumb)": P[:, 1:].reshape(len(P), -1),
    "C 10 pairwise tip distances": pairwise(P),
    "D 4 fingers + 4 thumb->finger dists": np.c_[P[:, 1:].reshape(len(P), -1), pairwise(P)[:, :4]],
    "E per-hand URDF-reach normalized 15D": np.concatenate([reach_norm(P[H == h], h).reshape(-1, 15) for h in np.unique(H)])[
        np.argsort(np.argsort(np.concatenate([np.where(H == h)[0] for h in np.unique(H)])))],
    "F fingers 12D + thumb reach-normalized": None,
}
thumb_rn = np.zeros((len(P), 3))
for h in np.unique(H):
    thumb_rn[H == h] = reach_norm(P[H == h], h)[:, 0]
feats["F fingers 12D + thumb reach-normalized"] = np.c_[P[:, 1:].reshape(len(P), -1), thumb_rn]


def mixing(Y):
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    out = {}
    for h in np.unique(H):
        m = H == h
        d_o, _ = cKDTree(Y[~m]).query(Y[m])
        d_s = np.full(m.sum(), np.nan)
        idx = np.where(m)[0]
        for e in np.unique(E[m]):
            me, mo = m & (E == e), m & (E != e)
            if mo.sum():
                d_s[np.searchsorted(idx, np.where(me)[0])] = cKDTree(Y[mo]).query(Y[me])[0]
        out[h.replace("_right", "").replace("_hrdexdb", "").replace("_unitree", "").replace("_realdex", "")] = \
            np.nanmedian(d_o / np.maximum(d_s, 1e-6))
    return out


rows = {k: mixing(v) for k, v in feats.items()}
df = pd.DataFrame(rows).T
df["median"] = df.median(axis=1)
pd.set_option("display.width", 250)
print(f"samples={len(P)}  (mixing ratio: ~1 mixed, >>1 embodiment-specific)")
print(df.round(2).to_string())

# --- G: relational + per-finger extension ratio (|tip - finger base| / same at the URDF rest pose), data independent
ext = np.zeros((len(P), 5))
for hid in np.unique(H):
    h = load_hand(hid); lm = h.frame.landmarks; nm = h.fk.actuated_joints
    bases = [h.fk.joint_origin_in_root(h.fk.chain(h.tip_links[0])[0].name)] + \
            [lm[f"base_{f}"] for f in ("index", "middle", "ring", "pinky")]
    from unidex.kinematics.canonical import to_canonical
    b = to_canonical(np.stack(bases), h.frame, "right") / h.frame.scale
    rest = h.canonical(np.zeros((1, len(nm))), nm)[1][0]
    L0 = np.linalg.norm(rest - b, axis=1)
    m = H == hid
    ext[m] = np.linalg.norm(P[m] - b[None], axis=2) / L0
G = np.c_[pairwise(P), ext]
rows["G pairwise dists + 5 extension ratios"] = mixing(G)
rows["H pairwise dists + 4 finger ext (no thumb ext)"] = mixing(np.c_[pairwise(P), ext[:, 1:]])
df = pd.DataFrame(rows).T
df["median"] = df.median(axis=1)
print("\n" + df.round(2).to_string())
