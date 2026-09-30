"""Oracle experiment, stage A (no images): build time-based training windows per dataset.

Anchor every STEP s (time grid, never frame index), for each (dataset, episode):
  inputs   arm(t), arm(t-0.25), arm(t-0.5)              (all arm streams concatenated, measured joints)
           hand native_q(t) per side (+ active mask), current hand_posture_class_v1 per side (8 = none)
           text: instruction (TF-IDF + SVD, fitted per dataset)
  oracle   future posture class per side at ORACLE_H, future fingertips_palm_norm (15D) per side at ORACLE_H
  targets  arm(t+h) - arm(t), hand native_q(t+h) - native_q(t) for h in TARGET_H
  meta     episode group, hand_change = max over sides of |tips(t+2) - tips(t)| (for event windows)
Classes: nearest stored sample in time (labels are not interpolated). Sides whose hand stream is not verified /
not used get mask 0, zeros and class 8.
Output: unified/oracle/<dataset>.npz (git-ignored).
usage: python scripts/oracle/build_windows.py [dataset ...]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from unidex.hands import REPO  # noqa: E402

UNI, OUT = REPO / "unified", REPO / "unified" / "oracle"
STEP, HIST = 0.2, (0.25, 0.5)
TARGET_H = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
ORACLE_H = (0.5, 1.0, 1.5, 2.0)
NONE = 8
ARMS = {"egosteer": ["left_arm", "right_arm"], "humanoid_everyday_h1": ["arm_joint"],
        "dexora": ["left_arm", "right_arm"], "trex": ["arm_joint_left", "arm_joint_right"]}
MAX_EP = {"egosteer": 12000}
SEED = 0


def interp(t, F, tq):
    return np.stack([np.interp(tq, t, F[:, j]) for j in range(F.shape[1])], 1)


def nearest(t, v, tq):
    i = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
    i -= (tq - t[i - 1]) < (t[i] - tq)
    return v[i]


def one_episode(args):
    ds, ep_dir, hands, labels = args
    try:
        arms = []
        for a in ARMS[ds]:
            s = pd.read_parquet(ep_dir / f"stream_{a}.parquet")
            arms.append((s.t_s.to_numpy(), np.stack(s.data.to_numpy()).astype(np.float32)))
        t_arm = arms[0][0]
        arm = np.concatenate([interp(t, F, t_arm) if len(t) != len(t_arm) else F for t, F in arms], 1)
        t0, t1 = t_arm[0] + max(HIST), t_arm[-1] - max(TARGET_H)
        side_data = {}
        for side in ("left", "right"):
            h = hands.get(side)
            if h is None:
                continue
            fr = pd.read_parquet(UNI / h["path"], columns=["t_s", "native_q", "fingertips_palm_norm", "valid"])
            fr = fr[fr.valid]
            if len(fr) < 10:
                continue
            lab = labels.get(side)
            side_data[side] = (fr.t_s.to_numpy(), np.stack(fr.native_q.to_numpy()).astype(np.float32),
                               np.stack(fr.fingertips_palm_norm.to_numpy()).astype(np.float32), lab)
            t0, t1 = max(t0, fr.t_s.iloc[0] + max(HIST)), min(t1, fr.t_s.iloc[-1] - max(TARGET_H))
        if t1 <= t0 or not side_data:
            return None
        a = np.arange(t0, t1, STEP)
        out = {"arm_hist": np.concatenate([interp(t_arm, arm, a - d) for d in (0.0,) + HIST], 1),
               "arm_tgt": np.stack([interp(t_arm, arm, a + h) - interp(t_arm, arm, a) for h in TARGET_H], 1)}
        change = np.zeros(len(a))
        for side in ("left", "right"):
            nq_dim = None
            if side in side_data:
                t, q, tips, lab = side_data[side]
                nq_dim = q.shape[1]
                out[f"{side}_mask"] = np.ones(len(a), np.float32)
                out[f"{side}_q"] = interp(t, q, a)
                out[f"{side}_q_tgt"] = np.stack([interp(t, q, a + h) - out[f"{side}_q"] for h in TARGET_H], 1)
                if lab is not None:
                    lt, lc = lab
                    out[f"{side}_cls"] = nearest(lt, lc, a).astype(np.int8)
                    out[f"{side}_cls_fut"] = np.stack([nearest(lt, lc, a + h) for h in ORACLE_H], 1).astype(np.int8)
                else:
                    out[f"{side}_cls"] = np.full(len(a), NONE, np.int8)
                    out[f"{side}_cls_fut"] = np.full((len(a), len(ORACLE_H)), NONE, np.int8)
                tipf = tips.reshape(len(tips), 15)
                out[f"{side}_geo_fut"] = np.stack([interp(t, tipf, a + h) for h in ORACLE_H], 1)
                change = np.maximum(change, np.linalg.norm(interp(t, tipf, a + 2.0) - interp(t, tipf, a), axis=1))
            out[f"{side}_qdim"] = nq_dim
        out["change"] = change
        return out
    except FileNotFoundError:
        return None


def build(ds):
    import pyarrow.parquet as pq
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    hs = pd.read_parquet(UNI / "hand_streams.parquet")
    hs = hs[(hs.dataset_id == ds) & (hs.canonicalization_status == "ok")]
    man = pd.read_parquet(UNI / "manifest.parquet")
    man = man[man.dataset_id == ds].set_index("source_episode_id")
    eps = sorted(set(hs.source_episode_id))
    if ds in MAX_EP and len(eps) > MAX_EP[ds]:
        eps = sorted(np.random.default_rng(SEED).choice(eps, MAX_EP[ds], replace=False))
    lab = pq.read_table(UNI / "derived/hand_posture_class_v1" / f"{ds}.parquet",
                        columns=["source_episode_id", "side", "t_s", "posture_class"]).to_pandas()
    lab = lab[lab.source_episode_id.isin(set(eps)) & (lab.posture_class >= 0)]
    L = {k: (g.t_s.to_numpy(), g.posture_class.to_numpy()) for k, g in lab.groupby(["source_episode_id", "side"],
                                                                                    observed=True)}
    H = {k: r for k, r in hs.set_index(["source_episode_id", "side"])[["path"]].iterrows()}
    jobs = []
    for e in eps:
        hands = {s: {"path": H[(e, s)]["path"]} for s in ("left", "right") if (e, s) in H}
        labs = {s: L[(e, s)] for s in ("left", "right") if (e, s) in L}
        jobs.append((ds, UNI / man.loc[e, "episode_dir"], hands, labs))
    res = []
    with ProcessPoolExecutor(12) as ex:
        for e, r in zip(eps, ex.map(one_episode, jobs, chunksize=32)):
            if r is not None:
                res.append((e, r))
    qdim = {s: next((r[f"{s}_qdim"] for _, r in res if r[f"{s}_qdim"]), 0) for s in ("left", "right")}
    n = [len(r["arm_hist"]) for _, r in res]
    D = {"group": np.repeat(np.arange(len(res)), n), "episode": np.array([e for e, _ in res]),
         "arm_hist": np.concatenate([r["arm_hist"] for _, r in res]),
         "arm_tgt": np.concatenate([r["arm_tgt"] for _, r in res]),
         "change": np.concatenate([r["change"] for _, r in res])}
    for s in ("left", "right"):
        dq = qdim[s]
        def get(r, key, shape, fill=0):
            m = len(r["arm_hist"])
            return r[key] if key in r else np.full((m,) + shape, fill, dtype=np.float32 if fill == 0 else np.int8)
        D[f"{s}_mask"] = np.concatenate([get(r, f"{s}_mask", ()) for _, r in res])
        D[f"{s}_q"] = np.concatenate([get(r, f"{s}_q", (dq,)) for _, r in res])
        D[f"{s}_q_tgt"] = np.concatenate([get(r, f"{s}_q_tgt", (len(TARGET_H), dq)) for _, r in res])
        D[f"{s}_cls"] = np.concatenate([get(r, f"{s}_cls", (), NONE) for _, r in res])
        D[f"{s}_cls_fut"] = np.concatenate([get(r, f"{s}_cls_fut", (len(ORACLE_H),), NONE) for _, r in res])
        D[f"{s}_geo_fut"] = np.concatenate([get(r, f"{s}_geo_fut", (len(ORACLE_H), 15)) for _, r in res])
    text = [str(man.loc[e, "instruction_en"] or "") for e, _ in res]
    tf = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform(text)
    emb = TruncatedSVD(min(64, tf.shape[1] - 1), random_state=SEED).fit_transform(tf).astype(np.float32)
    D["text"] = emb[D["group"]]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ds}.npz", target_h=np.array(TARGET_H), oracle_h=np.array(ORACLE_H), **D)
    print(ds, "episodes", len(res), "anchors", len(D["group"]), "arm dim", D["arm_hist"].shape[1] // 3,
          "hand q dims", qdim, "text dims", emb.shape[1], flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:] or list(ARMS):
        build(ds)
