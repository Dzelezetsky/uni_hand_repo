"""EgoSteer V2 numeric check: our replay of the authors' RY-H2 FK (robot-stack hand_fk_node.py) vs the fingertip
positions shipped in observation.state[44:74]. Frame-free test: the 10 pairwise fingertip distances per hand.
usage: egosteer_fk_check.py [data parquet]"""
import itertools
import sys

import mujoco
import numpy as np
import pandas as pd

MJCF = "external_hand_models/egosteer_ruiyan/ruiyan_hand_mjcf/{}/hand.xml"
MULT = np.array([0.6, 1, 1, 1, 1, 1.0])
MIMIC = [("1_2", "1_3", 1.675), ("2_1", "2_2", 1.0), ("3_1", "3_2", 1.0), ("4_1", "4_2", 1.0), ("5_1", "5_2", 1.0)]
ACTIVE = ["1_1", "1_2", "2_1", "3_1", "4_1", "5_1"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


class HandFK:
    def __init__(self, side):
        self.m = mujoco.MjModel.from_xml_path(MJCF.format(side))
        self.d = mujoco.MjData(self.m)
        self.p = "hand1" if side == "left" else "hand2"
        j = lambda s: self.m.joint(f"{self.p}_joint_link_{s}")
        self.act = [(j(s).qposadr[0], *j(s).range) for s in ACTIVE]
        self.mim = [(j(a).qposadr[0], j(b).qposadr[0], k) for a, b, k in MIMIC]
        self.sites = [self.m.site(f"{side}_{f}_tip").id for f in FINGERS]

    def q(self, state6):
        n = np.clip(state6 / MULT, 0, 1)
        return np.array([lo + x * (hi - lo) for x, (_, lo, hi) in zip(n, self.act)])

    def tips(self, state6):
        self.d.qpos[:] = 0
        for x, (adr, lo, hi) in zip(np.clip(state6 / MULT, 0, 1), self.act):
            self.d.qpos[adr] = lo + x * (hi - lo)
        for a, b, k in self.mim:
            self.d.qpos[b] = k * self.d.qpos[a]
        mujoco.mj_kinematics(self.m, self.d)
        return np.array([self.d.site_xpos[s] for s in self.sites])


def pdist(t):
    return np.array([np.linalg.norm(t[..., i, :] - t[..., j, :], axis=-1) for i, j in itertools.combinations(range(5), 2)]).T


def main(path):
    df = pd.read_parquet(path, columns=["observation.state", "action", "episode_index"])
    S = np.stack(df["observation.state"].to_numpy())
    print(f"{path}: {len(S)} frames, {df.episode_index.nunique()} episodes")
    idx = np.random.default_rng(0).choice(len(S), min(5000, len(S)), replace=False)
    for side, hs, ts in (("left", slice(14, 20), slice(44, 59)), ("right", slice(20, 26), slice(59, 74))):
        fk = HandFK(side)
        h = S[idx, hs]
        print(f"\n[{side}] state ranges min {h.min(0).round(3)} max {h.max(0).round(3)}")
        ours = np.array([fk.tips(x) for x in h])
        theirs = S[idx, ts].reshape(-1, 5, 3)
        e = np.abs(pdist(ours) - pdist(theirs)) * 1000
        print(f"  pairwise tip-distance |ours - shipped| mm: median {np.median(e):.4f}  p99 {np.percentile(e, 99):.4f}  max {e.max():.4f}")
        # rigid fit ours->theirs per frame residual (Kabsch) to be sure it is a pure rigid transform
        r = []
        for a, b in zip(ours[:500], theirs[:500]):
            a0, b0 = a - a.mean(0), b - b.mean(0)
            U, _, Vt = np.linalg.svd(a0.T @ b0)
            D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            r.append(np.linalg.norm((U @ D @ Vt).T @ a0.T - b0.T, axis=0).max())
        print(f"  per-frame rigid-fit residual mm: median {np.median(r)*1000:.4f} max {np.max(r)*1000:.4f}")
        # sensitivity: does a wrong mapping get caught? (inverted direction, no thumb multiplier)
        for name, alt in (("inverted 1-n", lambda x: MULT - x), ("no /0.6 on thumb", lambda x: x * np.r_[0.6, 1, 1, 1, 1, 1])):
            ea = np.abs(pdist(np.array([fk.tips(alt(x)) for x in h[:1000]])) - pdist(theirs[:1000])) * 1000
            print(f"  control [{name}]: mean {ea.mean():.2f} mm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_data/egosteer/data/chunk-000/file-000.parquet")


def check_urdf(path="raw_data/egosteer/data/chunk-000/file-000.parquet"):
    """URDF (scripts/models/mjcf_to_urdf_ruiyan.py) through unidex.kinematics == MuJoCo replay == shipped tips."""
    sys.path.insert(0, ".")
    from unidex.kinematics.fk import HandFK as UrdfFK
    df = pd.read_parquet(path, columns=["observation.state"])
    S = np.stack(df["observation.state"].to_numpy())[:3000]
    for side, hs, ts in (("left", slice(14, 20), slice(44, 59)), ("right", slice(20, 26), slice(59, 74))):
        mj = HandFK(side)
        uf = UrdfFK(f"external_hand_models/egosteer_ruiyan/ruiyan_hand_mjcf/{side}/ruiyan_ryh2_{side}.urdf")
        q = np.array([mj.q(x) for x in S[:, hs]])
        names = [f"{mj.p}_joint_link_{a}" for a in ACTIVE]
        P = uf.link_poses(q, names, [f"{side}_{f}_tip" for f in FINGERS])
        ours = np.stack([P[f"{side}_{f}_tip"][:, :3, 3] for f in FINGERS], 1)
        ref = np.array([mj.tips(x) for x in S[:, hs]])
        print(f"[{side}] URDF vs MuJoCo (base frame) max {np.abs(ours - ref).max() * 1000:.5f} mm; "
              f"URDF vs shipped pairwise max {np.abs(pdist(ours) - pdist(S[:, ts].reshape(-1, 5, 3))).max() * 1000:.5f} mm")
