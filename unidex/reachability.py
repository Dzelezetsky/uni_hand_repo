"""Kinematic reachability of a (dataset, hand) mapping, sampled in the NATIVE hand space.

Sampling URDF joints independently is wrong for coupled hands (several URDFs model coupled joints as independent);
the native control space pushed through the verified mapping respects the real coupling. Every native range states
its source. Samples whose model joints leave the URDF limits by more than LIMIT_TOL are dropped (fraction reported).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hands import REPO, UNIFIED, load_hand
from .mappings import MAPPINGS

LIMIT_TOL = 0.05  # rad, same as the converter's limit QA


@dataclass(frozen=True)
class NativeRange:
    lo: np.ndarray | None  # None -> taken from the data (per channel min/max over the unified store)
    hi: np.ndarray | None
    source: str


NATIVE_RANGES = {
    "realdex__shadow_e_right": NativeRange(
        None, None, "URDF joint limits (identity mapping). Shadow J1/J2 are one tendon, but the split between them is "
                    "free under contact (underactuated), so independent sampling within limits is physically possible."),
    "humanoid_everyday_h1__inspire_rh56dfx_right": NativeRange(
        np.zeros(6), np.ones(6), "Unitree normalization 0..1 (xr_teleoperate robot_hand_inspire.py)"),
    "humanoid_everyday_h1__inspire_rh56dfx_left": NativeRange(
        np.zeros(6), np.ones(6), "Unitree normalization 0..1 (xr_teleoperate robot_hand_inspire.py)"),
    "dexora__xhand1_right": NativeRange(None, None, "URDF joint limits (identity rad mapping)"),
    "trex__sharpa_wave_right": NativeRange(None, None, "URDF joint limits (identity rad mapping)"),
    "egosteer__ruiyan_ryh2_right": NativeRange(
        np.zeros(6), np.array([0.6, 1, 1, 1, 1, 1.0]), "RY-H2 motor range of the authors' FK (robot-stack hand_fk_node.py)"),
    "hrdexdb__inspire_rh56f1_right": NativeRange(
        None, None, "data-observed p0.5..p99.5 per channel over the unified store (no official F1 range at hand): "
                    "reach AS USED in HRDexDB, may underestimate the hardware range"),
}


def _data_range(mapping_id: str):
    hs = pd.read_parquet(UNIFIED / "hand_streams.parquet")
    hs = hs[(hs.mapping_id == mapping_id) & (hs.canonicalization_status == "ok")]
    x = np.concatenate([np.stack(pd.read_parquet(UNIFIED / p, columns=["native_q"]).native_q.to_numpy())
                        for p in hs.path])
    return np.nanpercentile(x, 0.5, 0), np.nanpercentile(x, 99.5, 0)  # robust: rare all-zero frames exist


def sample(mapping_id: str, n: int = 200_000, seed: int = 0) -> dict:
    """Uniform samples in native space -> {'raw', 'model_q', 'names', 'tips_norm' (n,5,3), 'normals' (n,5,3), 'info'}."""
    m = MAPPINGS[mapping_id]
    hand = load_hand(m.hand_model_id)
    names = list(m.model_joints)
    rng_spec = NATIVE_RANGES[mapping_id]
    lo, hi = native_box(mapping_id)
    raw = lo + np.random.default_rng(seed).random((n, len(lo))) * (hi - lo)
    q = m(raw)
    jl = np.array([hand.fk.joints[j].lower if hand.fk.joints[j].lower is not None else -np.inf for j in names])
    ju = np.array([hand.fk.joints[j].upper if hand.fk.joints[j].upper is not None else np.inf for j in names])
    ok = ((q >= jl - LIMIT_TOL) & (q <= ju + LIMIT_TOL)).all(1)
    q = q[ok]
    return dict(raw=raw[ok], model_q=q, names=names, tips_norm=hand.canonical(q, names)[1],
                normals=hand.canonical_pad_normals(q, names),
                info=dict(mapping_id=mapping_id, hand_model_id=m.hand_model_id, n_sampled=n,
                          dropped_limit_frac=float(1 - ok.mean()), native_lo=lo.tolist(), native_hi=hi.tolist(),
                          range_source=rng_spec.source))


def native_box(mapping_id: str) -> tuple[np.ndarray, np.ndarray]:
    m = MAPPINGS[mapping_id]
    spec = NATIVE_RANGES[mapping_id]
    if spec.lo is not None:
        return spec.lo, spec.hi
    if spec.source.startswith("URDF joint limits"):  # identity mappings: native range = URDF limits
        hand = load_hand(m.hand_model_id)
        return (np.array([hand.fk.joints[j].lower for j in m.model_joints]),
                np.array([hand.fk.joints[j].upper for j in m.model_joints]))
    return _data_range(mapping_id)


def best_fit(mapping_id: str, feature_fn, targets: np.ndarray, init_raw: np.ndarray, iters: int = 60,
             with_normals: bool = False) -> tuple:
    """Batched box-constrained Levenberg-Marquardt in NATIVE space: for every target feature vector find native hand
    parameters of `mapping_id` minimizing |feature_fn(tips_norm) - target|. Returns (raw (N,d), residual (N,F)).
    Optimizes u in [0,1]^d (raw = lo + u * span) so damping is scale-free across native units.
    with_normals=True: feature_fn(tips_norm, pad_normals) instead of feature_fn(tips_norm)."""
    m = MAPPINGS[mapping_id]
    hand = load_hand(m.hand_model_id)
    names = list(m.model_joints)
    lo, hi = native_box(mapping_id)
    span = np.maximum(hi - lo, 1e-12)

    def feats(u):
        q = m(lo + u * span)
        if with_normals:
            return feature_fn(hand.canonical(q, names)[1], hand.canonical_pad_normals(q, names))
        return feature_fn(hand.canonical(q, names)[1])

    u = np.clip((np.asarray(init_raw, float) - lo) / span, 0, 1)
    N, d = u.shape
    lam = np.full(N, 1e-2)
    r = feats(u) - targets
    cost = (r ** 2).sum(1)
    h = 1e-4
    for _ in range(iters):
        up = np.repeat(u[:, None], d, 1) + np.eye(d)[None] * h           # (N, d, d) forward differences
        fp = feats(up.reshape(-1, d)).reshape(N, d, -1)
        J = (fp - (r + targets)[:, None]) / h                           # (N, d, F)
        A = J @ J.transpose(0, 2, 1)                                    # (N, d, d)
        g = J @ r[..., None]
        damp = lam[:, None] * (np.einsum("nii->ni", A) + 1e-3)          # +1e-3: joints that do not move any feature
        step = -np.linalg.solve(A + damp[..., None] * np.eye(d)[None], g)[..., 0]
        un = np.clip(u + step, 0, 1)
        rn = feats(un) - targets
        cn = (rn ** 2).sum(1)
        better = cn < cost
        u[better], r[better], cost[better] = un[better], rn[better], cn[better]
        lam = np.clip(np.where(better, lam * 0.3, lam * 10), 1e-6, 1e8)
    return lo + u * span, r
