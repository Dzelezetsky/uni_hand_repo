"""Derived layer hand_posture_class_v1: one posture class per valid sample of every VERIFIED hand stream.

Model frozen in config/hand_posture_class_v1.json (KMeans k = 7 centroids on the 10 pairwise fingertip distances,
family-balanced over 6 hand families; GRASP_LABELING.md "Result 2026-09-30 (2)"). A label is a POSTURE, not a grasp.

Output: unified/derived/hand_posture_class_v1/<dataset_id>.parquet, one row per sample:
  source_episode_id, side, hand_model_id, frame_index, t_s,
  posture_class (int8, -1 = invalid sample), dist (z-space distance to the centroid), margin (2nd - 1st distance),
  reachable (bool: this hand family can form the class, reachable_rms < threshold)
+ unified/derived/hand_posture_class_v1.json (model provenance + class shares per dataset / family).
Usage: python -m unidex.posture
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd

from .hands import REPO, UNIFIED

VERSION = "hand_posture_class_v1"
MODEL = REPO / "config" / f"{VERSION}.json"
UNI = UNIFIED
OUT = UNI / "derived" / VERSION
PAIRS = list(combinations(range(5), 2))


def load_model():
    m = json.loads(MODEL.read_text())
    m["_mu"], m["_sd"] = np.array(m["mu"]), np.array(m["sd"])
    m["_cz"] = (np.array(m["centroids"]) - m["_mu"]) / m["_sd"]
    return m


def classify(tips_norm: np.ndarray, m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tips_norm (T, 5, 3) -> (class, dist, margin)."""
    X = np.stack([np.linalg.norm(tips_norm[:, i] - tips_norm[:, j], axis=1) for i, j in PAIRS], 1)
    d = np.linalg.norm(((X - m["_mu"]) / m["_sd"])[:, None] - m["_cz"][None], axis=2)
    s = np.sort(d, 1)
    return d.argmin(1), s[:, 0], s[:, 1] - s[:, 0]


def _label_dataset(args):
    ds, rows = args
    m = load_model()
    thr = m["reach_threshold_pw"]
    out = []
    for r in rows:
        fr = pd.read_parquet(UNI / r["path"], columns=["frame_index", "t_s", "valid", "fingertips_palm_norm"])
        tips = np.stack(fr.fingertips_palm_norm.to_numpy()).reshape(-1, 5, 3)
        c, dist, marg = classify(tips, m)
        valid = fr.valid.to_numpy() & np.isfinite(tips).all((1, 2))
        c = np.where(valid, c, -1).astype(np.int8)
        reach = np.array(m["reachable_rms"][m["family_of_hand_model"][r["hand_model_id"]]]) < thr
        out.append(pd.DataFrame({
            "source_episode_id": r["source_episode_id"], "side": r["side"], "hand_model_id": r["hand_model_id"],
            "frame_index": fr.frame_index.to_numpy(), "t_s": fr.t_s.to_numpy(),
            "posture_class": c, "dist": np.where(valid, dist, np.nan).astype(np.float32),
            "margin": np.where(valid, marg, np.nan).astype(np.float32),
            "reachable": np.where(c >= 0, reach[np.clip(c, 0, None)], False)}))
    df = pd.concat(out, ignore_index=True)
    for col in ("source_episode_id", "side", "hand_model_id"):
        df[col] = df[col].astype("category")
    df.to_parquet(OUT / f"{ds}.parquet", index=False)
    v = df[df.posture_class >= 0]
    return ds, len(df), v.posture_class.value_counts(normalize=True).sort_index().round(4).to_dict(), \
        float((~v.reachable).mean())


def build():
    m = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    hs = pd.read_parquet(UNI / "hand_streams.parquet")
    hs = hs[hs.canonicalization_status == "ok"]
    unknown = set(hs.hand_model_id) - set(m["family_of_hand_model"])
    assert not unknown, f"hand models without a family in {MODEL.name}: {unknown}"
    jobs = [(ds, g[["path", "source_episode_id", "side", "hand_model_id"]].to_dict("records"))
            for ds, g in hs.groupby("dataset_id")]
    summary = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for ds, n, share, unreach in ex.map(_label_dataset, jobs):
            summary[ds] = {"samples": n, "class_share": share, "share_in_unreachable_class": round(unreach, 4)}
            print(ds, n, share, f"unreachable {unreach:.2%}", flush=True)
    meta = {k: m[k] for k in ("version", "feature_version", "clustering_version", "canonical_version", "classes")}
    meta.update(model_file=str(MODEL.relative_to(REPO)), per_dataset=summary)
    (UNI / "derived" / f"{VERSION}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    build() if len(sys.argv) < 2 or sys.argv[1] == VERSION else sys.exit(f"unknown version {sys.argv[1]}")
