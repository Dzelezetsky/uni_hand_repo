"""Derived layer grasp_moments_v1: which hand samples are in hand-object interaction, from MEASURED/annotated
signals only (never from hand posture). Output: unified/derived/grasp_moments_v1.parquet

  hrdexdb (inspire_f1)  object_lifted: tracked object (object_6d_pose_v2, per video frame) is > LIFT_M above its
                        initial height in the robot base frame (inv(C2R) @ pose, z up). An object off the table must
                        be held. Sanity: failed grasps never exceed 0.2 cm, successful ones reach 6-19 cm.
  realdex               contact: dataset annotation contact.txt (authors: hand mesh within 1 mm of the tracked object)
  humanoid_everyday_h1  none: no contact / object signal in the H1 release -> not included
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .hands import REPO

UNI = REPO / "unified"
VERSION = "grasp_moments_v1"
# v2 (2026-09-23): comparable "established grasp" definition across datasets. RealDex uses the authors' grasp
# segments (segment.txt, used by RealDex's own training export): last SEG_END_FRAMES frames of each segment. Objects
# without segment.txt are excluded (their contact.txt also covers approach / release, which is not comparable to
# HRDexDB's lifted-object hold). HRDexDB unchanged (object lifted > 3 cm).
SEG_END_FRAMES = 6
LIFT_M = 0.03


def _hrdexdb(r):
    ep = r.source_episode_id
    emb, obj, seq = ep.split("/")
    d = REPO / "raw_data/hrdexdb" / ep
    pose = REPO / f"raw_data/hrdexdb/object_6d_pose_v2/{emb}/{obj}_{seq}.npz"
    if not pose.exists():  # 4 F1 episodes only have object_6d_pose_v1 (convention not checked) -> excluded
        return np.zeros(len(pd.read_parquet(UNI / r.path)), bool), "excluded_no_object_pose_v2"
    z = np.load(pose)
    keys = sorted(z.files, key=lambda k: int(k.split("_")[1]))
    idx = np.array([int(k.split("_")[1]) for k in keys])
    h = (np.linalg.inv(np.load(d / "C2R.npy"))[None] @ np.stack([z[k] for k in keys]))[:, 2, 3]
    lifted_frames = set(idx[(h - np.median(h[:30])) > LIFT_M])
    cf = pd.read_parquet(UNI / "episodes/hrdexdb" / ep.replace("/", "__") / "camera_frames.parquet")
    cf = cf[cf.camera_id == cf.camera_id.iloc[0]].dropna()
    fr = pd.read_parquet(UNI / r.path)
    near = np.searchsorted(cf.t_s.to_numpy(), fr.t_s.to_numpy()).clip(0, len(cf) - 1)
    frame_of_sample = cf.frame_index.to_numpy()[near]
    dt = np.abs(cf.t_s.to_numpy()[near] - fr.t_s.to_numpy())
    flag = np.array([f in lifted_frames for f in frame_of_sample]) & (dt < 0.05)
    return flag, "object_lifted_gt_3cm"


def _realdex(r):
    d = REPO / "raw_data/realdex/extracted/storage/group/4dvlab/youzhuo/bags" / r.source_episode_id
    cf = set(np.loadtxt(d / "contact.txt", dtype=int, ndmin=1).tolist())
    fr = pd.read_parquet(UNI / r.path)
    return np.array([i in cf for i in fr.frame_index]), "dataset_contact_annotation"


def _realdex_segments(r):
    d = REPO / "raw_data/realdex/extracted/storage/group/4dvlab/youzhuo/bags" / r.source_episode_id
    fr = pd.read_parquet(UNI / r.path)
    flag = np.zeros(len(fr), bool)
    if (d / "segment.txt").exists():
        for a, e in np.loadtxt(d / "segment.txt", dtype=int, ndmin=2):
            flag[max(a, e - SEG_END_FRAMES + 1):e + 1] = True
        return flag, "dataset_grasp_segment_end"
    return flag, "excluded_no_segment_annotation"


SOURCES = {"hrdexdb": _hrdexdb, "realdex": _realdex}
SOURCES_V2 = {"hrdexdb": _hrdexdb, "realdex": _realdex_segments}


def build(version=VERSION):
    sources = SOURCES_V2 if version.endswith("v2") else SOURCES
    hs = pd.read_parquet(UNI / "hand_streams.parquet")
    hs = hs[(hs.canonicalization_status == "ok") & hs.dataset_id.isin(sources)]
    out = []
    for _, r in hs.iterrows():
        flag, src = sources[r.dataset_id](r)
        fr = pd.read_parquet(UNI / r.path)
        out.append(pd.DataFrame({"dataset_id": r.dataset_id, "source_episode_id": r.source_episode_id,
                                 "side": r.side, "hand_model_id": r.hand_model_id, "frame_index": fr.frame_index,
                                 "t_s": fr.t_s, "in_interaction": flag, "moment_source": src}))
        print(f"{r.dataset_id} {r.source_episode_id}: {flag.mean():.0%} of samples ({src})")
    df = pd.concat(out, ignore_index=True)
    (UNI / "derived").mkdir(exist_ok=True)
    df.to_parquet(UNI / "derived" / f"{version}.parquet")
    json.dump({"version": version, "lift_threshold_m": LIFT_M, "seg_end_frames": SEG_END_FRAMES, "doc": __doc__},
              open(UNI / "derived" / f"{version}.json", "w"), indent=1)
    return df


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else VERSION)
