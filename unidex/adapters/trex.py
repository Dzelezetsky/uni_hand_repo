"""T-Rex (zekaiwang/trex_dataset, LeRobot v3.0): Dexmate Vega-1 + 2x Sharpa Wave (22 joints each), 30 fps.

Proprioception of all episodes is small (data/ ~3.3 GB) and is imported completely; RGB is referenced in the
concatenated v3 video files (videos/<key>/chunk-XXX/file-YYY.mp4, episode segment = [from_timestamp, to_timestamp)).
camera_frames.frame_index = frame index INSIDE that file (round(from_timestamp * fps) + i), so readers decode the
original file directly. Cameras are local only where the video file was downloaded.
"""
from __future__ import annotations

import functools
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "trex"
ROOT = Path("raw_data/trex")
FPS = 30.0
ARM, HAND = 7, 22
SLICES = {"left_arm": slice(0, 7), "left": slice(7, 29), "right_arm": slice(29, 36), "right": slice(36, 58)}
RGB_KEYS = {"head_left": "head", "left_wrist": "wrist_left", "right_wrist": "wrist_right"}
ACTIVE_STD = 1e-3


@functools.lru_cache
def _meta():
    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "meta/episodes/**/*.parquet"),
                                                                  recursive=True))])
    return m.set_index("episode_index").sort_index()  # includes caption / motor_primitive / object / target


@functools.lru_cache
def _info():
    return json.load(open(ROOT / "meta/info.json"))


@functools.lru_cache(maxsize=4)
def _data_file(chunk, file):
    return pd.read_parquet(ROOT / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet")


def list_episodes(root: Path = ROOT) -> list[str]:
    m = _meta()
    have = {(int(c), int(f)) for c, f in m[["data/chunk_index", "data/file_index"]].drop_duplicates().values
            if (root / f"data/chunk-{int(c):03d}/file-{int(f):03d}.parquet").exists()}
    keep = [i for i, r in m.iterrows() if (int(r["data/chunk_index"]), int(r["data/file_index"])) in have]
    return [f"episode_{i:06d}" for i in keep]


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    idx = int(eid.split("_")[-1])
    r = _meta().loc[idx]
    d = _data_file(int(r["data/chunk_index"]), int(r["data/file_index"]))
    d = d[d.episode_index == idx].sort_values("frame_index")
    t = d["timestamp"].to_numpy(float)
    t = t - t[0]
    st = np.stack(d["observation.state"].to_numpy()).astype(float)
    ac = np.stack(d["action"].to_numpy()).astype(float)
    names = _info()["features"]["observation.state"]["names"]
    anames = _info()["features"]["action"]["names"]
    hands = []
    for side in ("left", "right"):
        q = st[:, SLICES[side]]
        hands.append(HandStream(
            side=side, hand_family="sharpa_wave", t=t, native_q=q, native_names=names[SLICES[side]],
            native_units="rad", state_source="measured_joint", hand_model_id=f"sharpa_wave_{side}",
            mapping_id=f"trex__sharpa_wave_{side}", native_action=ac[:, SLICES[side]], action_t=t,
            action_names=anames[SLICES[side]],
            status_override=None if q.std(0).max() > ACTIVE_STD else "hand_not_used_in_episode",
            notes="state = measured joint positions; action = 30 Hz joint-space targets (Manus glove retargeting)"))
    cams = []
    for key, role in RGB_KEYS.items():
        c, f = int(r[f"videos/observation.images.{key}/chunk_index"]), int(r[f"videos/observation.images.{key}/file_index"])
        vp = root / f"videos/observation.images.{key}/chunk-{c:03d}/file-{f:03d}.mp4"
        start = int(round(float(r[f"videos/observation.images.{key}/from_timestamp"]) * FPS))
        ft = d["frame_index"].to_numpy() / FPS  # video frame i of the episode <-> row with frame_index i
        cams.append(Camera(camera_id=key, camera_role=role, rgb_ref=str(vp), fps=FPS, width=640, height=360,
                           frame_t=ft, local=vp.exists(), frame_index_offset=start))
    streams = {
        "arm_joint_left": Stream(t=t, data=st[:, SLICES["left_arm"]], names=names[SLICES["left_arm"]], units="rad",
                                 source="measured_joint"),
        "arm_joint_right": Stream(t=t, data=st[:, SLICES["right_arm"]], names=names[SLICES["right_arm"]], units="rad",
                                  source="measured_joint"),
        "tactile_fingertip_wrench": Stream(t=t, data=np.stack(d["observation.tactile_force"].to_numpy()).astype(float),
                                           names=_info()["features"]["observation.tactile_force"]["names"],
                                           units="N,Nm", source="estimated from tactile images (dataset)"),
    }
    caption = r["caption"] if isinstance(r["caption"], str) else str(r["tasks"][0])
    extra = {k: r[k] for k in ("motor_primitive", "object", "target")}
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="dexmate_vega1+sharpa_wave_x2", t0_unix=None, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=caption, instruction_en=caption, annotation_source="dataset",
        annotation_level="task", language="en",
        extra={**{k: (None if not isinstance(v, str) else v) for k, v in extra.items()},
               "caption_note": "human-verified caption (T-Rex README)",
               "tactile_videos": "20 tactile video streams in the original repo (not referenced here)"},
    )
