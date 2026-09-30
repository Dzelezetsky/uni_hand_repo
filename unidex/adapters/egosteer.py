"""EgoSteer-RealWorld (EgoSteer/EgoSteer-RealWorld, LeRobot v3.0, Apache-2.0): RealMan dual RM75-6F arms + two Ruiyan
RY-H2 hands (hand1 = left, hand2 = right), head + chest RealSense D455 (RGB-D, 640x480), 30 Hz (nearest-neighbour
resampling of native 80 Hz hand / 100 Hz arm streams). 54,454 episodes, 193 tasks.

observation.state[74] = measured, action[74] = commanded, same layout: arms 7+7 rad | hands 6+6 motor/4095 |
wrist poses 9+9 | FK fingertips 15+15 in the head-camera (= world) frame. The fingertips are the authors' FK of the
same joints; they are kept as a stream and used to verify our mapping (scripts/analysis/egosteer_fk_check.py).
Text: native task name -> instruction_original/en (annotation_source = dataset, level = task). The per-episode
`instructions` (L1-L3) were written by Qwen3-VL-Flash, corrected by annotators in Chinese and back-translated; they
are kept in extra as VLM-origin text, never as native annotation.
"""
from __future__ import annotations

import functools
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "egosteer"
ROOT = Path("raw_data/egosteer")
SL = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left": slice(14, 20), "right": slice(20, 26),
      "left_wrist": slice(26, 35), "right_wrist": slice(35, 44), "left_tips": slice(44, 59), "right_tips": slice(59, 74)}
ACTIVE_STD = 1e-3


@functools.lru_cache
def _meta(root=ROOT):
    info = json.load(open(root / "meta/info.json"))
    m = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet"))))
    return info, m.set_index("episode_index").sort_index()


@functools.lru_cache(maxsize=2)
def _data_file(root, chunk, file):
    return pd.read_parquet(root / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet")


def list_episodes(root: Path = ROOT) -> list[str]:
    _, m = _meta(root)
    have = {(int(p.parent.name[-3:]), int(p.stem[-3:])) for p in root.glob("data/chunk-*/file-*.parquet")}
    ok = [(c, f) in have for c, f in zip(m["data/chunk_index"], m["data/file_index"])]
    return [f"episode_{i:06d}" for i in m.index[ok]]


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    info, m = _meta(root)
    idx = int(eid.split("_")[-1])
    r = m.loc[idx]
    d = _data_file(root, int(r["data/chunk_index"]), int(r["data/file_index"]))
    d = d[d.episode_index == idx].sort_values("frame_index")
    t = d["timestamp"].to_numpy(float)
    t = t - t[0]
    st = np.stack(d["observation.state"].to_numpy()).astype(float)
    ac = np.stack(d["action"].to_numpy()).astype(float)
    names = info["features"]["observation.state"]["names"]
    fps = float(info["fps"])
    hands = []
    for side in ("left", "right"):
        q = st[:, SL[side]]
        hands.append(HandStream(
            side=side, hand_family="ruiyan_ryh2", t=t, native_q=q, native_names=names[SL[side]],
            native_units="ruiyan_motor_0_1", state_source="measured_actuator",
            hand_model_id=f"ruiyan_ryh2_{side}_egosteer", mapping_id=f"egosteer__ruiyan_ryh2_{side}",
            native_action=ac[:, SL[side]], action_t=t, action_names=names[SL[side]],
            status_override=None if q.std(0).max() > ACTIVE_STD else "hand_not_used_in_episode",
            notes="state = RY-H2 motor position feedback / 4095 (robot-stack hand_control_node.py); action = command"))
    w2c_chest = np.stack(d["observation.camera.chest_world2cam"].to_numpy()).astype(float)[0].reshape(4, 4)
    cams = []
    for key, pose in (("head", np.eye(4)), ("chest", np.linalg.inv(w2c_chest))):
        c, f = int(r[f"videos/observation.images.{key}/chunk_index"]), int(r[f"videos/observation.images.{key}/file_index"])
        vp = root / f"videos/observation.images.{key}/chunk-{c:03d}/file-{f:03d}.mp4"
        dc, df = (int(r[f"videos/observation.images.{key}_depth/chunk_index"]),
                  int(r[f"videos/observation.images.{key}_depth/file_index"]))
        start = int(round(float(r[f"videos/observation.images.{key}/from_timestamp"]) * fps))
        cams.append(Camera(
            camera_id=key, camera_role=key, rgb_ref=str(vp), fps=fps, width=640, height=480,
            intrinsics=np.asarray(r[f"calibration/{key}_intrinsics"], float).reshape(3, 3).tolist(),
            extrinsics=pose.tolist(), extrinsics_frame="camera-to-world, world = head camera frame (dataset convention)",
            depth_ref=f"hf://datasets/EgoSteer/EgoSteer-RealWorld/videos/observation.images.{key}_depth/"
                      f"chunk-{dc:03d}/file-{df:03d}.mp4",
            frame_t=d["frame_index"].to_numpy() / fps, local=vp.exists(), frame_index_offset=start))
    streams = {k: Stream(t=t, data=st[:, SL[k]], names=names[SL[k]],
                         units="rad" if "arm" in k else ("m,rot6d" if "wrist" in k else "m"),
                         source="measured_joint" if "arm" in k else "authors_fk_head_camera_frame")
               for k in ("left_arm", "right_arm", "left_wrist", "right_wrist", "left_tips", "right_tips")}
    task = r["tasks"][0]
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="realman_rm75x2+ruiyan_ryh2_x2", t0_unix=None, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=task, instruction_en=task, annotation_source="dataset", annotation_level="task",
        language="en",
        extra={"split": r["split"], "instructions_vlm": list(r["instructions"]),
               "instructions_vlm_source": "Qwen3-VL-Flash (L1 gist / L2 descriptive / L3 sequential), human-corrected "
                                          "in Chinese, back-translated to English (dataset README)",
               "hand_eye": {k: np.asarray(r[f"calibration/{k}"], float).tolist() for k in
                            ("head_cam_to_left_base", "head_cam_to_right_base", "chest_cam_to_left_base",
                             "chest_cam_to_right_base")}},
    )
