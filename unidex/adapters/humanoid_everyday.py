"""Humanoid Everyday, H1 subset (USC-PSI-Lab/Humanoid-Everyday-H1, LeRobot v2.1): Unitree H1 + 2x Inspire DFX."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "humanoid_everyday_h1"
ROOT = Path("raw_data/humanoid_everyday_h1")
INSPIRE_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]
# observation.hand_joints = raw DDS hand_state (rt/inspire/state): ids 0-5 right hand, 6-11 left hand
# (unitreerobotics/DFX_inspire_service README). Verified visually on this sample, see report.
SIDE_SLICES = {"right": slice(0, 6), "left": slice(6, 12)}
# H1 intrinsics from the Humanoid-Everyday README (RealSense D435 color, 640x480)
H1_K = [[606.8150634765625, 0.0, None], [0.0, 606.3350219726562, None], [0.0, 0.0, 1.0]]
H1_ACTIVE_STD = 1e-3  # a hand whose 6 values never move is flagged as unused in that episode


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(p.stem for p in root.glob("data/*/*.parquet"))


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    idx = int(eid.split("_")[-1])
    ch = idx // 1000
    df = pd.read_parquet(root / f"data/chunk-{ch:03d}/{eid}.parquet")
    tasks = {j["task_index"]: j for j in map(json.loads, open(root / "meta/tasks.jsonl"))}
    t = df["timestamp"].to_numpy(float)
    hq = np.stack(df["observation.hand_joints"].to_numpy()).astype(float)
    act = np.stack(df["action"].to_numpy()).astype(float)  # [left6, right6] hand (converted angles) + 14 arm
    hands = []
    for side, sl in SIDE_SLICES.items():
        q = hq[:, sl]
        active = bool(q.std(0).max() > H1_ACTIVE_STD)
        hands.append(HandStream(
            side=side, hand_family="inspire_rh56dfx", t=t, native_q=q, native_names=INSPIRE_NAMES,
            native_units="normalized_0_1_open", state_source="measured_actuator",
            hand_model_id=f"inspire_rh56dfx_{side}_unitree", mapping_id=f"humanoid_everyday_h1__inspire_rh56dfx_{side}",
            native_action=act[:, 0:6] if side == "left" else act[:, 6:12], action_t=t,
            action_names=["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"],
            status_override=None if active else "hand_not_used_in_episode",
            notes="action = he2lerobot convert_h1_hand(retargeted angles) -> NOT the same units as state"))
    task = tasks[int(df["task_index"].iloc[0])]
    vid = root / f"videos/chunk-{ch:03d}/egocentric/{eid}.mp4"
    cams = [Camera(camera_id="egocentric_d435", camera_role="head", rgb_ref=str(vid), fps=30.0, width=640, height=480,
                   intrinsics=None, frame_t=t)]
    streams = {"arm_joint": Stream(t=t, data=np.stack(df["observation.arm_joints"].to_numpy()).astype(float),
                                   units="rad", source="measured_joint"),
               "leg_joint": Stream(t=t, data=np.stack(df["observation.leg_joints"].to_numpy()).astype(float),
                                   units="rad", source="measured_joint")}
    instr = task["task"].split("/", 1)[-1].replace("_", " ")
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="unitree_h1+inspire_rh56dfx_x2", t0_unix=None, duration_s=float(t[-1] - t[0]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=task["task"], instruction_en=instr, annotation_source="dataset",
        annotation_level="task", language="en",
        extra={"task_category": task.get("category"), "task_description": task.get("description") or None,
               "instruction_en_derivation": "task name with category prefix stripped and '_' -> ' '"},
    )
