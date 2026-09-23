"""AgiBot World (Alpha/Beta), dexterous-hand episodes only (state/effector/position has 12 columns).
Hand = AgiBot G1 five-finger dexterous hand, 6 joint angles (rad) per hand; no public kinematic model
(AgiBot-World issue #155) -> native import only."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "agibot_world_dexhand"
ROOT = Path("raw_data/agibot")
CAMS = ["head_color", "hand_left_fisheye_color", "hand_right_fisheye_color", "head_left_fisheye_color",
        "head_right_fisheye_color", "head_center_fisheye_color", "back_left_fisheye_color", "back_right_fisheye_color"]


def list_episodes(root: Path = ROOT) -> list[str]:
    out = []
    for p in sorted(root.glob("proprio_stats/*/*/proprio_stats.h5")):
        with h5py.File(p, "r") as h:
            if h["state/effector/position"].shape[1] == 12:
                out.append(f"{p.parent.parent.name}/{p.parent.name}")
    return out


def _task_info(task, ep):
    for rel in ("AgiBotWorld-Beta", "AgiBotWorld-Alpha"):
        p = ROOT / rel / "task_info" / f"task_{task}.json"
        if p.exists():
            for e in json.load(open(p)):
                if int(e["episode_id"]) == int(ep):
                    return e
    return None


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    task, ep = eid.split("/")
    h = h5py.File(root / "proprio_stats" / task / ep / "proprio_stats.h5", "r")
    ts = h["timestamp"][:].astype(np.int64)
    t0 = ts[0]
    t = (ts - t0) / 1e9
    s, a = h["state/effector/position"][:], h["action/effector/position"][:]
    hands = [HandStream(side=side, hand_family="agibot_g1_dexhand", t=t, native_q=s[:, sl],
                        native_names=[f"j{i}" for i in range(6)], native_units="rad", state_source="measured_joint",
                        native_action=a[:, sl], action_t=t, action_names=[f"j{i}" for i in range(6)],
                        status_override="missing_exact_hand_model",
                        notes="joint order/semantics undocumented; j5 constant in sampled episodes")
             for side, sl in (("left", slice(0, 6)), ("right", slice(6, 12)))]
    streams = {"arm_joint": Stream(t=t, data=h["state/joint/position"][:], units="rad", source="measured_joint"),
               "flange_position": Stream(t=t, data=h["state/end/position"][:].reshape(len(t), -1), units="m",
                                         names=["l_x", "l_y", "l_z", "r_x", "r_y", "r_z"], source="measured"),
               "flange_orientation": Stream(t=t, data=h["state/end/orientation"][:].reshape(len(t), -1),
                                            units="quat_xyzw", source="measured"),
               "head_position": Stream(t=t, data=h["state/head/position"][:], units="rad", source="measured")}
    cams = []
    for c in CAMS:
        p = root / "observations" / task / ep / "videos" / f"{c}.mp4"
        cams.append(Camera(camera_id=c, camera_role="head" if c.startswith("head") else "wrist" if "hand" in c else "body",
                           rgb_ref=str(p) if p.exists() else f"hf://datasets/agibot-world/.../observations/{task} ({ep})",
                           fps=30.0, local=p.exists()))
    info = _task_info(task, ep)
    slices = info["label_info"]["action_config"] if info else []
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="agibot_g1+agibot_dexhand_x2", t0_unix=t0 / 1e9, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=info["task_name"] if info else None, instruction_en=info["task_name"] if info else None,
        annotation_source="dataset" if info else "none", annotation_level="task", language="en",
        extra={"subtask_slices": slices, "init_scene_text": info.get("init_scene_text") if info else None,
               "camera_params": "parameters/*.tar not downloaded"},
    )
