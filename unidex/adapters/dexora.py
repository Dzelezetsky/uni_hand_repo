"""Dexora Real-World (Dexora/Dexora_Real-World_Dataset, LeRobot v2.1): 2x AIRBOT arms + 2x XHand1, 20 fps.

Episode-centric folders airbot_{articulation,assemble,dexterous,pick_and_place} are used; the `dexora/` task-level
folder is a second VIEW of the same episodes and is skipped (no duplicates).
Text: meta/episode_instruction_mapping.jsonl has `instruction` (natural language) and `action_name`; they do not
always describe the same task (e.g. airbot_articulation ep 0: action_name place_wipes_blocks..., instruction "open the
laptop"). Both are kept; instruction_original = instruction (what LeRobot `tasks` stores).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "dexora"
ROOT = Path("raw_data/dexora")
SUBSETS = ("airbot_articulation", "airbot_assemble", "airbot_dexterous", "airbot_pick_and_place")
SL = {"left_arm": slice(0, 6), "right_arm": slice(6, 12), "left": slice(12, 24), "right": slice(24, 36),
      "head": slice(36, 38), "spine": slice(38, 39)}
CAMS = {"top": "head", "wrist_left": "wrist_left", "wrist_right": "wrist_right", "front": "third_person"}
ACTIVE_STD = 1e-3


@functools.lru_cache
def _meta(sub):
    info = json.load(open(ROOT / sub / "meta/info.json"))
    m = {j["episode_index"]: j for j in map(json.loads, open(ROOT / sub / "meta/episode_instruction_mapping.jsonl"))}
    return info, m


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(f"{sub}/{p.stem}" for sub in SUBSETS for p in (root / sub).glob("data/*/*.parquet"))


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    sub, name = eid.split("/")
    idx = int(name.split("_")[-1])
    info, mp = _meta(sub)
    ch = idx // info.get("chunks_size", 1000)
    d = pd.read_parquet(root / sub / f"data/chunk-{ch:03d}/{name}.parquet").sort_values("frame_index")
    t = d["timestamp"].to_numpy(float)
    t = t - t[0]
    st = np.stack(d["observation.state"].to_numpy()).astype(float)
    ac = np.stack(d["action"].to_numpy()).astype(float)
    names = info["features"]["observation.state"]["names"]
    hands = []
    for side in ("left", "right"):
        q = st[:, SL[side]]
        hands.append(HandStream(
            side=side, hand_family="xhand1", t=t, native_q=q, native_names=names[SL[side]], native_units="rad",
            state_source="measured_joint", hand_model_id=f"xhand1_{side}", mapping_id=f"dexora__xhand1_{side}",
            native_action=ac[:, SL[side]], action_t=t, action_names=names[SL[side]],
            status_override=None if q.std(0).max() > ACTIVE_STD else "hand_not_used_in_episode",
            notes="state = /observation/<side>/joint_state (Dexora airbot_lerobot.py); action = teleop command"))
    cams = []
    for key, role in CAMS.items():
        vp = root / sub / f"videos/chunk-{ch:03d}/observation.images.{key}/{name}.mp4"
        cams.append(Camera(camera_id=key, camera_role=role, rgb_ref=str(vp), fps=float(info["fps"]), width=640,
                           height=480, frame_t=d["frame_index"].to_numpy() / float(info["fps"]), local=vp.exists()))
    streams = {k: Stream(t=t, data=st[:, SL[k]], names=names[SL[k]], units="rad", source="measured_joint")
               for k in ("left_arm", "right_arm", "head", "spine")}
    j = mp.get(idx, {})
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="airbot_play_x2+xhand1_x2", t0_unix=None, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=j.get("instruction"), instruction_en=j.get("instruction"),
        annotation_source="dataset" if j.get("instruction") else "none", annotation_level="task", language="en",
        extra={"subset": sub, "action_id": j.get("action_id"), "action_name": j.get("action_name"),
               "category": j.get("category"),
               "text_note": "action_name and instruction are not always consistent in the release (see adapter doc)"},
    )
