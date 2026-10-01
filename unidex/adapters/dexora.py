"""Dexora Real-World (Dexora/Dexora_Real-World_Dataset, LeRobot v2.1): 2x AIRBOT arms + 2x XHand1, 20 fps.

Episode-centric folders airbot_{articulation,assemble,dexterous,pick_and_place} are used; the `dexora/` task-level
folder is a second VIEW of the same episodes and is skipped (no duplicates).
Text: meta/episode_instruction_mapping.jsonl has `instruction` (natural language) and `action_name`. The
action/task NAMES are unreliable (40-75% disagree with the instruction; checked on video: articulation ep 0 opens a
laptop, action_name place_wipes_blocks...; pick_and_place ep 3000 moves a book, task place_coffee_cups_in_basin);
the instructions matched the video in every checked case. instruction_original = instruction; action_name is kept in
`extra` only as provenance and must not be used as text.
Release bug (fixed here): some mapping rows repeat an earlier episode_index (pick_and_place lines 2084-2260 restart
at 0; assemble line 1404 and dexterous line 1337 say 0). Such a row belongs to the episode of its line number (checked:
its action_name then equals the episode parquet's task_index name for every row; pick_and_place ep 0 / 2084 on video).
A row is used only if its action_name equals the parquet's task name. pick_and_place episodes 2261-6296 have no
mapping row in the release (meta covers 2261 of 6297) -> annotation_source "none".
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
    tasks = {j["task_index"]: j["task"] for j in map(json.loads, open(ROOT / sub / "meta/tasks.jsonl"))}
    m = {}
    for line_no, j in enumerate(map(json.loads, open(ROOT / sub / "meta/episode_instruction_mapping.jsonl"))):
        if j["episode_index"] in m:  # release bug: a repeated index belongs to the episode of its line number
            j = {**j, "mapping_note": f"index {j['episode_index']} repeated at line {line_no}"}
            m[line_no] = j
        else:
            m[j["episode_index"]] = j
    return info, m, tasks


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(f"{sub}/{p.stem}" for sub in SUBSETS for p in (root / sub).glob("data/*/*.parquet"))


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    sub, name = eid.split("/")
    idx = int(name.split("_")[-1])
    info, mp, tasks = _meta(sub)
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
    task_name = tasks[int(d["task_index"].iloc[0])]
    j = mp.get(idx, {})
    if j and j["action_name"] != task_name:  # mapping row not aligned with this episode -> no text
        j = {}
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="airbot_play_x2+xhand1_x2", t0_unix=None, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=j.get("instruction"), instruction_en=j.get("instruction"),
        annotation_source="dataset" if j.get("instruction") else "none", annotation_level="task", language="en",
        extra={"subset": sub, "action_id": j.get("action_id"), "action_name": j.get("action_name"),
               "category": j.get("category"), "mapping_note": j.get("mapping_note"),
               "text_note": "action_name/task names are unreliable in the release; use the instruction (adapter doc)"},
    )
