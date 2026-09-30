"""OpenArm Banana 1072 (June777/openarm_banana_all_1072episodes, LeRobot v2.1, 20 fps): bimanual OpenArm v10 +
Inspire RH56F1 hands, 4 views 640x360. Merge of three captures by the rig's lab (vclab): floor2 (ep 0-329),
american (330-665), monkey (666-1071; resampled from 30 fps, video re-aligned by 0.218 s by the uploader).

Hand values are stored in rad through the uploader's affines; unidex.mappings.BANANA_F1 inverts them to the F1 counts
and applies the official HRDexDB F1 conversion (the uploader's finger affine is self-declared unmeasured).
Right hand only: the left arm is parked in all captures, american's left hand is grafted from floor2 (not measured),
and there is no left F1 model -> left keeps native data only.
Known uploader edits carried as notes: monkey thumb_2 zeroed (resting count under a wrong affine), monkey right-hand
action reconstructed as state[t+8] (dropped here), american ep 533 right-hand feedback dropout (46 frames, flagged
invalid by the converter's gross-limit rule).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "openarm_banana"
ROOT = Path("raw_data/openarm_banana/all_1072")
SL = {"head": slice(0, 2), "left_arm": slice(2, 9), "left": slice(9, 15), "right_arm": slice(15, 22),
      "right": slice(22, 28), "left_eef": slice(28, 37), "right_eef": slice(37, 46)}
CAMS = {"head": "head", "chest": "chest", "external": "third_person", "wristcam_right": "wrist_right"}


@functools.lru_cache
def _meta(root=ROOT):
    info = json.load(open(root / "meta/info.json"))
    eps = {j["episode_index"]: j for j in map(json.loads, open(root / "meta/episodes.jsonl"))}
    return info, eps


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(p.stem for p in root.glob("data/*/*.parquet"))


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    info, eps = _meta(root)
    idx = int(eid.split("_")[-1])
    ch = idx // info["chunks_size"]
    d = pd.read_parquet(root / f"data/chunk-{ch:03d}/{eid}.parquet").sort_values("frame_index")
    t = d["timestamp"].to_numpy(float)
    t = t - t[0]
    st = np.stack(d["observation.state"].to_numpy()).astype(float)
    ac = np.stack(d["action"].to_numpy()).astype(float)
    names = info["features"]["observation.state"]["names"]
    v = eps[idx].get("vclab", {})
    src = v.get("source_dataset")
    right_cmd = v.get("action_source", {}).get("right_hand") == "commanded"
    hands = [
        HandStream(
            side="right", hand_family="inspire_rh56f1", t=t, native_q=st[:, SL["right"]], native_names=names[SL["right"]],
            native_units="rad_uploader_affine", state_source="measured_actuator",
            hand_model_id="inspire_rh56f1_right_hrdexdb", mapping_id="openarm_banana__inspire_rh56f1_right",
            native_action=ac[:, SL["right"]] if right_cmd else None, action_t=t if right_cmd else None,
            action_names=names[SL["right"]] if right_cmd else None,
            notes=f"source capture {src}; state = F1 ANGLEACT feedback through the uploader's affine (inverted by the "
                  "mapping). " + ("monkey: thumb_2 zeroed by the uploader (resting count, thumb never bent); "
                                  "reconstructed action dropped. " if src == "monkey" else "")),
        HandStream(
            side="left", hand_family="inspire_rh56f1", t=t, native_q=st[:, SL["left"]], native_names=names[SL["left"]],
            native_units="rad_uploader_affine", state_source="measured_actuator" if src != "american" else "unknown",
            notes="no left RH56F1 model; left arm parked in all captures"
                  + ("; american left hand GRAFTED from floor2 (not this episode's data)" if src == "american" else "")),
    ]
    cams = []
    for key, role in CAMS.items():
        vp = root / f"videos/chunk-{ch:03d}/observation.images.{key}/{eid}.mp4"
        cams.append(Camera(camera_id=key, camera_role=role, rgb_ref=str(vp), fps=float(info["fps"]), width=640,
                           height=360, frame_t=d["frame_index"].to_numpy() / float(info["fps"]), local=vp.exists()))
    streams = {k: Stream(t=t, data=st[:, SL[k]], names=names[SL[k]], units="rad" if "arm" in k else "m+rot6d_rows",
                         source="measured_joint" if "arm" in k else "fk_of_arm_joints")
               for k in ("left_arm", "right_arm", "left_eef", "right_eef")}
    task = eps[idx]["tasks"][0]
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="openarm_v10_x2+inspire_rh56f1_x2", t0_unix=None, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=task, instruction_en=task, annotation_source="dataset", annotation_level="task",
        language="en",
        extra={"source_capture": src, "vclab": v,
               "camera_note": "view names mapped by physical position by the uploader; monkey 'ego' (now chest) was "
                              "rotated 90 deg in the 400-episode release"},
    )
