"""RoboMIND (x-humanoid-robomind/RoboMIND), Tien Kung humanoid subsets.

- h5_tienkung_xsens_1rgb: 2x Inspire RH56DFX (paper arXiv:2412.13877: "Tien Kung utilizes two Inspire-Robots RH56DFX
  dexterous hands"), 6 normalized values per hand [little, ring, middle, index, thumb bend, thumb rotation].
  puppet/* is bit-identical to master/* in every sampled episode -> no separate measured state: state_source unknown.
- h5_tienkung_gello_1rgb: same robot, hand stored only as 1D closure per hand -> no per-finger mapping.
No timestamps and no recording rate are published: t is NaN, frame_index is the only clock.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "robomind_tienkung"
ROOT = Path("raw_data/robomind")
INSPIRE_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(str(p.relative_to(root)).replace("/data/trajectory.hdf5", "")
                  for p in root.glob("h5_tienkung_*/**/data/trajectory.hdf5"))


def _instructions():
    p = ROOT / "RoboMIND/static/RoboMIND_v1_2_instr.csv"
    return dict(pd.read_csv(p)[["task", "instruction"]].values) if p.exists() else {}


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    path = root / eid / "data/trajectory.hdf5"
    f = h5py.File(path, "r")
    variant = eid.split("/")[0]
    task = eid.split("/")[1]
    T = f["puppet/joint_position"].shape[0]
    t = np.full(T, np.nan)
    hands, streams = [], {}
    if variant == "h5_tienkung_xsens_1rgb":
        ee = f["puppet/end_effector"][:]
        same = np.array_equal(ee, f["master/end_effector"][:])
        for side, sl in (("left", slice(0, 6)), ("right", slice(6, 12))):
            q = ee[:, sl]
            hands.append(HandStream(
                side=side, hand_family="inspire_rh56dfx", t=t, native_q=q, native_names=INSPIRE_NAMES,
                native_units="normalized_0_1", state_source="unknown",
                hand_model_id=f"inspire_rh56dfx_{side}_unitree", mapping_id=f"robomind_xsens__inspire_rh56dfx_{side}",
                status_override=None if q.std(0).max() > 1e-3 else "hand_not_used_in_episode",
                notes="puppet/end_effector == master/end_effector bit-identical" if same else ""))
        streams["arm_joint"] = Stream(t=t, data=f["puppet/joint_position"][:], units="rad", source="unknown")
    else:  # gello: [left arm 7, left closure, right arm 7, right closure]
        jp = f["puppet/joint_position"][:]
        for side, col in (("left", 7), ("right", 15)):
            hands.append(HandStream(
                side=side, hand_family="inspire_rh56dfx", t=t, native_q=jp[:, col:col + 1],
                native_names=["hand_closure"], native_units="closure_1d", state_source="unknown",
                status_override="missing_joint_mapping",
                notes="1D hand closure only; per-finger state not recorded in the Gello subset"))
        streams["arm_joint"] = Stream(t=t, data=np.delete(jp, [7, 15], axis=1), units="rad", source="unknown")
    lang = f["language_raw"][0].decode() if "language_raw" in f else None
    instr = _instructions().get(task, lang)
    cam = Camera(camera_id="camera_top", camera_role="head", rgb_ref=f"hdf5://{path}!observations/rgb_images/camera_top",
                 depth_ref=f"hdf5://{path}!observations/depth_images/camera_top", width=640, height=480,
                 frame_t=None)
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id=f"tienkung+inspire_rh56dfx_x2 ({variant})", t0_unix=None, duration_s=float("nan"),
        hands=hands, cameras=[cam], streams=streams,
        instruction_original=lang or instr, instruction_en=instr or lang, annotation_source="dataset",
        annotation_level="task", language="en",
        extra={"variant": variant, "n_frames": T, "time_source": "none: no timestamps / rate in the release",
               "frame_index_is_clock": True},
    )
