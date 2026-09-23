"""Fourier ActionNet (FourierIntelligence/ActionNet). Three hand/robot combinations found by probing all 322 tars:
  GR1 (robot 32D) + 6-DoF Fourier hand (hand 12D)  -> FDH-6
  GR2 (robot 29D) + 12-DoF hand (hand 24D)         -> FDH-12 (no public model)
  GR2 (robot 29D) + 6-DoF hand (hand 12D)          -> assumed FDH-6 family, not confirmed
Hand values are Fourier DexHand SDK *motor position params (0..~12), not joint angles* (fourier_dhx DexHand.get_angle
docstring); no public motor->joint transmission -> native import only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "fourier_actionnet"
ROOT = Path("raw_data/actionnet")
FDH6_NAMES = ["pinky_proximal", "ring_proximal", "middle_proximal", "index_proximal", "thumb_proximal_pitch",
              "thumb_proximal_yaw"]  # fourier-lerobot convert_hdf5_to_lerobot.py STATE_HAND6DOF_NAMES (per side)


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(p.stem for p in root.glob("*.hdf5"))


def _meta():
    p = ROOT / "ActionNet/metadata.json"
    return {m["id"]: m for m in json.load(open(p))} if p.exists() else {}


def _cam_times(p):
    out = []
    for s in json.load(open(p)):
        d, frac = s.rsplit("_", 1)
        out.append(datetime.strptime(d, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc).timestamp() + int(frac) / 1e6)
    return np.array(out)


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    f = h5py.File(root / f"{eid}.hdf5", "r")
    ts = f["timestamp"][:].astype(float)
    t0 = ts[0]
    t = ts - t0
    hs, ha = f["state/hand"][:], f["action/hand"][:]
    robot_dim = f["state/robot"].shape[1]
    per = hs.shape[1] // 2
    if per == 6:
        family = "fourier_fdh6" if robot_dim == 32 else "fourier_6dof_on_gr2"
        names = FDH6_NAMES
    else:
        family, names = "fourier_fdh12", [f"j{i}" for i in range(per)]
    hands = []
    for side, sl in (("left", slice(0, per)), ("right", slice(per, 2 * per))):
        q = hs[:, sl]
        active = q.std(0).max() > 1e-2
        hands.append(HandStream(
            side=side, hand_family=family, t=t, native_q=q, native_names=names, native_units="dhx_motor_position",
            state_source="measured_actuator", native_action=ha[:, sl], action_t=t, action_names=names,
            status_override=("hand_not_used_in_episode" if not active else
                             "missing_exact_hand_model" if per == 12 else "missing_joint_mapping"),
            notes="motor position params (fourier_dhx), not joint angles; no public motor->joint transmission"))
    cams = []
    for cdir in sorted((root / eid).glob("*")):
        tj = cdir / "timestamps.json"
        cams.append(Camera(camera_id=cdir.name, camera_role="head" if cdir.name == "top" else cdir.name,
                           rgb_ref=str(cdir / "rgb.mp4"), depth_ref=None,  # depth.mkv exists in source, not downloaded
                           frame_t=(_cam_times(tj) - t0) if tj.exists() else None))
    attrs = {k: (v.decode() if isinstance(v, bytes) else v.tolist() if hasattr(v, "tolist") else v)
             for k, v in f.attrs.items()}
    m = _meta().get(eid, {})
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id=("fourier_gr1t" if robot_dim == 32 else "fourier_gr2") + f"+{family}_x2",
        t0_unix=t0, duration_s=float(t[-1]), hands=hands, cameras=cams,
        streams={"robot_joint": Stream(t=t, data=f["state/robot"][:], units="rad", source="measured_joint"),
                 "pose": Stream(t=t, data=f["state/pose"][:], units="m,rot6d", frame="robot base",
                                names=["l_xyz+rot6d", "r_xyz+rot6d", "head_xyz+rot6d"], source="fk")},
        instruction_original=m.get("prompt"), instruction_en=m.get("prompt"),
        annotation_source="dataset" if m.get("prompt") else "none", annotation_level="task", language="en",
        extra={"task_name": attrs.get("task_name"), "metadata_task": m.get("task"), "attrs": attrs,
               "camera_time_note": "timestamps.json strings parsed as UTC; offset vs robot clock unverified"},
    )
