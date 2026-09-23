"""VITRA-TeleData (microsoft/VITRA-TeleData): Realman RM75 arm + XHand1 (12 DoF), head RealSense D455."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "vitra_teledata"
ROOT = Path("raw_data/vitra")
XHAND_NAMES = ["thumb_bend", "thumb_rota1", "thumb_rota2", "index_bend", "index_j1", "index_j2",
               "mid_j1", "mid_j2", "ring_j1", "ring_j2", "pinky_j1", "pinky_j2"]


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(p.stem for p in (root / "annotation").glob("*.h5"))


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    f = h5py.File(root / "annotation" / f"{eid}.h5", "r")
    fps = float(f["meta/fps"][()])
    T = int(f["meta/frame_count"][()])
    t = np.arange(T) / fps  # h5 rows are one-to-one with video frames (README); no per-frame clock is stored
    hands = []
    for side in ("left", "right"):
        if not bool(f[f"meta/has_{side}"][()]):
            continue
        mask = f[f"mask/{side}_hand"][:].astype(bool)
        hands.append(HandStream(
            side=side, hand_family="xhand1", t=t[mask], native_q=f[f"state/{side}_hand_joint"][:][mask],
            native_names=XHAND_NAMES, native_units="rad", state_source="measured_joint",
            hand_model_id=f"xhand1_{side}" if side == "right" else None,
            mapping_id="vitra_teledata__xhand1_right" if side == "right" else None,
            native_action=f[f"action/{side}_hand_joint"][:][mask], action_t=t[mask], action_names=XHAND_NAMES,
            status_override=None if side == "right" else "missing_exact_hand_model",
        ))
    K = np.asarray(f["observation/camera/intrinsics"][:]).tolist()
    video = root / "videos" / f"{eid}.mp4"
    cams = [Camera(camera_id="head_d455", camera_role="head", rgb_ref=str(video), fps=fps, intrinsics=K,
                   extrinsics=np.asarray(f["kinematics/head_camera_to_right_arm_base"][:]).tolist(),
                   extrinsics_frame="T^{head_camera}_{arm_base} (camera pose in right arm base), per episode",
                   frame_t=t)]
    streams = {
        "right_arm_joint": Stream(t=t, data=f["state/right_arm_joint"][:], units="rad", source="measured_joint"),
        "right_hand_mount_pose_in_cam": Stream(t=t, data=f["state/right_hand_mount_pose_in_cam"][:],
                                               names=["x", "y", "z", "rx", "ry", "rz"], units="m,rotvec",
                                               frame="head_camera", source="arm_fk"),
        "right_hand_mount_pose": Stream(t=t, data=f["state/right_hand_mount_pose"][:],
                                        names=["x", "y", "z", "rx", "ry", "rz"], units="m,rotvec", frame="arm_base",
                                        source="arm_fk"),
    }
    instr = f["meta/instruction"][()].decode()
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="realman_rm75+xhand1", t0_unix=None, duration_s=float(t[-1]) if T else 0.0,
        hands=hands, cameras=cams, streams=streams,
        instruction_original=instr, instruction_en=instr, annotation_source="dataset", annotation_level="task",
        language="en",
        extra={"right_ee_urdf_to_hand_mount": np.asarray(f["kinematics/right_ee_urdf_to_hand_mount"][:]).tolist(),
               "time_source": "frame_index/fps (no per-frame clock in source)"},
    )
