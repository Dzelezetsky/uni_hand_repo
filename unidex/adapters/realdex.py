"""RealDex (4DVLab/RealDex): UR10e + Shadow Dexterous Hand E (right), 4x Azure Kinect RGB-D. Grasping only."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "realdex"
ROOT = Path("raw_data/realdex")
BAGS = ROOT / "extracted/storage/group/4dvlab/youzhuo/bags"
ZIP_PREFIX = "storage/group/4dvlab/youzhuo/bags"
SHADOW_22 = ["FFJ4", "FFJ3", "FFJ2", "FFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1",
             "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"]


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(str(p.parent.relative_to(BAGS)) for p in BAGS.glob("*/*/final_data.npy"))


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    d = BAGS / eid
    obj = eid.split("/")[0]
    fd = np.load(d / "final_data.npy", allow_pickle=True).item()
    ts_ns = np.loadtxt(d / "rgbimage_timestamp.txt").astype(np.int64)  # one row per exported frame (= qpos row)
    q = np.asarray(fd["qpos"], float)
    assert len(q) == len(ts_ns), (len(q), len(ts_ns))
    t0 = ts_ns[0]
    t = (ts_ns - t0) / 1e9
    hand = HandStream(side="right", hand_family="shadow_dexterous_hand_e", t=t, native_q=q, native_names=SHADOW_22,
                      native_units="rad", state_source="measured_joint", hand_model_id="shadow_e_right_realdex",
                      mapping_id="realdex__shadow_e_right",
                      notes="joint angles recovered by the authors from ROS TF (robot_state_publisher of encoders)")
    wrist = np.concatenate([np.asarray(fd["global_transl"]), np.asarray(fd["global_orient"]).reshape(-1, 9)], axis=1)
    streams = {"rh_forearm_pose_world": Stream(t=t, data=wrist, names=["x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)],
                                               units="m,rotmat", frame="world", source="ros_tf")}
    contact_path = d / "contact.txt"
    if contact_path.exists():
        cf = np.loadtxt(contact_path, dtype=int, ndmin=1)
        flag = np.zeros(len(t), bool)
        flag[cf[cf < len(t)]] = True
        streams["contact_flag"] = Stream(t=t, data=flag[:, None].astype(float), names=["in_contact"],
                                         source="dataset contact.txt (frame indices)")
    zip_path = ROOT / "zips" / f"{obj}.zip"
    cam_params = json.load(open(ROOT / "zips/camera_param.json")) if (ROOT / "zips/camera_param.json").exists() else {}
    cameras = []
    for c in range(4):
        cameras.append(Camera(
            camera_id=f"cam{c}", camera_role="third_person_static",
            rgb_ref=f"zip://{zip_path}!{ZIP_PREFIX}/{eid}/cam{c}/rgb/image_raw/",
            depth_ref=f"zip://{zip_path}!{ZIP_PREFIX}/{eid}/cam{c}/depth_to_rgb/image_raw/",
            fps=None, frame_t=None, intrinsics=cam_params.get(f"cam{c}", {}).get("K") if isinstance(cam_params, dict) else None,
            local=zip_path.exists()))
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="ur10e+shadow_dexterous_hand_e", t0_unix=t0 / 1e9, duration_s=float(t[-1]),
        hands=[hand], cameras=cameras, streams=streams,
        instruction_original=None, instruction_en=None, annotation_source="none", annotation_level=None,
        extra={"object_id": obj, "rgb_frame_timestamps_ns_file": str(d / "rgbimage_timestamp.txt"),
               "object_pose_note": f"final_data object_* has {len(fd['object_transl'])} rows vs {len(t)} frames: "
                                   "not frame-aligned in this export, not imported"},
    )
