"""DexWild robot subset (boardd/dexwild-dataset *_data/robot): xArm + LEAP Hand V2 Advanced, wrist cameras.
Human (Manus glove) episodes are excluded."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..mappings import _LEAP_RAW
from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "dexwild_robot"
ROOT = Path("raw_data/dexwild")


def _files(root=ROOT):
    return sorted(root.glob("robot_*_first*.hdf5"))


def list_episodes(root: Path = ROOT) -> list[str]:
    out = []
    for f in _files(root):
        task = f.name.split("_")[1]
        with h5py.File(f, "r", locking=False) as h:
            out += [f"{task}/{k}" for k in sorted(h.keys()) if "right_leapv2" in h[k] or "left_leapv2" in h[k]]
    return out


def _arr(g, name):
    o = g[name]
    return (o[name] if isinstance(o, h5py.Group) else o)[()]


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    task, ep = eid.split("/")
    f = [p for p in _files(root) if p.name.startswith(f"robot_{task}_")][0]
    h = h5py.File(f, "r", locking=False)
    g = h[ep]
    hands, streams, t_all = [], {}, []
    for side in ("left", "right"):
        if f"{side}_leapv2" not in g:
            continue
        a = np.asarray(_arr(g, f"{side}_leapv2"), float)
        t_all.append(a[:, 0])
        hands.append((side, a))
        if f"{side}_arm_eef" in g:
            e = np.asarray(_arr(g, f"{side}_arm_eef"), float)
            streams[f"{side}_arm_eef"] = (e[:, 0], e[:, 1:])
    t0 = float(min(x.min() for x in t_all))
    scale = 1e-9 if t0 > 1e15 else 1.0  # ns vs s timestamps
    hs = []
    for side, a in hands:
        hs.append(HandStream(
            side=side, hand_family="leap_hand_v2_advanced", t=(a[:, 0] - t0) * scale, native_q=a[:, 1:],
            native_names=list(_LEAP_RAW), native_units="rad_command", state_source="command",
            hand_model_id=f"leap_v2_adv_{side}" if side == "right" else None,
            mapping_id="dexwild_robot__leap_v2_adv_right" if side == "right" else None,
            status_override=None if side == "right" else "missing_exact_hand_model",
            notes="right_leapv2 = /leapv2_node/cmd_raw_leap_r commanded targets; no measured hand state is stored"))
    st = {k: Stream(t=(tt - t0) * scale, data=v, names=["x", "y", "z", "qx", "qy", "qz", "qw"][:v.shape[1]],
                    units="m,quat(assumed xyzw)", source="arm_state") for k, (tt, v) in streams.items()}
    cams = []
    for cam in ("right_thumb_cam", "right_pinky_cam", "left_thumb_cam", "left_pinky_cam", "zed_obs"):
        if cam in g and len(g[cam]):
            names = sorted(g[cam].keys())
            try:
                ft = (np.array([float(n.split(".")[0]) for n in names]) - t0) * scale
            except ValueError:
                ft = None
            cams.append(Camera(camera_id=cam, camera_role="wrist" if "cam" in cam else "scene",
                               rgb_ref=f"hdf5://{f}!{ep}/{cam}/", frame_t=ft))
        elif cam.startswith("right"):  # exists in the source episode but frames not copied into this sample
            cams.append(Camera(camera_id=cam, camera_role="wrist",
                               rgb_ref=f"hf://datasets/boardd/dexwild-dataset/{task}_data/robot ({ep}/{cam})",
                               local=False))
    dur = max(float(x.t[-1]) for x in hs)
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="xarm+leap_hand_v2_advanced", t0_unix=t0 * scale, duration_s=dur, hands=hs, cameras=cams,
        streams=st, instruction_original=None, instruction_en=None, annotation_source="none",
        extra={"task": task, "source_file": str(f), "source_num_episodes": int(h.attrs.get("source_num_episodes", -1))},
    )
