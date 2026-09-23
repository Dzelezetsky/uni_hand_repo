"""HRDexDB (HRDexDB/HRDexDB): xArm6 + Inspire RH56DFTP / RH56F1 (right), up to ~24 synchronized cameras.
Allegro and gripper subsets are excluded (not five-finger)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "hrdexdb"
ROOT = Path("raw_data/hrdexdb")
EMBODIMENTS = {
    "inspire_dftp": dict(family="inspire_rh56dftp", model="inspire_rh56dftp_right_hrdexdb",
                         mapping="hrdexdb__inspire_rh56dftp_right",
                         names=["little", "ring", "middle", "index", "thumb_bend", "thumb_rotation"],
                         units="inspire_0_1000"),
    "inspire_f1": dict(family="inspire_rh56f1", model="inspire_rh56f1_right_hrdexdb",
                       mapping="hrdexdb__inspire_rh56f1_right",
                       names=["thumb_rotation", "thumb_bend", "index", "middle", "ring", "little"],
                       units="inspire_f1_0_1800"),
}


def _meta():
    return (pd.read_parquet(ROOT / "metadata/episodes.parquet"), pd.read_parquet(ROOT / "metadata/cameras.parquet"))


def list_episodes(root: Path = ROOT) -> list[str]:
    return sorted(str(p.parent.relative_to(root)) for emb in EMBODIMENTS
                  for p in root.glob(f"{emb}/*/*/raw") if (p / "hand").exists())


def _frame_times(frame_id, t, n_frames):
    """raw/timestamps: 1-based video frame ids with capture times; frames without a timestamp stay NaN."""
    out = np.full(n_frames, np.nan)
    ok = (frame_id >= 1) & (frame_id <= n_frames)
    out[frame_id[ok] - 1] = t[ok]
    return out


def _f(a):
    return np.asarray(a, dtype=float)


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    eps, cams_df = _meta()
    row = eps[eps.episode_id == eid].iloc[0]
    emb = row.embodiment
    cfg = EMBODIMENTS[emb]
    d = root / eid
    raw = d / "raw"
    if emb == "inspire_dftp":
        q, tq = _f(np.load(raw / "hand/position.npy")), _f(np.load(raw / "hand/time.npy"))
        a, ta = _f(np.load(raw / "hand/action.npy")), tq
        tact = None
    else:
        q = _f(np.load(raw / "hand/right_joint_states.npy", allow_pickle=True))
        tq = _f(np.load(raw / "hand/right_joint_states_time.npy", allow_pickle=True))
        a = _f(np.load(raw / "hand/right_commands.npy", allow_pickle=True))
        ta = _f(np.load(raw / "hand/right_commands_time.npy", allow_pickle=True))
        tact = np.load(raw / "hand/right_tactile.npy", allow_pickle=True)
    vid_t = _f(np.load(raw / "timestamps/timestamp.npy"))
    vid_fid = np.load(raw / "timestamps/frame_id.npy")
    arm_q, arm_t = _f(np.load(raw / "arm/position.npy", allow_pickle=True)), _f(np.load(raw / "arm/time.npy", allow_pickle=True))
    t0 = float(min(tq.min(), vid_t.min(), arm_t.min()))

    hand = HandStream(side="right", hand_family=cfg["family"], t=tq - t0, native_q=q, native_names=cfg["names"],
                      native_units=cfg["units"], state_source="measured_actuator", hand_model_id=cfg["model"],
                      mapping_id=cfg["mapping"], native_action=a, action_t=ta - t0, action_names=cfg["names"])
    intr = json.load(open(d / "cam_param/intrinsics.json"))
    extr = json.load(open(d / "cam_param/extrinsics.json"))
    cameras = []
    for _, c in cams_df[cams_df.episode_id == eid].iterrows():
        cid = str(c.camera_id)
        local = (root / c.video_path).exists()
        ci = intr.get(cid, {})
        cameras.append(Camera(
            camera_id=cid, camera_role="third_person_static",
            rgb_ref=str(root / c.video_path) if local else f"hf://datasets/HRDexDB/HRDexDB/{c.video_path}",
            fps=float(c.fps), width=int(c.width), height=int(c.height),
            intrinsics=ci.get("intrinsics_undistort") or ci.get("original_intrinsics"),
            extrinsics=(extr[cid] + [[0, 0, 0, 1]]) if cid in extr else None,
            extrinsics_frame="3x4 world->camera [R|t] from cam_param/extrinsics.json (+ bottom row); C2R.npy maps "
                             "camera-world to robot base",
            frame_t=_frame_times(vid_fid, vid_t - t0, int(c.num_frames)), local=local))
    streams, issues = {}, []
    if len(arm_q) == len(arm_t):
        streams["arm_joint"] = Stream(t=arm_t - t0, data=arm_q, units="rad", source="measured_joint", frame="xarm6")
    else:  # ambiguous alignment in the source: do not guess which row is extra
        issues.append(f"arm stream not imported: position {len(arm_q)} rows vs time {len(arm_t)}")
    if emb == "inspire_dftp":
        streams["hand_force"] = Stream(t=tq - t0, data=_f(np.load(raw / "hand/force.npy")), units="inspire_raw",
                                       source="measured_actuator")
    if tact is not None and len(tact):
        keys = sorted(tact[0].keys())
        streams["tactile_right"] = Stream(
            t=_f(np.load(raw / "hand/right_tactile_time.npy", allow_pickle=True)) - t0,
            data=np.stack([np.concatenate([np.ravel(np.asarray(x[k], float)) for k in keys]) for x in tact]),
            names=keys, units="raw", source="measured_tactile")
    gr = json.load(open(d / "grasp_result.json")) if (d / "grasp_result.json").exists() else {}
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id=f"xarm6+{cfg['family']}", t0_unix=t0,
        duration_s=float(max(tq.max(), vid_t.max()) - t0), hands=[hand], cameras=cameras, streams=streams,
        instruction_original=None, instruction_en=None, annotation_source="none", annotation_level=None,
        extra={"object_id": row.object_id, "grasp_success": gr.get("grasp_success"),
               "paired_human_episode": row.paired_episode_id, "C2R": np.load(d / "C2R.npy").tolist(),
               "object_pose_ref": row.object_pose_path, "mesh_ref": row.mesh_path,
               "video_frame_ids": vid_fid.tolist(), "import_issues": issues},
    )
