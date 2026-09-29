"""Sharpa Robotic Origami Challenge (SharpaIT/Robotic_Origami_Challenge): bimanual robot "north_ces" with two Sharpa
Wave hands (22 joints each), 30 fps, CC-BY-4.0. 143 recording sessions ("seasons"), each a LeRobot v3.0 dataset under
season_*/lerobot3.0 (a v2.1 copy of the same data also exists and is not used). Of 143 seasons, 129 have v3 data;
2 have only v2.1 data and 12 no data files at all (videos/meta only) -> those 14 are skipped.

observation.state[65] = [L_arm7 | L_hand22 | R_arm7 | R_hand22 | motor7]. Episodes are long (~7 min) full
paper-airplane folds. The only native task string is the placeholder "north ces task"; instruction_en is the
README's dataset-level description (annotation_source = template).
"""
from __future__ import annotations

import functools
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Camera, Episode, HandStream, Stream

DATASET_ID = "sharpa_origami"
ROOT = Path("raw_data/origami")
SL = {"left_arm": slice(0, 7), "left": slice(7, 29), "right_arm": slice(29, 36), "right": slice(36, 58),
      "motor": slice(58, 65)}
RGB_KEYS = {"head_left": "head", "head_right": "head_right", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
INSTRUCTION_EN = "fold a traditional paper airplane"
ACTIVE_STD = 1e-3


@functools.lru_cache
def _season(season):
    d = ROOT / season / "lerobot3.0"
    info = json.load(open(d / "meta/info.json"))
    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "meta/episodes/**/*.parquet"), recursive=True))])
    return info, m.set_index("episode_index").sort_index()


@functools.lru_cache(maxsize=2)
def _data_file(season, chunk, file):
    return pd.read_parquet(ROOT / season / f"lerobot3.0/data/chunk-{chunk:03d}/file-{file:03d}.parquet")


def list_episodes(root: Path = ROOT) -> list[str]:
    out = []
    for d in sorted(root.glob("season_*/lerobot3.0")):
        season = d.parent.name
        if not any((d / "meta/episodes").glob("**/*.parquet")):
            continue  # 12 seasons ship no data at all, 2 only a v2.1 copy (not used, ~1.5 % of the release)
        _, m = _season(season)
        for i, r in m.iterrows():
            if (d / f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet").exists():
                out.append(f"{season}/episode_{i:06d}")
    return out


def load_episode(eid: str, root: Path = ROOT) -> Episode:
    season, name = eid.split("/")
    idx = int(name.split("_")[-1])
    info, m = _season(season)
    r = m.loc[idx]
    d = _data_file(season, int(r["data/chunk_index"]), int(r["data/file_index"]))
    d = d[d.episode_index == idx].sort_values("frame_index")
    t = d["timestamp"].to_numpy(float)
    t = t - t[0]
    st = np.stack(d["observation.state"].to_numpy()).astype(float)
    ac = np.stack(d["action"].to_numpy()).astype(float)
    names = info["features"]["observation.state"]["names"]
    fps = float(info["fps"])
    hands = []
    for side in ("left", "right"):
        q = st[:, SL[side]]
        hands.append(HandStream(
            side=side, hand_family="sharpa_wave", t=t, native_q=q, native_names=names[SL[side]], native_units="rad",
            state_source="measured_joint", hand_model_id=f"sharpa_wave_{side}", mapping_id=f"origami__sharpa_wave_{side}",
            native_action=ac[:, SL[side]], action_t=t, action_names=names[SL[side]],
            status_override=None if q.std(0).max() > ACTIVE_STD else "hand_not_used_in_episode",
            notes="state = joint-space robot state (README); channel order inferred, see mapping evidence"))
    cams = []
    for key, role in RGB_KEYS.items():
        c, f = int(r[f"videos/observation.images.{key}/chunk_index"]), int(r[f"videos/observation.images.{key}/file_index"])
        vp = root / season / f"lerobot3.0/videos/observation.images.{key}/chunk-{c:03d}/file-{f:03d}.mp4"
        start = int(round(float(r[f"videos/observation.images.{key}/from_timestamp"]) * fps))
        cams.append(Camera(camera_id=key, camera_role=role, rgb_ref=str(vp), fps=fps,
                           frame_t=d["frame_index"].to_numpy() / fps, local=vp.exists(), frame_index_offset=start))
    streams = {
        "arm_joint_left": Stream(t=t, data=st[:, SL["left_arm"]], names=names[SL["left_arm"]], units="rad",
                                 source="measured_joint"),
        "arm_joint_right": Stream(t=t, data=st[:, SL["right_arm"]], names=names[SL["right_arm"]], units="rad",
                                  source="measured_joint"),
        "torso_motor": Stream(t=t, data=st[:, SL["motor"]], names=names[SL["motor"]], source="measured_joint"),
        "tactile_fingertip_wrench": Stream(t=t, data=np.stack(d["observation.tactile"].to_numpy()).astype(float),
                                           names=info["features"]["observation.tactile"]["names"],
                                           units="N,Nm", source="dataset"),
    }
    task = r["tasks"][0] if len(r["tasks"]) else None
    return Episode(
        dataset_id=DATASET_ID, source_episode_id=eid, trajectory_group_id=f"{DATASET_ID}/{eid}",
        embodiment_id="sharpa_north_ces+sharpa_wave_x2", t0_unix=None, duration_s=float(t[-1]),
        hands=hands, cameras=cams, streams=streams,
        instruction_original=task, instruction_en=INSTRUCTION_EN, annotation_source="template",
        annotation_level="dataset", language="en",
        extra={"season": season, "instruction_note": "native task string is a placeholder; instruction_en = README "
               "dataset description (same for every episode)"},
    )
