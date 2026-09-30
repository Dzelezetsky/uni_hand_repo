"""Stage-1 data build (run with the UnifiedDex venv: needs pandas/pyarrow; training only needs numpy/safetensors).

For every episode with >= 1 verified hand stream and a local primary-camera video, write
  stage1_data/episodes/<dataset>/<episode>.safetensors
    frame_t        (F,)    float64  time of each video frame of the primary camera (s, episode clock)
    frame_offset   ()      int64    index of the episode's first frame inside rgb_ref (LeRobot v3 concatenated files)
    <side>_t       (T,)    float64  hand sample times (s, same clock)          side in {left, right}, only if verified
    <side>_geo     (T,15)  float32  fingertips_palm_norm (canonical, palm widths)
    <side>_valid   (T,)    uint8
    <side>_cls     (T,)    int8     hand_posture_class_v1 (-1 invalid / missing)
and stage1_data/index.jsonl (one line per episode: dataset, episode, file, rgb_ref, fps, width, height, t0, t1,
instruction_sha, sides, family per side) + stage1_data/instructions.json {sha: text}.
Video is NOT copied or re-encoded (read from the original files at training time).
usage: python stage1/build_data.py [dataset ...] [--out stage1_data]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

sys.path.insert(0, ".")
from unidex.hands import REPO, UNIFIED  # noqa: E402

UNI = UNIFIED
PRIMARY_CAMERA = {"humanoid_everyday_h1": "egocentric_d435", "egosteer": "head", "trex": "head_left",
                  "sharpa_origami": "head_left", "dexora": "top", "openarm_banana": "head",
                  "vitra_teledata": "head_d455"}
FAMILIES = ["Shadow", "DFX", "F1", "XHand", "Sharpa", "Ruiyan"]
MODEL = json.loads((REPO / "config/hand_posture_class_v1.json").read_text())


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def resolve_video(ref: str) -> Path | None:
    p = REPO / ref
    if p.exists():
        return p
    if (REPO / (ref + ".mp4")).exists():
        return REPO / (ref + ".mp4")
    return None


def one(args):
    ds, ep, ep_dir, cam, hands, out_file, labels = args
    fr = pd.read_parquet(UNI / ep_dir / "camera_frames.parquet")
    fr = fr[fr.camera_id == cam["camera_id"]].sort_values("frame_index")
    if len(fr) < 10:
        return None
    arrays = {"frame_t": fr.t_s.to_numpy(np.float64), "frame_offset": np.array(int(cam["frame_index_offset"] or 0))}
    sides, fams = [], []
    for side, h in hands.items():
        d = pd.read_parquet(UNI / h["path"], columns=["frame_index", "t_s", "fingertips_palm_norm", "valid"])
        geo = np.stack(d.fingertips_palm_norm.to_numpy()).reshape(len(d), 15).astype(np.float32)
        cls = np.full(len(d), -1, np.int8)
        lab = labels.get(side)
        if lab is not None:
            m = pd.Series(lab[1], index=lab[0])
            cls = m.reindex(d.frame_index.to_numpy()).fillna(-1).to_numpy().astype(np.int8)
        arrays.update({f"{side}_t": d.t_s.to_numpy(np.float64), f"{side}_geo": geo,
                       f"{side}_valid": d.valid.to_numpy().astype(np.uint8), f"{side}_cls": cls})
        sides.append(side)
        fams.append(FAMILIES.index(MODEL["family_of_hand_model"][h["hand_model_id"]]))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(out_file))
    return dict(dataset=ds, episode=ep, sides=sides, families=fams, t0=float(arrays["frame_t"][0]),
                t1=float(arrays["frame_t"][-1]))


def build(datasets, out: Path):
    import pyarrow.parquet as pq
    hs = pd.read_parquet(UNI / "hand_streams.parquet")
    hs = hs[hs.canonicalization_status == "ok"]
    cams = pd.read_parquet(UNI / "cameras.parquet")
    man = pd.read_parquet(UNI / "manifest.parquet").set_index(["dataset_id", "source_episode_id"])
    instr = json.loads((out / "instructions.json").read_text()) if (out / "instructions.json").exists() else {}
    index_path = out / "index.jsonl"
    old = [json.loads(l) for l in index_path.read_text().splitlines()] if index_path.exists() else []
    rows = [r for r in old if r["dataset"] not in datasets]
    for ds in datasets:
        cam_id = PRIMARY_CAMERA[ds]
        c = cams[(cams.dataset_id == ds) & (cams.camera_id == cam_id)].set_index("source_episode_id")
        h = hs[hs.dataset_id == ds]
        lab = pq.read_table(UNI / "derived/hand_posture_class_v1" / f"{ds}.parquet",
                            columns=["source_episode_id", "side", "frame_index", "posture_class"]).to_pandas()
        L = {k: (g.frame_index.to_numpy(), g.posture_class.to_numpy())
             for k, g in lab.groupby(["source_episode_id", "side"], observed=True)}
        jobs, meta = [], {}
        missing_video = 0
        for ep, g in h.groupby("source_episode_id"):
            if ep not in c.index:
                continue
            cr = c.loc[ep]
            video = resolve_video(cr.rgb_ref)
            if video is None:
                missing_video += 1
                continue
            hands = {r.side: {"path": r.path, "hand_model_id": r.hand_model_id} for r in g.itertuples()}
            text = man.loc[(ds, ep), "instruction_en"] or ""
            instr[sha(text)] = text
            f = out / "episodes" / ds / f"{ep.replace('/', '__')}.safetensors"
            meta[ep] = dict(file=str(f.relative_to(out)), rgb_ref=str(video.relative_to(REPO)), fps=float(cr.fps),
                            width=int(cr.width) if cr.width == cr.width else None,
                            height=int(cr.height) if cr.height == cr.height else None, instruction_sha=sha(text))
            jobs.append((ds, ep, man.loc[(ds, ep), "episode_dir"], cr.to_dict(), hands, f,
                         {s: L[(ep, s)] for s in hands if (ep, s) in L}))
        n = 0
        with ProcessPoolExecutor(12) as ex:
            for r in ex.map(one, jobs, chunksize=16):
                if r is not None:
                    rows.append({**r, **meta[r["episode"]]})
                    n += 1
        print(f"{ds}: {n} episodes written, {missing_video} skipped (primary video '{cam_id}' not local)", flush=True)
    index_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "instructions.json").write_text(json.dumps(instr, indent=0))
    print(f"index: {len(rows)} episodes, {len(instr)} unique instructions -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(PRIMARY_CAMERA))
    ap.add_argument("--out", type=Path, default=REPO / "stage1_data")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    build(a.datasets, a.out)
