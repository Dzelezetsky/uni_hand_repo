"""Convert dataset episodes into the unified store.

unified/
  manifest.parquet          one row per episode (trajectory group): text, modality flags, durations
  hand_streams.parquet      one row per (episode, hand side): provenance + canonicalization status + QA
  cameras.parquet           one row per (episode, camera): references to ORIGINAL video/images + calibration
  hand_models.parquet       registry snapshot (model status/source/urdf hash, palm frame, scale)
  episodes/<dataset>/<episode>/
      hand_<side>.parquet         t_s, native_q, model_q, fingertips_palm_m[15], fingertips_palm_norm[15], valid
                                  (+ pad_normals_palm[15] unit vectors, only if the hand's pad normals are verified)
      hand_<side>_action.parquet  t_s, native_action (if the dataset has separate commands)
      camera_frames.parquet       camera_id, frame_index, t_s (only where per-frame times are known)
      stream_<name>.parquet       t_s, data (+ names/units in parquet metadata)
      episode.json                full Episode metadata (extra, provenance)

Usage: python -m unidex.convert [dataset ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import schema as S
from .hands import REPO, load_hand, mapping_verification, pad_normal_verification, registry
from .kinematics.canonical import CANONICAL_VERSION, FINGERS, PAD_NORMAL_VERSION
from .mappings import MAPPINGS

OUT = REPO / "unified"
LIMIT_TOL = 0.05  # rad
V5_TOL = 0.15    # rad, verification criterion V5 (config/verification.yaml)
GROSS_TOL = 0.3   # rad beyond a URDF limit = physically impossible (e.g. sensor dropout frames) -> valid = False


def adapters():
    from .adapters import (actionnet, agibot, dexora, dexwild, egosteer, hrdexdb, humanoid_everyday, openarm_banana,
                           origami, realdex, robomind, trex, vitra)
    return {m.DATASET_ID: m for m in (vitra, realdex, humanoid_everyday, hrdexdb, dexwild, robomind, agibot, actionnet,
                                      trex, dexora, origami, openarm_banana, egosteer)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _safe(eid):
    return eid.replace("/", "__")


def _write(df: pd.DataFrame, path: Path, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if meta:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               b"unidex": json.dumps(meta, default=str).encode()})
    pq.write_table(table, path)


def _validation_status(hand_model_id):
    v = registry().get("validation", {}).get(hand_model_id, {})
    return v.get("status", "unvalidated"), v.get("evidence", "")


def canonicalize(h: S.HandStream) -> tuple[dict, pd.DataFrame]:
    """Returns (provenance/QA row, per-sample frame)."""
    reg = registry()["hand_models"]
    spec = reg.get(h.hand_model_id) if h.hand_model_id else None
    mapping = MAPPINGS.get(h.mapping_id) if h.mapping_id else None
    if h.status_override:
        status = h.status_override
    elif spec is None:
        status = S.CANON_MISSING_MODEL
    elif mapping is None:
        status = S.CANON_MISSING_MAPPING
    else:
        status = S.CANON_OK
    ver = mapping_verification(mapping.mapping_id) if mapping else None
    if status == S.CANON_OK and ver["status"] != "verified":  # strict gate: no approximately-right geometry
        status = S.CANON_EXCLUDED
    frame = pd.DataFrame({"frame_index": np.arange(len(h.t)), "t_s": h.t.astype(float),
                          "native_q": list(np.asarray(h.native_q, float))})
    pad_status = pad_normal_verification(h.hand_model_id)["status"] if spec else None
    qa = dict(limit_violation_frac=None, max_limit_violation_rad=None, n_invalid_gross_limit=None, limit_violation_frac_v5=None, nan_frac=float(np.isnan(h.native_q).mean()))
    if status == S.CANON_OK:
        hand = load_hand(h.hand_model_id)
        mq = mapping(h.native_q)
        names = list(mapping.model_joints)
        lo = np.array([hand.fk.joints[n].lower if hand.fk.joints[n].lower is not None else -np.inf for n in names])
        hi = np.array([hand.fk.joints[n].upper if hand.fk.joints[n].upper is not None else np.inf for n in names])
        viol = np.maximum(lo - mq, 0) + np.maximum(mq - hi, 0)
        qa.update(limit_violation_frac=float((viol > LIMIT_TOL).any(1).mean()),
                  max_limit_violation_rad=float(np.nanmax(viol)) if len(viol) else 0.0)
        m, n = hand.canonical(mq, names)
        gross = (viol > GROSS_TOL).any(1)
        valid = np.isfinite(m).all(axis=(1, 2)) & ~gross
        qa["n_invalid_gross_limit"] = int(gross.sum())
        qa["limit_violation_frac_v5"] = float(((viol > V5_TOL) & ~gross[:, None]).any(1).mean())
        frame["model_q"] = list(mq)
        frame["fingertips_palm_m"] = list(m.reshape(len(m), 15).astype(np.float32))
        frame["fingertips_palm_norm"] = list(n.reshape(len(n), 15).astype(np.float32))
        frame["valid"] = valid
        if pad_status == "verified":
            pn = hand.canonical_pad_normals(mq, names)
            frame["pad_normals_palm"] = list(pn.reshape(len(pn), 15).astype(np.float32))
    prov = dict(
        side=h.side, hand_family=h.hand_family, hand_model_id=h.hand_model_id,
        hand_model_status=spec["model_status"] if spec else None,
        hand_model_source=spec["model_source"] if spec else None,
        hand_model_version=_sha(REPO / spec["urdf"]) if spec else None,
        mapping_id=h.mapping_id if mapping else None, mapping_version=mapping.version if mapping else None,
        mapping_evidence=mapping.evidence if mapping else None,
        state_source=h.state_source, geometry_source=(("command_fk" if h.state_source == "command" else
                                                       "actuator_fk" if h.state_source == "measured_actuator" else
                                                       "joint_fk") if status == S.CANON_OK else None),
        native_units=h.native_units, native_dim=int(h.native_q.shape[1]), native_names=list(h.native_names),
        model_joint_names=list(mapping.model_joints) if status == S.CANON_OK else None,
        canonicalization_status=status, canonical_version=CANONICAL_VERSION if status == S.CANON_OK else None,
        pad_normal_status=pad_status if status == S.CANON_OK else None,
        pad_normal_version=PAD_NORMAL_VERSION if status == S.CANON_OK and pad_status == "verified" else None,
        validation_status=_validation_status(h.hand_model_id)[0] if spec else None,
        verification_status=ver["status"] if ver else None,
        verification_reason=(ver.get("reason") or ver.get("note")) if ver else None,
        n_samples=int(len(h.t)), t_start_s=float(h.t[0]) if len(h.t) else None,
        t_end_s=float(h.t[-1]) if len(h.t) else None,
        rate_hz=float((len(h.t) - 1) / (h.t[-1] - h.t[0])) if len(h.t) > 1 and h.t[-1] > h.t[0] else None,
        has_time=bool(np.isfinite(h.t).all()),
        has_native_action=h.native_action is not None, notes=h.notes, **qa)
    return prov, frame


def convert_episode(e: S.Episode, out: Path = OUT) -> tuple[dict, list[dict], list[dict]]:
    d = out / "episodes" / e.dataset_id / _safe(e.source_episode_id)
    base = dict(dataset_id=e.dataset_id, source_episode_id=e.source_episode_id,
                trajectory_group_id=e.trajectory_group_id, embodiment_id=e.embodiment_id)
    hand_rows = []
    for h in e.hands:
        prov, frame = canonicalize(h)
        rel = d / f"hand_{h.side}.parquet"
        _write(frame, rel, {**base, **prov})
        if h.native_action is not None:
            _write(pd.DataFrame({"t_s": h.action_t.astype(float), "native_action": list(np.asarray(h.native_action, float))}),
                   d / f"hand_{h.side}_action.parquet", {**base, "names": h.action_names, "notes": h.notes})
        hand_rows.append({**base, **prov, "path": str(rel.relative_to(out))})
    cam_rows, frames = [], []
    for c in e.cameras:
        cam_rows.append({**base, "camera_id": c.camera_id, "camera_role": c.camera_role, "rgb_ref": c.rgb_ref,
                         "depth_ref": c.depth_ref, "local": c.local, "fps": c.fps, "width": c.width,
                         "height": c.height, "intrinsics": json.dumps(c.intrinsics) if c.intrinsics else None,
                         "extrinsics": json.dumps(c.extrinsics) if c.extrinsics else None,
                         "extrinsics_frame": c.extrinsics_frame, "frame_index_offset": c.frame_index_offset,
                         "n_frames_timed": int(np.isfinite(c.frame_t).sum()) if c.frame_t is not None else 0})
        if c.frame_t is not None:
            frames.append(pd.DataFrame({"camera_id": c.camera_id,
                                        "frame_index": c.frame_index_offset + np.arange(len(c.frame_t)),
                                        "t_s": c.frame_t}))
    if frames:
        _write(pd.concat(frames), d / "camera_frames.parquet", base)
    for name, s in e.streams.items():
        _write(pd.DataFrame({"t_s": np.asarray(s.t, float), "data": list(np.asarray(s.data, float))}),
               d / f"stream_{name}.parquet", {**base, "names": s.names, "units": s.units, "frame": s.frame,
                                              "source": s.source})
    (d / "episode.json").write_text(json.dumps({
        **base, "t0_unix": e.t0_unix, "duration_s": e.duration_s, "instruction_original": e.instruction_original,
        "instruction_en": e.instruction_en, "annotation_source": e.annotation_source,
        "annotation_level": e.annotation_level, "language": e.language, "extra": e.extra,
        "schema_version": S.SCHEMA_VERSION}, indent=1, default=str))
    canon_sides = [r["side"] for r in hand_rows if r["canonicalization_status"] == S.CANON_OK]
    man = {**base, "t0_unix": e.t0_unix, "duration_s": e.duration_s,
           "instruction_original": e.instruction_original, "instruction_en": e.instruction_en,
           "annotation_source": e.annotation_source, "annotation_level": e.annotation_level, "language": e.language,
           "hand_sides": ",".join(h.side for h in e.hands), "canonical_hand_sides": ",".join(canon_sides),
           "n_cameras": len(e.cameras), "n_cameras_local": sum(c.local for c in e.cameras),
           "has_text": e.instruction_original is not None, "has_depth": any(c.depth_ref for c in e.cameras),
           "has_tactile": any(k.startswith("tactile") for k in e.streams),
           "has_object_state": any(k.startswith("object") for k in e.streams),
           "has_contact_labels": "contact_flag" in e.streams,
           "streams": ",".join(e.streams), "episode_dir": str(d.relative_to(out)), "schema_version": S.SCHEMA_VERSION}
    return man, hand_rows, cam_rows


def hand_models_table() -> pd.DataFrame:
    rows = []
    for hid, spec in registry()["hand_models"].items():
        h = load_hand(hid)
        rows.append(dict(hand_model_id=hid, family=spec["family"], side=spec["side"],
                         model_status=spec["model_status"], model_source=spec["model_source"], urdf=spec["urdf"],
                         hand_model_version=_sha(REPO / spec["urdf"]), palm_link=spec["palm_link"],
                         palm_scale_m=h.frame.scale, palm_R=json.dumps(h.frame.R.tolist()),
                         palm_origin=json.dumps(h.frame.origin.tolist()),
                         tip_links=json.dumps(h.tip_links), validation_status=_validation_status(hid)[0],
                         validation_evidence=_validation_status(hid)[1]))
    for fam, m in registry().get("missing_models", {}).items():
        rows.append(dict(hand_model_id=None, family=fam, model_status="missing_exact_model",
                         model_source=m["reason"]))
    return pd.DataFrame(rows)


def main(datasets=None):
    ads = adapters()
    datasets = datasets or list(ads)
    man, hands, cams, errors = [], [], [], []
    for ds in datasets:
        mod = ads[ds]
        for eid in mod.list_episodes():
            try:
                m, h, c = convert_episode(mod.load_episode(eid))
                man.append(m); hands += h; cams += c
                print(f"[{ds}] {eid}: " + ", ".join(f"{r['side']}={r['canonicalization_status']}" for r in h), flush=True)
            except Exception as ex:  # keep going; failures are reported, not hidden
                errors.append(dict(dataset_id=ds, source_episode_id=eid, error=repr(ex)))
                print(f"[{ds}] {eid}: ERROR {ex!r}", flush=True)
    OUT.mkdir(exist_ok=True)

    def merged(rows, name):  # replace only the converted datasets' rows in an existing global table
        new = pd.DataFrame(rows)
        path = OUT / name
        if path.exists():
            old = pd.read_parquet(path)
            new = pd.concat([old[~old.dataset_id.isin(datasets)], new], ignore_index=True)
        return new

    _write(merged(man, "manifest.parquet"), OUT / "manifest.parquet", {"schema_version": S.SCHEMA_VERSION})
    _write(merged(hands, "hand_streams.parquet"), OUT / "hand_streams.parquet",
           {"canonical_version": CANONICAL_VERSION, "fingertip_order": FINGERS})
    _write(merged(cams, "cameras.parquet"), OUT / "cameras.parquet")
    _write(hand_models_table(), OUT / "hand_models.parquet")
    pd.DataFrame(errors, columns=["dataset_id", "source_episode_id", "error"]).to_csv(OUT / "errors.csv", index=False)
    print(f"episodes={len(man)} hand_streams={len(hands)} cameras={len(cams)} errors={len(errors)}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
