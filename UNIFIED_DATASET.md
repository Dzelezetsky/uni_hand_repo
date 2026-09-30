# UNIFIED_DATASET.md — Unified Store Format (unidex_v0.1)

Produced by `python -m unidex.convert` from dataset adapters in `unidex/adapters/`.
Nothing is resampled or re-encoded: every stream keeps its native timestamps, videos/images stay where they are and
are referenced.

## Layout

```text
unified/
  manifest.parquet        one row per episode (= trajectory group)
  hand_streams.parquet    one row per (episode, hand side): provenance, canonicalization status, QA
  cameras.parquet         one row per (episode, camera): reference to the ORIGINAL rgb/depth + calibration
  hand_models.parquet     registry snapshot: model status/source/URDF hash, palm frame, palm scale, validation
  errors.csv              episodes that failed conversion (never silently dropped)
  REPORT.md               coverage report (python -m unidex.report)
  validation/*.png        RGB frame + canonical skeleton at the same timestamp, cross-embodiment plots
  episodes/<dataset_id>/<episode_key>/
      hand_<side>.parquet          per hand sample (native rate)
      hand_<side>_action.parquet   commanded targets, if the dataset stores them separately
      camera_frames.parquet        camera_id, frame_index, t_s (NaN = frame without a timestamp)
      stream_<name>.parquet        arm joints, wrist/EEF pose, tactile, contact flags, ...
      episode.json                 episode metadata incl. dataset-specific `extra`
```

All `t_s` are seconds relative to the episode `t0` (`t0_unix` in the manifest when the source has a wall clock).
Training windows must be sampled by time, joining streams by nearest / interpolated timestamps.

## hand_<side>.parquet

| column | meaning |
|---|---|
| `t_s` | sample time |
| `native_q` | hand vector exactly as stored by the dataset (`native_names`, `native_units` in hand_streams) |
| `model_q` | joint values of the hand model (`model_joint_names`), only when canonicalized |
| `fingertips_palm_m` | 15 floats = 5 x XYZ, meters, canonical palm frame, order thumb, index, middle, ring, pinky |
| `fingertips_palm_norm` | same divided by palm scale `s = |b_index - b_pinky|` |
| `valid` | all 15 values finite |

The parquet file metadata (`unidex` key) repeats the full provenance row.

## hand_streams.parquet (provenance, one row per hand stream)

`dataset_id, source_episode_id, trajectory_group_id, embodiment_id, side, hand_family, hand_model_id,
hand_model_status, hand_model_source, hand_model_version (URDF sha256[:16]), mapping_id, mapping_version,
mapping_evidence, state_source, geometry_source, native_units, native_dim, native_names, model_joint_names,
canonicalization_status, canonical_version, validation_status, n_samples, t_start_s, t_end_s, rate_hz,
has_native_action, notes, limit_violation_frac, max_limit_violation_rad, nan_frac, path`

- `state_source`: `measured_joint | measured_actuator | command | vision_estimated | unknown`
- `geometry_source`: `joint_fk | actuator_fk | command_fk` (commands are NOT measured poses)
- `canonicalization_status`: `ok | excluded_not_verified | missing_exact_hand_model | missing_joint_mapping | hand_not_used_in_episode`
  — geometry is written ONLY for `ok`, i.e. (dataset, hand) mappings marked `verified` in `config/verification.yaml`
  (strict policy: no approximately-right geometry is stored). `verification_status/verification_reason` columns explain.
- `validation_status` comes from the `validation:` block of `config/hands.yaml`
  (`visually_checked | auto_checks_only | unvalidated_assumptions | unvalidated`)
- QA: `limit_violation_frac` = fraction of samples where any mapped joint is > 0.05 rad outside the URDF limits.
  Values are never clamped.

## manifest.parquet

Text fields keep provenance: `instruction_original, instruction_en, annotation_source (dataset|human|template|vlm|none),
annotation_level, language`. Modality flags: `has_text, has_depth, has_tactile, has_object_state,
has_contact_labels, n_cameras, n_cameras_local`, plus `hand_sides`, `canonical_hand_sides`, `streams`.

## cameras.parquet

`camera_id, camera_role, rgb_ref, depth_ref, local, fps, width, height, intrinsics (json 3x3), extrinsics (json 4x4),
extrinsics_frame, n_frames_timed`. `rgb_ref` schemes: plain path (mp4), `zip://<zip>!<member dir>/` (RealDex PNGs),
`hdf5://<file>!<group>/` (DexWild frames), `hf://...` (exists in the source dataset but not downloaded here,
`local = False`). No fixed global camera schema; multi-view = several rows with the same `trajectory_group_id`.

## Canonical geometry (see KINEMATICS.md, `unidex/kinematics/canonical.py`)

- Palm landmarks from each hand's own URDF: finger bases = origins of the first joint of index/middle/ring and of
  the pinky knuckle (Shadow: LFJ4 with LFJ5 = 0); wrist = `wrist_frame` origin or an explicit `wrist_point`.
- +Y wrist -> knuckle centroid, +X pinky -> index base, +Z = X x Y (palmar side for right hands).
- Origin = knuckle centroid (v0.1+knuckle_origin; KINEMATICS.md allowed a validated alternative to (w+b)/2 —
  wrist points are not comparable across URDFs).
- Left hands: computed in their own frame, then Z negated -> right-hand convention (unit-tested).

## Derived layers (`unified/derived/`, versioned, never written into the episode files)

- `grasp_moments_v1/v2.parquet` — interaction samples from measured signals only (`unidex/moments.py`).
- `hand_posture_class_v1/<dataset_id>.parquet` (`python -m unidex.posture`; git-ignored, ~0.7 GB, rebuildable) —
  one row per sample of every verified hand stream: `source_episode_id, side, hand_model_id, frame_index, t_s,
  posture_class` (int8, 0..6, -1 invalid), `dist` / `margin` (z-space distance to the nearest / gap to the 2nd
  centroid, for confidence filtering), `reachable` (the hand family can form that class's centroid; False = nearest
  class is outside the hand's range, e.g. F1 in `open_hand`). Frozen model: `config/hand_posture_class_v1.json`
  (centroids, z-normalization, class names, feature/clustering/canonical versions); summary
  `unified/derived/hand_posture_class_v1.json`. Classes are POSTURES, not grasps (GRASP_LABELING.md).
