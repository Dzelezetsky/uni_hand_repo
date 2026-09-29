"""In-memory episode representation produced by dataset adapters (UNIFIED_DATASET v0.1).

Nothing here is resampled: every stream keeps its own timestamps (seconds, relative to the episode's t0).
Missing modalities are simply absent (None / empty), never filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "unidex_v0.1"

STATE_SOURCES = ("measured_joint", "measured_actuator", "command", "vision_estimated", "unknown")
ANNOTATION_SOURCES = ("dataset", "human", "template", "vlm", "none")

# canonicalization_status vocabulary (single enum used everywhere)
CANON_OK = "ok"
CANON_MISSING_MODEL = "missing_exact_hand_model"
CANON_MISSING_MAPPING = "missing_joint_mapping"
CANON_HAND_INACTIVE = "hand_not_used_in_episode"
CANON_EXCLUDED = "excluded_not_verified"  # model + mapping exist but did not pass config/verification.yaml


@dataclass
class HandStream:
    side: str                                # "left" | "right"
    hand_family: str                         # physical hand product, e.g. "xhand1", "inspire_rh56dftp"
    t: np.ndarray                            # (T,) seconds rel. to episode t0
    native_q: np.ndarray                     # (T, n) exactly as stored by the dataset
    native_names: list[str]
    native_units: str
    state_source: str                        # see STATE_SOURCES
    hand_model_id: str | None = None         # key in config/hands.yaml; None -> no model
    mapping_id: str | None = None            # key in unidex.mappings.MAPPINGS
    native_action: np.ndarray | None = None  # (Ta, n_a) commanded targets, if the dataset has them separately
    action_t: np.ndarray | None = None
    action_names: list[str] | None = None
    status_override: str | None = None       # force a non-ok canonicalization status (with reason in notes)
    notes: str = ""


@dataclass
class Camera:
    camera_id: str
    camera_role: str                         # e.g. head, wrist_thumb, third_person, egocentric
    rgb_ref: str | None                      # local path / uri (zip://, hf://) to the ORIGINAL video/image source
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    intrinsics: list | None = None           # 3x3
    extrinsics: list | None = None           # 4x4 camera pose
    extrinsics_frame: str | None = None      # what extrinsics are relative to / convention
    depth_ref: str | None = None
    frame_t: np.ndarray | None = None        # (F,) seconds rel. to t0, if known
    local: bool = True                       # False: referenced but not downloaded in this sample
    frame_index_offset: int = 0              # first frame of this episode inside rgb_ref (LeRobot v3 concatenated videos)


@dataclass
class Stream:
    """Any other time series (arm joints, wrist pose, object pose, tactile, contact flags...)."""
    t: np.ndarray
    data: np.ndarray
    names: list[str] | None = None
    units: str = ""
    frame: str = ""
    source: str = ""


@dataclass
class Episode:
    dataset_id: str
    source_episode_id: str
    trajectory_group_id: str
    embodiment_id: str                        # robot platform, e.g. "realman_rm75+xhand1"
    t0_unix: float | None
    duration_s: float
    hands: list[HandStream]
    cameras: list[Camera] = field(default_factory=list)
    streams: dict[str, Stream] = field(default_factory=dict)
    instruction_original: str | None = None
    instruction_en: str | None = None
    annotation_source: str = "none"
    annotation_level: str | None = None       # task | subtask | object | None
    language: str | None = None
    extra: dict = field(default_factory=dict)  # dataset-specific episode metadata (success flag, object id, ...)
