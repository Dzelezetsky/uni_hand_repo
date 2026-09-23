"""Canonical palm frame and 5-fingertip geometry (KINEMATICS.md v0.1).

Palm landmarks are taken from the hand's own kinematic model:
  w          wrist point           = origin of `wrist_frame` (a link of the hand model)
  b_I..b_P   finger base points    = origins of the FIRST joint of each finger chain (fixed on the palm)
Axes:
  +Y  wrist -> mean(b_I, b_M, b_R, b_P)
  +X  pinky base -> index base, orthogonalized against Y
  +Z  X x Y
  origin o = b (knuckle centroid),  scale s = |b_I - b_P|

Origin choice: KINEMATICS.md allows (w + b)/2 or a validated alternative. The wrist point is defined very differently
across URDFs (flange, wrist pivot, CAD origin), so we only use it for the +Y direction and put the origin at the
knuckle centroid, which is anatomically comparable across hands.

Because all landmarks are fixed on the palm link, the canonical frame is a constant transform per hand model.

Left hands: coordinates are computed in the left hand's own anatomical frame and then Z is negated, which maps a
left hand onto the right-hand convention (reflection across the sagittal plane; see tests/test_canonical.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fk import HandFK

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CANONICAL_VERSION = "fingertips_palm_v0.1+knuckle_origin"


@dataclass
class PalmFrame:
    R: np.ndarray       # (3, 3) columns = canonical x, y, z expressed in the FK root (palm link) frame
    origin: np.ndarray  # (3,)
    scale: float        # |b_I - b_P| in meters
    landmarks: dict = field(default_factory=dict)


def palm_frame_from_landmarks(w, b_index, b_middle, b_ring, b_pinky) -> PalmFrame:
    w, bI, bM, bR, bP = (np.asarray(v, float) for v in (w, b_index, b_middle, b_ring, b_pinky))
    b = (bI + bM + bR + bP) / 4
    y = b - w
    y /= np.linalg.norm(y)
    x0 = bI - bP
    x = x0 - (x0 @ y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return PalmFrame(R=np.stack([x, y, z], axis=1), origin=b, scale=float(np.linalg.norm(bI - bP)),
                     landmarks=dict(wrist=w, base_index=bI, base_middle=bM, base_ring=bR, base_pinky=bP))


def to_canonical(points_root: np.ndarray, frame: PalmFrame, side: str) -> np.ndarray:
    """points_root (..., 3) in FK root frame -> canonical right-hand convention coordinates (..., 3)."""
    p = (points_root - frame.origin) @ frame.R
    if side == "left":
        p = p * np.array([1.0, 1.0, -1.0])
    elif side != "right":
        raise ValueError(side)
    return p


def _tip_offset(spec: dict) -> np.ndarray:
    if "mesh_extremal" in spec:
        import trimesh
        v = np.asarray(trimesh.load(spec["mesh_extremal"], force="mesh").vertices)
        return v[np.argmax(np.linalg.norm(v, axis=1))]
    return np.asarray(spec.get("offset", [0.0, 0.0, 0.0]), float)


class CanonicalHand:
    """FK + canonicalization for one hand model (one side)."""

    def __init__(self, urdf_path: str, side: str, palm_link: str, wrist_frame: str | None,
                 finger_base_joints: dict[str, str], tips: dict[str, dict], wrist_point: list | None = None):
        """tips[finger] = {"link": name, "offset": [x, y, z]} (offset in that link's frame, default 0)
        or {"link": name, "mesh_extremal": stl_path}: offset = mesh vertex farthest from the link origin."""
        self.side = side
        self.fk = HandFK(urdf_path, root_link=palm_link)
        if wrist_point is not None:  # explicit point in palm-link coordinates (model has no wrist frame)
            w = np.asarray(wrist_point, float)
        elif wrist_frame == palm_link:
            w = np.zeros(3)
        else:
            w = self.fk.link_poses(np.zeros((1, 0)), [], [wrist_frame], zero_missing=True)[wrist_frame][0, :3, 3]
        bases = {f: self.fk.joint_origin_in_root(finger_base_joints[f]) for f in ("index", "middle", "ring", "pinky")}
        self.frame = palm_frame_from_landmarks(w, bases["index"], bases["middle"], bases["ring"], bases["pinky"])
        self.tip_links = [tips[f]["link"] for f in FINGERS]
        self.tip_offsets = np.array([_tip_offset(tips[f]) for f in FINGERS], float)

    def fingertips_root(self, q: np.ndarray, joint_names: list[str]) -> np.ndarray:
        """(T, 5, 3) fingertip positions in the palm-link (FK root) frame."""
        poses = self.fk.link_poses(q, joint_names, self.tip_links)
        out = []
        for link, off in zip(self.tip_links, self.tip_offsets):
            P = poses[link]
            out.append(P[:, :3, 3] + np.einsum("tij,j->ti", P[:, :3, :3], off))
        return np.stack(out, axis=1)

    def canonical(self, q: np.ndarray, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (fingertips_palm_m (T,5,3), fingertips_palm_norm (T,5,3))."""
        m = to_canonical(self.fingertips_root(q, joint_names), self.frame, self.side)
        return m, m / self.frame.scale
