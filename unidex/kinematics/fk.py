"""Batched forward kinematics for hand URDFs (revolute / prismatic / fixed joints, URDF <mimic> coupling).

The URDF is parsed with yourdfpy (no meshes). FK is evaluated for a whole trajectory at once:
    q: (T, n_actuated) in the order of `HandFK.actuated_joints`
    returns {link_name: (T, 4, 4)} poses expressed in `root_link`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yourdfpy


def _axis_angle_batch(axis, angle):
    """Rodrigues: axis (3,), angle (T,) -> (T, 3, 3)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # (4, 4)
    axis: np.ndarray
    mimic_of: str | None
    multiplier: float
    offset: float
    lower: float | None
    upper: float | None


class HandFK:
    def __init__(self, urdf_path: str, root_link: str | None = None):
        robot = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=False,
                                   build_collision_scene_graph=False, load_collision_meshes=False)
        self.urdf_path = urdf_path
        self.joints: dict[str, _Joint] = {}
        for j in robot.robot.joints:
            origin = np.eye(4) if j.origin is None else np.asarray(j.origin, float)
            lim = j.limit
            self.joints[j.name] = _Joint(
                name=j.name, type=j.type, parent=j.parent, child=j.child, origin=origin,
                axis=np.array([1.0, 0, 0]) if j.axis is None else np.asarray(j.axis, float),
                mimic_of=j.mimic.joint if j.mimic is not None else None,
                multiplier=float(j.mimic.multiplier) if j.mimic is not None and j.mimic.multiplier is not None else 1.0,
                offset=float(j.mimic.offset) if j.mimic is not None and j.mimic.offset is not None else 0.0,
                lower=None if lim is None or lim.lower is None else float(lim.lower),
                upper=None if lim is None or lim.upper is None else float(lim.upper),
            )
        self.child_joint = {j.child: j for j in self.joints.values()}
        links = {l.name for l in robot.robot.links}
        roots = [l for l in links if l not in self.child_joint]
        self.root_link = root_link or (roots[0] if len(roots) == 1 else None)
        if self.root_link is None:
            raise ValueError(f"ambiguous root links {roots}; pass root_link")
        self.links = links
        self.actuated_joints = [j.name for j in robot.robot.joints
                                if j.type in ("revolute", "continuous", "prismatic") and j.mimic is None]

    def chain(self, link: str) -> list[_Joint]:
        """Joints from root_link down to `link`."""
        out = []
        while link != self.root_link:
            if link not in self.child_joint:
                raise KeyError(f"{link} is not below {self.root_link}")
            j = self.child_joint[link]
            out.append(j)
            link = j.parent
        return out[::-1]

    def joint_values(self, q: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
        """Map (T, n) values given in `joint_names` order to all movable joints (mimic joints resolved)."""
        q = np.atleast_2d(np.asarray(q, float))
        vals = {n: q[:, i] for i, n in enumerate(joint_names)}
        pending = [j for j in self.joints.values() if j.mimic_of is not None and j.name not in vals]
        while pending:  # resolve (possibly chained) mimic joints
            left = [j for j in pending if j.mimic_of not in vals]
            for j in pending:
                if j.mimic_of in vals:
                    vals[j.name] = j.multiplier * vals[j.mimic_of] + j.offset
            if len(left) == len(pending):
                break  # mimic of a joint without a value: reported as missing in link_poses
            pending = left
        return vals

    def link_poses(self, q: np.ndarray, joint_names: list[str], links: list[str],
                   zero_missing: bool = False) -> dict[str, np.ndarray]:
        """`zero_missing` is only for rest-pose queries (e.g. fixed palm landmarks), never for data."""
        q = np.atleast_2d(np.asarray(q, float))
        T = q.shape[0]
        vals = self.joint_values(q, joint_names)
        if zero_missing:
            for j in self.joints.values():
                vals.setdefault(j.name, np.zeros(T))
        cache: dict[str, np.ndarray] = {self.root_link: np.broadcast_to(np.eye(4), (T, 4, 4)).copy()}

        def pose(link):
            if link in cache:
                return cache[link]
            j = self.child_joint[link]
            M = pose(j.parent) @ j.origin[None]
            if j.type in ("revolute", "continuous"):
                if j.name not in vals:
                    raise KeyError(f"no value for joint {j.name}")
                L = np.broadcast_to(np.eye(4), (T, 4, 4)).copy()
                L[:, :3, :3] = _axis_angle_batch(j.axis, vals[j.name])
                M = M @ L
            elif j.type == "prismatic":
                L = np.broadcast_to(np.eye(4), (T, 4, 4)).copy()
                L[:, :3, 3] = vals[j.name][:, None] * (j.axis / np.linalg.norm(j.axis))[None]
                M = M @ L
            cache[link] = M
            return M

        return {l: pose(l) for l in links}

    def joint_origin_in_root(self, joint_name: str) -> np.ndarray:
        """Origin of a joint frame in root_link coordinates, with any upstream joints at zero.
        For the first joint of a finger (parent = palm) this is a fixed palm landmark."""
        j = self.joints[joint_name]
        P = self.link_poses(np.zeros((1, 0)), [], [j.parent], zero_missing=True)[j.parent]
        return (P[0] @ j.origin)[:3, 3]
