"""Mesh rendering of a hand model in the canonical palm frame (orthographic, painter's algorithm, matplotlib only).

Used for manual checks of model-level quantities that a stick skeleton cannot show (e.g. pad normals: does the arrow
leave the finger pad or the nail?).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from matplotlib.collections import PolyCollection

from .hands import REPO, registry
from .kinematics.canonical import CanonicalHand, to_canonical

SEARCH_ROOTS = [REPO / "external_hand_models", REPO / "external_code", REPO / "raw_data"]


@functools.lru_cache(maxsize=None)
def _find_suffix(suffix: str) -> Path | None:
    for root in SEARCH_ROOTS:
        hits = sorted(root.rglob(Path(suffix).name))
        for h in hits:
            if str(h).endswith(suffix):
                return h
    return None


def resolve_mesh(filename: str, urdf_path: Path) -> Path | None:
    if filename.startswith("package://"):
        rel = filename[len("package://"):]
        for d in [urdf_path.parent, *urdf_path.parents]:
            for cand in (d / rel, d / rel.split("/", 1)[1]):
                if cand.exists():
                    return cand
            if d == REPO:
                break
        return _find_suffix(rel)
    p = Path(filename)
    p = p if p.is_absolute() else urdf_path.parent / p
    if p.exists():
        return p
    # broken relative path: only look inside this URDF's own directory (never another robot's mesh of the same name)
    hits = sorted(urdf_path.parent.rglob(p.name))
    return hits[0] if len(hits) == 1 else None


@functools.lru_cache(maxsize=None)
def link_meshes(hand_model_id: str, max_faces: int = 40000) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """{link: [(vertices in link frame (V,3), faces (F,3)), ...]} for all links below the palm link."""
    spec = registry()["hand_models"][hand_model_id]
    urdf = REPO / spec["urdf"]
    robot = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=False,
                               build_collision_scene_graph=False, load_collision_meshes=False).robot
    out = {}
    for link in robot.links:
        for vis in link.visuals:
            g = vis.geometry
            if g is None or g.mesh is None:
                continue
            path = resolve_mesh(g.mesh.filename, urdf)
            if path is None:
                continue
            m = trimesh.load(str(path), force="mesh")
            if g.mesh.scale is not None:
                m.apply_scale(np.broadcast_to(np.asarray(g.mesh.scale, float), (3,)))
            if vis.origin is not None:
                m.apply_transform(np.asarray(vis.origin, float))
            f = np.asarray(m.faces)
            if len(f) > max_faces:  # uniform face subsample: fine for a shaded sketch
                f = f[np.random.default_rng(0).choice(len(f), max_faces, replace=False)]
            out.setdefault(link.name, []).append((np.asarray(m.vertices, float), f))
    return out


def hand_triangles(hand_model_id: str, hand: CanonicalHand, q: np.ndarray, names: list[str],
                   links: list[str] | None = None) -> np.ndarray:
    """(F, 3, 3) triangles of the posed hand in canonical coordinates (meters)."""
    meshes = link_meshes(hand_model_id)
    below = {l for l in meshes if l == hand.fk.root_link or _below(hand, l)}
    links = [l for l in (links or sorted(below)) if l in meshes]
    poses = hand.fk.link_poses(np.atleast_2d(q), names, links)
    tris = []
    for l in links:
        P = poses[l][0]
        for v, f in meshes[l]:
            vw = v @ P[:3, :3].T + P[:3, 3]
            tris.append(to_canonical(vw, hand.frame, hand.side)[f])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def _below(hand, link):
    try:
        hand.fk.chain(link)
        return True
    except KeyError:
        return False


def draw_mesh(ax, tris: np.ndarray, view: np.ndarray, up: np.ndarray, color=(0.75, 0.75, 0.78), alpha=1.0,
              scale=1000.0):
    """Orthographic projection along `view` (unit vector pointing FROM the viewer INTO the scene)."""
    view = view / np.linalg.norm(view)
    right = np.cross(view, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, view)
    xy = np.stack([tris @ right, tris @ up2], axis=-1) * scale
    depth = (tris @ view).mean(1)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(nrm @ view)
    order = np.argsort(-depth)  # far first
    cols = np.clip(np.asarray(color)[None] * shade[order, None], 0, 1)
    ax.add_collection(PolyCollection(xy[order], facecolors=np.c_[cols, np.full(len(cols), alpha)],
                                     edgecolors="none"))
    return right, up2


def project(p: np.ndarray, right: np.ndarray, up2: np.ndarray, scale=1000.0) -> np.ndarray:
    return np.stack([p @ right, p @ up2], axis=-1) * scale
