"""Hand model registry (config/hands.yaml) -> CanonicalHand instances."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

from .kinematics.canonical import CanonicalHand

REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "config" / "hands.yaml"


@functools.lru_cache
def registry() -> dict:
    return yaml.safe_load(open(REGISTRY_PATH))


@functools.lru_cache
def verification() -> dict:
    return yaml.safe_load(open(REPO / "config" / "verification.yaml"))


def mapping_verification(mapping_id: str) -> dict:
    """{'status': 'verified'|'excluded'|'unreviewed', 'checks': ..., 'reason': ...}"""
    return verification()["mappings"].get(mapping_id, {"status": "unreviewed", "reason": "not in config/verification.yaml"})


def pad_normal_verification(hand_model_id: str) -> dict:
    """Model-level check of the pad-normal definition for one hand model (config/verification.yaml: pad_normals)."""
    return verification().get("pad_normals", {}).get(hand_model_id, {"status": "unreviewed"})


@functools.lru_cache
def load_hand(hand_model_id: str) -> CanonicalHand:
    spec = registry()["hand_models"][hand_model_id]
    tips = {f: {k: (str(REPO / v) if k == "mesh_extremal" else v) for k, v in t.items()} for f, t in spec["tips"].items()}
    return CanonicalHand(str(REPO / spec["urdf"]), spec["side"], spec["palm_link"], spec.get("wrist_frame"),
                         spec["finger_base_joints"], tips, spec.get("wrist_point"))
