"""Dataset-native hand vector -> hand-model joint values.

Every mapping states where its ordering / units come from (`evidence`). A mapping never fills joints it does not
know: the model joints it outputs are exactly `model_joints`; any remaining model joint must be a URDF mimic joint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class HandMapping:
    mapping_id: str
    version: str
    hand_model_id: str
    raw_names: tuple[str, ...]
    raw_units: str
    model_joints: tuple[str, ...]
    fn: Callable[[np.ndarray], np.ndarray]
    evidence: str

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, float)
        assert raw.shape[-1] == len(self.raw_names), (self.mapping_id, raw.shape)
        return self.fn(raw)


def _identity(x):
    return x


# ---------------------------------------------------------------- VITRA-TeleData / XHand1 (right)
_XHAND_R = ("right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
            "right_hand_mid_joint1", "right_hand_mid_joint2", "right_hand_ring_joint1", "right_hand_ring_joint2",
            "right_hand_pinky_joint1", "right_hand_pinky_joint2")
VITRA_XHAND = HandMapping(
    mapping_id="vitra_teledata__xhand1_right", version="1.0", hand_model_id="xhand1_right",
    raw_names=("thumb_bend", "thumb_rota1", "thumb_rota2", "index_bend", "index_j1", "index_j2",
               "mid_j1", "mid_j2", "ring_j1", "ring_j2", "pinky_j1", "pinky_j2"),
    raw_units="rad", model_joints=_XHAND_R, fn=_identity,
    evidence="VITRA-TeleData README: /state/right_hand_joint (T,12) rad. Order = XHand SDK order; consistent with "
             "microsoft/VITRA robot_dataset.py XHAND_HUMAN_MAPPING (idx0 thumb flex, 1-2 thumb rot, 3 index abd, "
             "4-5 index flex, 6-7 mid, 8-9 ring, 10-11 pinky) and with SPIDER xhand_right.urdf joint order.",
)

# ---------------------------------------------------------------- RealDex / Shadow Hand E (right)
_SHADOW_22 = ("FFJ4", "FFJ3", "FFJ2", "FFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1",
              "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1")
REALDEX_SHADOW = HandMapping(
    mapping_id="realdex__shadow_e_right", version="1.0", hand_model_id="shadow_e_right_realdex",
    raw_names=_SHADOW_22, raw_units="rad", model_joints=tuple("rh_" + n for n in _SHADOW_22), fn=_identity,
    evidence="RealDex data_preprocess/main.py export_final_data: qpos = [qpos_seq[t][k] for k in qpos_key_list[2:]], "
             "qpos_key_list follows ShadowHandBuilder.joint_names (WRJ2, WRJ1 dropped); angles computed from ROS TF "
             "(utils/kintree.py compute_joint_angle).",
)

# ---------------------------------------------------------------- Humanoid Everyday H1 / Inspire DFX (Unitree)
# Unitree hardware order per hand: pinky, ring, middle, index, thumb-bend, thumb-rotation; 0..1 normalized, 1 = open.
# xr_teleoperate robot_hand_inspire.py: normalized = (max - q) / (max - min) with ranges
#   fingers [0, 1.7], thumb pitch (bend) [0, 0.5], thumb yaw (rotation) [-0.1, 1.3]   -> invert.
_INSPIRE_RANGES = np.array([[0.0, 1.7]] * 4 + [[0.0, 0.5], [-0.1, 1.3]])


def _inspire_unitree_denorm(x):
    lo, hi = _INSPIRE_RANGES[:, 0], _INSPIRE_RANGES[:, 1]
    return hi - np.clip(x, 0.0, 1.0) * (hi - lo)


def _inspire_unitree(side):
    p = "R" if side == "right" else "L"
    return HandMapping(
        mapping_id=f"humanoid_everyday_h1__inspire_rh56dfx_{side}", version="1.0",
        hand_model_id=f"inspire_rh56dfx_{side}_unitree",
        raw_names=("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"), raw_units="normalized_0_1_open",
        model_joints=tuple(f"{p}_{n}" for n in ("pinky_proximal_joint", "ring_proximal_joint", "middle_proximal_joint",
                                                 "index_proximal_joint", "thumb_proximal_pitch_joint",
                                                 "thumb_proximal_yaw_joint")),
        fn=_inspire_unitree_denorm,
        evidence="unitreerobotics/DFX_inspire_service README (order, right hand ids 0-5, left 6-11); "
                 "unitreerobotics/xr_teleoperate robot_hand_inspire.py normalize() ranges and hand_retargeting.py "
                 "inspire_api_joint_names (hardware order -> URDF joint names).",
    )


H1_INSPIRE_RIGHT = _inspire_unitree("right")
H1_INSPIRE_LEFT = _inspire_unitree("left")


def _inspire_robomind(side):
    base = _inspire_unitree(side)
    return HandMapping(
        mapping_id=f"robomind_xsens__inspire_rh56dfx_{side}", version="0.1", hand_model_id=base.hand_model_id,
        raw_names=("little", "ring", "middle", "index", "thumb_bend", "thumb_rotation"),
        raw_units="normalized_0_1", model_joints=base.model_joints, fn=_inspire_unitree_denorm,
        evidence="RoboMIND all_robot_h5_info.md (order: little, ring, middle, index, thumb bend, thumb rotation; "
                 "values 0..1), paper arXiv:2412.13877 (hands = Inspire RH56DFX). Normalization/URDF: Unitree "
                 "xr_teleoperate for the same RH56DFX. ASSUMPTION: 1 = open (Inspire native 1000 = open) and the same "
                 "linear ranges as Unitree; to be visually checked.",
    )


ROBOMIND_INSPIRE_RIGHT = _inspire_robomind("right")
ROBOMIND_INSPIRE_LEFT = _inspire_robomind("left")

# ---------------------------------------------------------------- HRDexDB / Inspire DFTP & F1 (right)
_HRDEX_JOINTS = ("right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint", "right_middle_1_joint",
                 "right_ring_1_joint", "right_little_1_joint")


def _dftp(a):  # verbatim from snuvclab/HRDexDB hrdexdb/common.py inspire_action_to_qpos_dof6
    q = np.zeros((a.shape[0], 6))
    q[:, 0] = 1.40 * (1.0 - a[:, 5] / 1000.0)
    q[:, 1] = 0.60 * (1.0 - a[:, 4] / 1000.0)
    for j, c in zip((2, 3, 4, 5), (3, 2, 1, 0)):
        q[:, j] = (-4e-8 * a[:, c] ** 3 + 3e-5 * a[:, c] ** 2 - 0.0704 * a[:, c] + 83.572) * np.pi / 180.0
    return q


def _f1(a):  # verbatim from snuvclab/HRDexDB hrdexdb/common.py inspire_f1_action_to_qpos_dof6
    q = np.zeros((a.shape[0], 6))
    q[:, 0] = (1800.0 - a[:, 0]) * np.pi / 1800.0
    q[:, 1] = (1350.0 - a[:, 1]) * np.pi / 1800.0
    for j in range(2, 6):
        q[:, j] = (1740.0 - a[:, j]) * np.pi / 1800.0
    return q


HRDEX_DFTP = HandMapping(
    mapping_id="hrdexdb__inspire_rh56dftp_right", version="1.0", hand_model_id="inspire_rh56dftp_right_hrdexdb",
    raw_names=("little", "ring", "middle", "index", "thumb_bend", "thumb_rotation"), raw_units="inspire_0_1000",
    model_joints=_HRDEX_JOINTS, fn=_dftp,
    evidence="snuvclab/HRDexDB hrdexdb/common.py load_robot_qpos + inspire_action_to_qpos_dof6 (raw/hand/position.npy), "
             "URDF xarm_inspire_DFTP.urdf shipped with the dataset.",
)
HRDEX_F1 = HandMapping(
    mapping_id="hrdexdb__inspire_rh56f1_right", version="1.0", hand_model_id="inspire_rh56f1_right_hrdexdb",
    raw_names=("thumb_rotation", "thumb_bend", "index", "middle", "ring", "little"), raw_units="inspire_f1_0_1800",
    model_joints=_HRDEX_JOINTS, fn=_f1,
    evidence="snuvclab/HRDexDB hrdexdb/common.py load_robot_qpos + inspire_f1_action_to_qpos_dof6 "
             "(raw/hand/right_joint_states.npy), URDF xarm_inspire_f1_right.urdf shipped with the dataset.",
)

# ---------------------------------------------------------------- DexWild robot / LEAP Hand V2 Advanced (right)
# Stored stream right_leapv2 = ROS topic /leapv2_node/cmd_raw_leap_r (col 0 = timestamp). It is produced by
# dexwild_ros2 retargeting/leap_v2_ik.py: PyBullet IK on the SAME robot.urdf (byte-identical), then
#   real_q[0:3] = [index_mcps, index_mcpf, index_pip] (pip = dip = avg), ... pinky, thumb = [mcps, mcpf, ip],
#   real_q[15] = palm_thumb, real_q[16] = palm_4_finger, and real_q[[1,4,7,10]] += 0.2 ("for better closure").
# We invert that exactly -> the IK solution in URDF joint space.
_LEAP_RAW = tuple(f"{f}_{j}" for f in ("index", "middle", "ring", "pinky", "thumb")
                  for j in ("mcp_side", "mcp_forward", "curl")) + ("palm_thumb", "palm_4_fingers")
_LEAP_MODEL = ("index_mcps", "index_mcpf", "index_pip", "index_dip",
               "middle_mcps", "middle_mcpf", "middle_pip", "middle_dip",
               "ring_mcps", "ring_mcpf", "ring_pip", "ring_dip",
               "pinky_mcps", "pinky_mcpf", "pinky_pip", "pinky_dip",
               "thumb_mcps", "thumb_mcpf", "thumb_ip", "palm_thumb", "palm_4_finger")
_LEAP_MCPF_OFFSET = 0.2


def _leap_v2(x):
    out = []
    for f in range(4):
        side, fwd, curl = x[:, 3 * f], x[:, 3 * f + 1], x[:, 3 * f + 2]
        out += [side, fwd - _LEAP_MCPF_OFFSET, curl, curl]
    out += [x[:, 12], x[:, 13], x[:, 14], x[:, 15], x[:, 16]]
    return np.stack(out, axis=1)


DEXWILD_LEAP = HandMapping(
    mapping_id="dexwild_robot__leap_v2_adv_right", version="1.0", hand_model_id="leap_v2_adv_right",
    raw_names=_LEAP_RAW, raw_units="rad_command", model_joints=_LEAP_MODEL, fn=_leap_v2,
    evidence="dexwild/dexwild_ros2 retargeting/leap_v2_ik.py compute_IK (joint order, pip=dip, +0.2 on finger MCP "
             "forward), PyBullet joint indexing of the identical robot.urdf; verified: FK of decoded commands matches "
             "the Manus glove IK targets with 21.5 mm median residual (vs 51.7 mm random, 27.7 sign-flipped, 30.8 "
             "thumb-swapped), unified/validation/leap_ik_consistency.txt. Geometry = IK solution, not measured pose.",
)

# ---------------------------------------------------------------- T-Rex / Sharpa Wave (both hands)
# observation.state[58] = [L_arm7 | L_hand22 | R_arm7 | R_hand22], hand joint positions in rad.
# Order = T-Rex dataset_quickstart schema.py SHARPA_HAND_JOINT_ORDER (authors: "verified manually against the data"),
# identical to the joint order of the official Sharpa wave_01 URDF.
SHARPA_HAND_JOINT_ORDER = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP")


def _trex(side):
    return HandMapping(
        mapping_id=f"trex__sharpa_wave_{side}", version="1.0", hand_model_id=f"sharpa_wave_{side}",
        raw_names=tuple(f"{side}_hand_q_{i}" for i in range(22)), raw_units="rad",
        model_joints=tuple(f"{side}_{n}" for n in SHARPA_HAND_JOINT_ORDER), fn=_identity,
        evidence="T-Rex dataset README (observation.state = joint positions, action = targets) + "
                 "github ZhuoyangLiu2005/T-Rex dataset_quickstart/src/trex_dataset_quickstart/schema.py "
                 "SHARPA_HAND_JOINT_ORDER and robot.py (Pinocchio FK on the vendored official Sharpa model).",
    )


TREX_SHARPA_LEFT = _trex("left")
TREX_SHARPA_RIGHT = _trex("right")


def _sharpa_origami(side):
    return HandMapping(
        mapping_id=f"origami__sharpa_wave_{side}", version="1.0", hand_model_id=f"sharpa_wave_{side}",
        raw_names=tuple(f"{side}_hand_j{i}" for i in range(22)), raw_units="rad",
        model_joints=tuple(f"{side}_{n}" for n in SHARPA_HAND_JOINT_ORDER), fn=_identity,
        evidence="SharpaIT/Robotic_Origami_Challenge README (state = joint space, hand j0..j21); channel order NOT "
                 "documented: inferred = Sharpa URDF / T-Rex order because every channel's range fits its URDF joint "
                 "(AA joints around 0, pinky_CMC 0..0.22 of 0.26, thumb_CMC_FE up to 1.42); user accepted 2026-09-30.",
    )


ORIGAMI_SHARPA_LEFT = _sharpa_origami("left")
ORIGAMI_SHARPA_RIGHT = _sharpa_origami("right")

# ---------------------------------------------------------------- Dexora / XHand1 (both hands)
# observation.state[39] = [L_arm6, R_arm6, L_hand12, R_hand12, head2, spine1]; hands in rad (values <= ~1.9).
# Hand order = official Dexora deploy/xhand_forwarder.py (index 0..11) = XHand SDK order = URDF joint names.
_XHAND_ORDER = ("thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2", "index_bend_joint", "index_joint1",
                "index_joint2", "mid_joint1", "mid_joint2", "ring_joint1", "ring_joint2", "pinky_joint1", "pinky_joint2")


def _dexora(side):
    return HandMapping(
        mapping_id=f"dexora__xhand1_{side}", version="1.0", hand_model_id=f"xhand1_{side}",
        raw_names=tuple(f"{side}_hand_joint_{i}" for i in range(1, 13)), raw_units="rad",
        model_joints=tuple(f"{side}_hand_{n}" for n in _XHAND_ORDER), fn=_identity,
        evidence="github dexoravla/Dexora deploy/xhand_forwarder.py joint index table (0 thumb_bend ... 11 "
                 "pinky_joint2); dataprocess/airbot_lerobot.py: state = /observation/<side>/joint_state; HF README.",
    )


DEXORA_XHAND_LEFT = _dexora("left")
DEXORA_XHAND_RIGHT = _dexora("right")

MAPPINGS = {m.mapping_id: m for m in (VITRA_XHAND, REALDEX_SHADOW, H1_INSPIRE_RIGHT, H1_INSPIRE_LEFT,
                                      ROBOMIND_INSPIRE_RIGHT, ROBOMIND_INSPIRE_LEFT,
                                      HRDEX_DFTP, HRDEX_F1, DEXWILD_LEAP,
                                      TREX_SHARPA_LEFT, TREX_SHARPA_RIGHT,
                                      DEXORA_XHAND_LEFT, DEXORA_XHAND_RIGHT,
                                      ORIGAMI_SHARPA_LEFT, ORIGAMI_SHARPA_RIGHT)}
