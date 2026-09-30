"""Convert the EgoSteer Ruiyan RY-H2 MJCF (egosteer/robot-stack assets/ruiyan_hand_mjcf/<side>/hand.xml, vendored in
external_hand_models/egosteer_ruiyan) into an equivalent URDF for unidex.kinematics.

- bodies -> links, one hinge per body -> revolute joint (origin = body pos/quat, joint pos = 0 in this model)
- fingertip <site>s -> fixed child links <side>_<finger>_tip
- coupling: the AUTHORS' FK code (hand_fk_node.py _get_mimic_relations) is used, i.e. follower = k * leader with
  (1_2 -> 1_3, 1.675) and (x_1 -> x_2, 1.0). This is what produced the dataset's fingertips. (The MJCF <equality>
  polycoef lists joint1 = 1_2, joint2 = 1_3, which in MuJoCo semantics would be the inverse relation for the thumb;
  it is only used by their simulator/IK, not by the recorded FK.)
- per-body mesh geoms -> <visual>; world-level cosmetic geoms other than the base are skipped.
Verified by scripts/analysis/egosteer_fk_check.py --urdf (URDF FK == shipped fingertips).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

DIR = Path("external_hand_models/egosteer_ruiyan/ruiyan_hand_mjcf")
MIMIC = {"1_3": ("1_2", 1.675), "2_2": ("2_1", 1.0), "3_2": ("3_1", 1.0), "4_2": ("4_1", 1.0), "5_2": ("5_1", 1.0)}


def origin(el, pos, quat):
    xyz = np.array(pos.split(), float) if pos else np.zeros(3)
    q = np.array(quat.split(), float) if quat else np.array([1.0, 0, 0, 0])
    rpy = R.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
    ET.SubElement(el, "origin", xyz=" ".join(f"{v:.9g}" for v in xyz), rpy=" ".join(f"{v:.12g}" for v in rpy))


def visual(link, g, meshfile):
    v = ET.SubElement(link, "visual")
    origin(v, g.get("pos"), g.get("quat"))
    ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename=meshfile)


for side in ("right", "left"):
    src = ET.parse(DIR / side / "hand.xml").getroot()
    p = "hand2" if side == "right" else "hand1"
    meshes = {m.get("name"): m.get("file") for m in src.iter("mesh")}
    robot = ET.Element("robot", name=f"ruiyan_ryh2_{side}_egosteer")
    base = ET.SubElement(robot, "link", name=f"{p}_link_base")
    wb = src.find("worldbody")
    for g in wb.findall("geom"):
        if g.get("mesh") == f"{p}_link_base_visuals":
            visual(base, g, meshes[g.get("mesh")])

    def add_body(body, parent):
        name = body.get("name")
        link = ET.SubElement(robot, "link", name=name)
        for g in body.findall("geom"):
            if g.get("mesh"):
                visual(link, g, meshes[g.get("mesh")])
        jel = body.find("joint")
        jname = jel.get("name")
        assert jel.get("pos", "0 0 0").split() == ["0", "0", "0"], jname
        j = ET.SubElement(robot, "joint", name=jname, type="revolute")
        origin(j, body.get("pos"), body.get("quat"))
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=name)
        ET.SubElement(j, "axis", xyz=jel.get("axis"))
        lo, hi = jel.get("range").split()
        ET.SubElement(j, "limit", lower=lo, upper=hi, effort="1", velocity="2")
        key = jname.split("joint_link_")[1]
        if key in MIMIC:
            lead, k = MIMIC[key]
            ET.SubElement(j, "mimic", joint=f"{p}_joint_link_{lead}", multiplier=str(k), offset="0")
        for s in body.findall("site"):
            tl = ET.SubElement(robot, "link", name=s.get("name"))
            tj = ET.SubElement(robot, "joint", name=f"{s.get('name')}_joint", type="fixed")
            origin(tj, s.get("pos"), s.get("quat"))
            ET.SubElement(tj, "parent", link=name)
            ET.SubElement(tj, "child", link=tl.get("name"))
        for b in body.findall("body"):
            add_body(b, name)

    for b in wb.findall("body"):
        if b.get("mocap") != "true":
            add_body(b, base.get("name"))
    ET.indent(robot)
    out = DIR / side / f"ruiyan_ryh2_{side}.urdf"
    ET.ElementTree(robot).write(out, xml_declaration=True, encoding="utf-8")
    print(out)
