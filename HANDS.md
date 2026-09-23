# HANDS.md — Kinematic Models / URDF Knowledge Base for UnifiedDexDataset

Last updated: 2026-09-15

This file is a practical knowledge base for Claude Code when implementing the hand-kinematics layer of `UnifiedDexDataset`.

The goal is to convert heterogeneous native dexterous-hand state vectors from multiple datasets into a common geometric representation:

\[
q_t^{native}
\rightarrow
FK
\rightarrow
[p_{thumb},p_{index},p_{middle},p_{ring},p_{pinky}]_{\text{relative to palm}}
\in \mathbb{R}^{5\times3}.
\]

The shared representation is **not** a shared joint-angle vector.

Different hands use different mechanics and different dimensionality:

- Inspire: ~6 active DoF / hand
- XHand: 12 DoF / hand
- LEAP Hand V2 Advanced: 17 motor DoF / hand
- Shadow Hand: ~20–22+ relevant articulated joint coordinates depending on representation
- Fourier: 6-DoF and 12-DoF variants

Never pad/truncate these vectors and treat them as a common space.

The common cross-embodiment representation must be derived through each hand's own kinematic model.

---

# 1. What files are actually required

For each physical hand model, the minimum sufficient package for our task is:

```text
hand_models/<hand_name>/
├── model.urdf / model.xacro / model.xml / model.mjcf / model.usd
├── mapping.yaml
└── README.md
```

A valid kinematic model must expose enough information to reconstruct:

- link/joint tree;
- joint origins;
- joint axes;
- fixed-joint transforms;
- link lengths / geometry offsets;
- joint limits;
- mimic/coupling relationships if present;
- palm / hand-base link;
- five fingertip links or transforms sufficient to define them.

Meshes such as:

```text
*.stl
*.dae
*.obj
```

are **not mathematically required for forward kinematics**, but they are highly desirable for visualization and validation.

If one actuator moves several physical joints and this coupling is not contained in the URDF/Xacro, add:

```text
coupling.yaml
```

or implement an explicit conversion function:

```python
actuator_to_physical_joint_positions(q)
```

---

# 2. Required per-dataset mapping

The kinematic model alone is not enough.

Each dataset adapter must additionally specify:

- raw vector dimensionality;
- raw value ordering;
- units (`rad`, `deg`, normalized 0–1, motor units, etc.);
- whether values are measured states or commands;
- dataset-coordinate → model-joint mapping;
- left/right handedness;
- any scale/offset calibration;
- any missing or passive joint reconstruction;
- palm link;
- fingertip links.

Recommended `mapping.yaml`:

```yaml
hand_model: xhand1
dataset: vitra_teledata
side: right

raw_dim: 12
raw_units: rad

state_source: measured_joint
state_confidence: high

raw_joint_order:
  - ...
  - ...

model_joint_order:
  - ...
  - ...

links:
  palm: ...
  wrist: ...
  thumb_tip: ...
  index_tip: ...
  middle_tip: ...
  ring_tip: ...
  pinky_tip: ...

canonicalization:
  enabled: true
  validation_status: validated
```

---

# 3. Status vocabulary

Use the following exact meanings.

## `exact_official`

The exact hand model is published by the manufacturer / official project and matches the dataset hardware.

FK can be enabled after checking dataset joint ordering.

## `official_family_needs_variant_validation`

An official model exists for the same product family, but the precise hardware revision used in the dataset is not yet proven to match.

Do not treat as final ground truth before validation.

## `third_party_candidate`

A good public URDF exists, but not from the dataset authors/manufacturer.

FK may be tested, but must be visually validated before bulk processing.

## `available_with_license_constraints`

A model exists but may not be freely redistributable.

Do not commit proprietary files into a public repository.

## `missing_exact_model`

Native data may be imported, but canonical FK must remain disabled.

## `dataset_not_public`

The hand model exists, but the actual dataset cannot yet be used.

---

# 4. Dataset → hand model summary

| Dataset / subset | Hand | Native state | Model status | FK status |
|---|---|---:|---|---|
| Fourier ActionNet, FDH-6 episodes | Fourier FDH-6 | 6D/hand | `exact_official` | ENABLE after mapping validation |
| Fourier ActionNet, FDH-12 episodes | Fourier FDH-12 | 12D/hand | `missing_exact_model` | DISABLE |
| RoboMIND, Tien Kung + Xsens | Inspire RH56BFX | 6D/hand | `official_family_needs_variant_validation` | DISABLE until validated |
| AgiBot World dex-hand subset | AgiBot G1 5-finger hand | 6D/hand | `missing_exact_model` | DISABLE |
| Humanoid Everyday H1 | Inspire RH56 / DFX family | 6D/hand | `official_family_needs_variant_validation` | DISABLE until validated |
| RealDex | Shadow Dexterous Hand | rich articulated vector | `exact_official` | ENABLE after RealDex mapping |
| DexWild robot subset | LEAP Hand V2 Advanced | 17D/hand | `exact_official` | ENABLE |
| VITRA-TeleData | RobotEra XHand1 | 12D/hand | `third_party_candidate` | ENABLE only after visual validation |
| HRDexDB Inspire DFTP | Inspire RH56DFTP | 6D/hand | `available_with_license_constraints` | ENABLE after authorized model obtained |
| HRDexDB Inspire F1 | Inspire RH56F1 | 6D/hand | `missing_exact_model` | DISABLE |
| RoboTacDex | BrainCo Revo2 Tactile | 6D/hand | `exact_official` | Hand model READY; dataset unavailable |

---

# 5. Fourier ActionNet — Fourier FDH-6

## Dataset

ActionNet / Fourier robot subset.

We only care about episodes recorded with the five-finger Fourier dexterous hand.

ActionNet may include both:

- 6-DoF Fourier hand;
- 12-DoF Fourier hand.

These must be treated as different hand models.

Dataset repository / format:

- https://github.com/FFTAI/fourier-lerobot
- https://action-net.org/

## Official robot model repository

Fourier publishes official robot/hand URDFs here:

- https://github.com/FFTAI/Wiki-GRx-Models

Important changelog:

- https://github.com/FFTAI/Wiki-GRx-Models/blob/master/CHANGELOG.md

The changelog explicitly records updates for `Fourier Hand 6dof` and updated GR1T1 / GR1T2 Fourier-hand URDFs.

## What to download

```bash
git clone https://github.com/FFTAI/Wiki-GRx-Models.git
```

Then inspect:

```bash
find Wiki-GRx-Models \
  \( -iname '*.urdf' -o -iname '*.xacro' \)
```

Search for files / robots containing:

```text
fourier_hand
GR1T1_fourier_hand
GR1T2_fourier_hand
```

## Required work

Identify:

```text
palm/base link
thumb tip/distal link
index tip/distal link
middle tip/distal link
ring tip/distal link
pinky tip/distal link
```

Also verify:

- ActionNet raw 6D state ordering;
- units;
- measured vs commanded values;
- left/right coordinate convention.

## Status

```yaml
hand_model: fourier_fdh6
model_status: exact_official
fk_enabled: true_after_mapping_validation
```

---

# 6. Fourier ActionNet — Fourier FDH-12

## Important

Do **not** reuse the FDH-6 URDF.

The 12-DoF product is a distinct kinematic configuration.

Fourier developer documentation distinguishes FDH-12 from FDH-6.

Useful documentation:

- https://support.fftai.com/en/modules/gr-endeffector

## Current problem

An exact public official FDH-12 URDF/Xacro was not confirmed during this investigation.

Therefore:

```text
12D native state
!=
6D URDF + extra values
```

Never force FDH-12 through the FDH-6 model.

## Required missing files

Ideal:

```text
FDH-12 left-hand URDF/Xacro
FDH-12 right-hand URDF/Xacro
meshes (optional but desirable)
```

At minimum:

```text
joint tree
joint axes
link lengths
joint limits
actuator/joint ordering
passive/mimic coupling
```

## Status

```yaml
hand_model: fourier_fdh12
model_status: missing_exact_model
fk_enabled: false
canonicalization_status: missing_kinematic_model
```

Native joint data should still be imported.

---

# 7. RoboMIND Tien Kung + Xsens — Inspire RH56BFX

## Dataset

- https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND

Only use:

```text
Tien Kung + Xsens
```

for the five-finger grasp-supervised corpus.

The relevant 6D hand representation is documented as:

```text
little
ring
middle
index
thumb bend
thumb rotation
```

per hand.

Use actual robot-side state where available, rather than only the master/teleoperator command.

## Exact hardware

The RoboMIND paper identifies the Tien Kung dexterous hands as Inspire RH56BFX.

Do **not** automatically equate:

```text
RH56BFX
RH56DFX
RH56DFTP
RH56F1
```

They belong to the same broad Inspire RH56 family but may differ in:

- phalanx dimensions;
- palm geometry;
- passive-joint coupling;
- fingertip offsets;
- sensor packaging.

## Public generic RH56 model

Useful open-source RH56 packages:

- https://github.com/renesas-rdk/inspire_rh56_hand_description
- https://github.com/renesas-rdk/inspire_rh56_hand_bringup

The bringup package states that complete left/right URDF configurations are included and depends on the description package for meshes/joint definitions.

Clone:

```bash
git clone https://github.com/renesas-rdk/inspire_rh56_hand_description.git
```

Optional control package:

```bash
git clone https://github.com/renesas-rdk/inspire_rh56_hand_bringup.git
```

This generic RH56 model may be useful as a **candidate** for geometry validation, but must not be marked exact RH56BFX without checking.

## Additional RoboMIND clue

RoboMIND simulation/config documentation exposes a larger physical-joint representation including passive joints.

This is evidence that:

\[
6D \text{ active hand state}
\rightarrow
\text{more physical finger joints}
\]

and that passive/coupled joint reconstruction matters.

## Validation required

Compare candidate RH56 model with RH56BFX using:

- neutral fingertip positions;
- palm width;
- finger link lengths;
- joint ranges;
- thumb geometry;
- active/passive coupling.

## Status

```yaml
hand_model: inspire_rh56bfx
model_status: official_family_needs_variant_validation
candidate_model: renesas_generic_rh56
fk_enabled: false
```

Do not bulk-process until validated.

---

# 8. Humanoid Everyday H1 — Inspire RH56 / RH56DFX family

## Dataset

- https://humanoideveryday.github.io/
- https://github.com/physical-superintelligence-lab/Humanoid-Everyday

Use only:

```text
H1
```

not the three-finger G1 subset.

The H1 Inspire hand uses 6 values per hand:

```text
pinky
ring
middle
index
thumb bend
thumb rotation
```

## Strong hardware clue: Unitree RH56DFX controller

Official Unitree controller:

- https://github.com/unitreerobotics/DFX_inspire_service

The repo explicitly identifies itself as:

```text
Unitree Robot RH56DFX Inspire Hand Controller
```

and documents bilateral ordering:

```text
Right:
pinky
ring
middle
index
thumb-bend
thumb-rotation

Left:
pinky
ring
middle
index
thumb-bend
thumb-rotation
```

This is highly consistent with Humanoid Everyday H1 data.

However, unless the Humanoid Everyday release explicitly confirms `RH56DFX`, keep the exact variant as not fully verified.

## Candidate kinematic model

Use the same generic RH56 model as a provisional candidate:

- https://github.com/renesas-rdk/inspire_rh56_hand_description

But validate against actual H1 hand geometry.

## Status

```yaml
hand_model: inspire_h1
hand_model_family: inspire_rh56
likely_variant: rh56dfx
model_status: official_family_needs_variant_validation
fk_enabled: false
```

---

# 9. RealDex — Shadow Dexterous Hand

## Dataset

- https://4dvlab.github.io/RealDex_page/
- https://github.com/4DVLab/RealDex
- https://arxiv.org/pdf/2402.13853

Hardware:

```text
UR10e + Shadow Dexterous Hand
```

This is one of the strongest datasets for hand geometry.

## Official Shadow model

Official Shadow Robot repository:

- https://github.com/shadow-robot/sr_common

Main Xacro:

- https://github.com/shadow-robot/sr_common/blob/noetic-devel/sr_description/robots/sr_hand.urdf.xacro

The Xacro supports:

```text
side = right / left
fingers = all
hand type/version
tip sensors
```

The Shadow organization describes `sr_common` as containing URDF models and messages.

## Download

```bash
git clone https://github.com/shadow-robot/sr_common.git
```

## What remains to solve

The major remaining problem is not the hand model; it is the mapping:

```text
RealDex native articulation vector
→
Shadow named joints
```

Need to determine exact RealDex ordering from:

- RealDex code;
- preprocessing scripts;
- dataset documentation.

Do not guess.

Typical Shadow joint naming contains finger-specific families such as:

```text
FF...   index / forefinger
MF...   middle
RF...   ring
LF...   little
TH...   thumb
WR...   wrist
```

Use the exact mapping from RealDex, not a generic assumed order.

## Fingertip links

Find exact right-hand tip links inside the Shadow description and map them to:

```text
thumb
index
middle
ring
pinky
```

## Status

```yaml
hand_model: shadow_hand
model_status: exact_official
fk_enabled: true_after_dataset_mapping
```

This should be one of the first proof-of-concept hands.

---

# 10. DexWild robot subset — LEAP Hand V2 Advanced

## Dataset

- https://dexwild.github.io/
- https://arxiv.org/pdf/2505.07813v2

Use only the robot subset.

Relevant hand:

```text
LEAP Hand V2 Advanced
```

Do not confuse this with the ordinary LEAP Hand / LEAP Hand V2.

## Official repository

- https://github.com/leap-hand/LEAP_Hand_V2_Adv_API

This repository explicitly describes:

```text
LEAP Hand v2 Advanced
17-DOF
```

and warns that it is not the standard LEAP Hand v2.

## Important official motor ordering

Direct 17D motor command order:

```text
Index:
  MCP side
  MCP forward
  curl

Middle:
  MCP side
  MCP forward
  curl

Ring:
  MCP side
  MCP forward
  curl

Pinky:
  MCP side
  MCP forward
  curl

Thumb:
  MCP side
  MCP forward
  curl

Palm:
  thumb palm articulation
  four-finger palm articulation
```

Total:

```text
3 × 5 + 2 = 17
```

The repo also documents tendon coupling and fingertip IK.

## Download

```bash
git clone https://github.com/leap-hand/LEAP_Hand_V2_Adv_API.git
```

Then inspect:

```bash
find LEAP_Hand_V2_Adv_API \
  \( -iname '*.urdf' -o \
     -iname '*.xacro' -o \
     -iname '*.xml' -o \
     -iname '*.mjcf' \)
```

## Required work

Identify:

```text
palm/base link
thumb tip
index tip
middle tip
ring tip
pinky tip
```

and verify DexWild stored 17D order matches official LEAP motor order.

## Status

```yaml
hand_model: leap_v2_advanced
model_status: exact_official
fk_enabled: true_after_mapping_validation
```

This is a high-priority FK backend.

---

# 11. VITRA-TeleData — RobotEra XHand1

## Dataset

- https://github.com/microsoft/VITRA/
- https://huggingface.co/datasets/microsoft/VITRA-TeleData

Use:

```text
/state/right_hand_joint
```

as the preferred ground-truth hand state.

Do not use only:

```text
/action/right_hand_joint
```

when actual measured state is available.

The released VITRA robot data uses a five-finger 12-DoF XHand.

## Public candidate URDF

A detailed public XHand URDF is available in Meta/Facebook Research SPIDER:

Repository:

- https://github.com/facebookresearch/spider

Candidate XHand model path:

- https://github.com/facebookresearch/spider/blob/main/spider/assets/robots/xhand/xhand_right.urdf

This URDF contains:

- palm / hand base;
- thumb;
- index;
- middle;
- ring;
- pinky chains;
- joint limits;
- fixed distal links;
- mesh references.

## Important limitation

This is **not** the Microsoft VITRA repository and is not necessarily the manufacturer's exact release used in VITRA.

Treat it as:

```text
third_party_candidate
```

until validated.

## Validation protocol

Before bulk processing:

1. Load 20–50 VITRA frames.
2. Read `state/right_hand_joint`.
3. Map the 12 values into candidate XHand URDF joints.
4. Render the resulting hand.
5. Compare to the RGB frame.
6. Check thumb shape, index flexion, finger ordering, neutral/open pose, and fully closed pose.
7. Only then mark the model validated.

## Licensing note

The ManipTrans-HRDexDB repository states that some dexterous-hand models, including XHand, cannot be redistributed by them because of licensing restrictions.

Reference:

- https://github.com/hahahataeyun/ManipTrans-HRDexDB

Therefore do not blindly copy an unknown proprietary XHand model into a public repository.

## Status

```yaml
hand_model: xhand1
model_status: third_party_candidate
candidate_model_source: facebookresearch/spider
fk_enabled: true_after_visual_validation
```

VITRA/XHand should be one of the first proof-of-concept adapters.

---

# 12. HRDexDB — Inspire RH56DFTP

## Dataset

- https://huggingface.co/datasets/HRDexDB/HRDexDB
- https://github.com/hahahataeyun/ManipTrans-HRDexDB

Use only five-finger Inspire subsets.

For DFTP:

```text
inspire_dftp
```

## Model availability

A vendor/distributor download page advertises RH56DFTP resources including:

```text
FTP_URDF_LEFT
FTP_URDF_RIGHT
3D Files
ROS2
```

Reference:

- https://researchrobots.eu/rh56dftphands.html

## Licensing warning

The ManipTrans-HRDexDB repository explicitly states that, due to licensing restrictions, some dexterous-hand URDF files such as XHand and Inspire FTP cannot be provided.

Repository:

- https://github.com/hahahataeyun/ManipTrans-HRDexDB

Therefore:

- obtain the model through an authorized source;
- do not commit proprietary files to a public repo;
- keep only loader/config instructions in our repository.

Recommended local structure:

```text
external_hand_models/
└── inspire_rh56dftp/
    ├── README.md
    └── authorized_model_goes_here/
```

## Status

```yaml
hand_model: inspire_rh56dftp
model_status: available_with_license_constraints
fk_enabled: true_after_authorized_model_obtained
redistribute_model: false
```

---

# 13. HRDexDB — Inspire RH56F1

## Dataset subset

```text
inspire_f1
```

This must be treated separately from:

```text
inspire_dftp
```

until proven kinematically identical.

## Current model status

An exact public RH56F1 URDF was not confirmed during this investigation.

Do not substitute:

```text
RH56DFTP
RH56DFX
generic RH56
```

without validation.

## Allowed behavior

Import:

```text
RGB
native hand state
hand actions
timestamps
object metadata
camera metadata
```

but set:

```yaml
canonicalization_status: missing_exact_hand_model
```

## Status

```yaml
hand_model: inspire_rh56f1
model_status: missing_exact_model
fk_enabled: false
```

---

# 14. AgiBot World — G1 five-finger dexterous hand

## Dataset

- https://github.com/OpenDriveLab/AgiBot-World
- https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

Use only dexterous-hand episodes, not parallel-gripper episodes.

## Critical blocker

There is an official/open AgiBot-World GitHub issue asking where to obtain the exact five-finger G1 dexterous-hand URDF/USD:

- https://github.com/OpenDriveLab/AgiBot-World/issues/155

The issue specifically asks for:

```text
5-finger dexterous hand model (URDF or USD) for AgiBot G1
```

This means we currently should **not** assume that a generic Inspire or other 6D hand model is correct.

## Allowed behavior

Import the native dataset fields:

```text
RGB
text
hand state
EEF pose
timestamps
camera metadata
```

but do not compute canonical fingertips from a guessed hand model.

## Status

```yaml
hand_model: agibot_g1_dexhand
model_status: missing_exact_model
fk_enabled: false
canonicalization_status: missing_exact_hand_model
```

---

# 15. RoboTacDex — BrainCo Revo2 Tactile

## Dataset

RoboTacDex is scientifically relevant but the dataset is not yet publicly downloadable.

The hand model itself is well supported.

## Exact hand

```text
BrainCo Revo2 Tactile
```

Five fingers, 6 active control coordinates per hand.

## Official BrainCo repositories

Organization:

- https://github.com/BrainCoTech

Revo2 description package:

- https://github.com/BrainCoTech/revo2_description

BrainCo states that the package contains URDF files and meshes of Revo2 dexterous hands.

## Revo2 active joint semantics

The six active coordinates are approximately:

```text
thumb flexion
thumb abduction
index flexion
middle flexion
ring flexion
pinky flexion
```

Use exact names from the BrainCo model/driver rather than hardcoding informal labels.

## Download

```bash
git clone https://github.com/BrainCoTech/revo2_description.git
```

## Status

```yaml
hand_model: brainco_revo2
model_status: exact_official
fk_enabled: true
dataset_available: false
```

When RoboTacDex becomes public, this should be straightforward to add.

---

# 16. Inspire hands must NOT be collapsed into one model prematurely

Current dataset set contains several Inspire variants:

```text
RoboMIND:
  RH56BFX

Humanoid Everyday H1:
  Inspire RH56 family
  likely RH56DFX, needs confirmation

HRDexDB:
  RH56DFTP
  RH56F1
```

Treat all of these as separate model IDs initially:

```text
inspire_rh56bfx
inspire_h1_unknown_or_dfx
inspire_rh56dftp
inspire_rh56f1
```

Only merge them later if actual geometry comparison proves that their relevant kinematic chains are equivalent.

Do not merge solely because all expose the same six active coordinates:

```text
pinky
ring
middle
index
thumb bend
thumb rotation
```

Equal control dimensionality does not imply equal geometry.

---

# 17. Recommended local external-model layout

Use:

```text
external_hand_models/
│
├── fourier_fdh6/
│   └── Wiki-GRx-Models/
│
├── fourier_fdh12/
│   └── MISSING.md
│
├── inspire_rh56_generic/
│   └── inspire_rh56_hand_description/
│
├── inspire_rh56bfx/
│   └── MISSING_EXACT_MODEL.md
│
├── inspire_h1/
│   └── NEED_VARIANT_CONFIRMATION.md
│
├── inspire_rh56dftp/
│   ├── README.md
│   └── authorized_model_goes_here/
│
├── inspire_rh56f1/
│   └── MISSING.md
│
├── shadow/
│   └── sr_common/
│
├── leap_v2_advanced/
│   └── LEAP_Hand_V2_Adv_API/
│
├── xhand1/
│   └── xhand_candidate/
│
├── agibot_g1_dexhand/
│   └── MISSING.md
│
└── brainco_revo2/
    └── revo2_description/
```

Do not commit third-party/proprietary assets if the license does not allow redistribution.

---

# 18. Repositories that can be cloned now

```bash
mkdir -p external_hand_models
cd external_hand_models
```

## Fourier

```bash
git clone \
  https://github.com/FFTAI/Wiki-GRx-Models.git \
  fourier_models
```

## Generic Inspire RH56

```bash
git clone \
  https://github.com/renesas-rdk/inspire_rh56_hand_description.git \
  inspire_rh56_generic
```

Optional bringup/control reference:

```bash
git clone \
  https://github.com/renesas-rdk/inspire_rh56_hand_bringup.git \
  inspire_rh56_bringup
```

## Unitree RH56DFX control reference

```bash
git clone \
  https://github.com/unitreerobotics/DFX_inspire_service.git \
  unitree_rh56dfx_service
```

## Shadow Hand

```bash
git clone \
  https://github.com/shadow-robot/sr_common.git \
  shadow_sr_common
```

## LEAP Hand V2 Advanced

```bash
git clone \
  https://github.com/leap-hand/LEAP_Hand_V2_Adv_API.git \
  leap_v2_advanced
```

## BrainCo Revo2

```bash
git clone \
  https://github.com/BrainCoTech/revo2_description.git \
  brainco_revo2
```

## XHand candidate model

Clone SPIDER or copy only the relevant model according to license:

```bash
git clone \
  https://github.com/facebookresearch/spider.git \
  spider
```

Candidate URDF:

```text
spider/assets/robots/xhand/xhand_right.urdf
```

---

# 19. Suggested central configuration

Create:

```text
config/hands.yaml
```

with entries like:

```yaml
hands:

  fourier_fdh6:
    datasets:
      - fourier_actionnet
    model_status: exact_official
    model_source: https://github.com/FFTAI/Wiki-GRx-Models
    fk_enabled: true
    validation_required: true

  fourier_fdh12:
    datasets:
      - fourier_actionnet
    model_status: missing_exact_model
    fk_enabled: false

  inspire_rh56bfx:
    datasets:
      - robomind_tienkung_xsens
    model_status: official_family_needs_variant_validation
    candidate_model_source: https://github.com/renesas-rdk/inspire_rh56_hand_description
    fk_enabled: false

  inspire_h1:
    datasets:
      - humanoid_everyday_h1
    model_status: official_family_needs_variant_validation
    likely_variant: rh56dfx
    control_reference: https://github.com/unitreerobotics/DFX_inspire_service
    candidate_model_source: https://github.com/renesas-rdk/inspire_rh56_hand_description
    fk_enabled: false

  shadow_hand:
    datasets:
      - realdex
    model_status: exact_official
    model_source: https://github.com/shadow-robot/sr_common
    fk_enabled: true
    validation_required: true

  leap_v2_advanced:
    datasets:
      - dexwild_robot
    model_status: exact_official
    model_source: https://github.com/leap-hand/LEAP_Hand_V2_Adv_API
    fk_enabled: true
    validation_required: true

  xhand1:
    datasets:
      - vitra_teledata
    model_status: third_party_candidate
    candidate_model_source: https://github.com/facebookresearch/spider
    candidate_urdf: spider/assets/robots/xhand/xhand_right.urdf
    fk_enabled: false
    enable_after_visual_validation: true

  inspire_rh56dftp:
    datasets:
      - hrdexdb_inspire_dftp
    model_status: available_with_license_constraints
    model_info: https://researchrobots.eu/rh56dftphands.html
    fk_enabled: false
    enable_after_authorized_model_obtained: true
    redistribute_model: false

  inspire_rh56f1:
    datasets:
      - hrdexdb_inspire_f1
    model_status: missing_exact_model
    fk_enabled: false

  agibot_g1_dexhand:
    datasets:
      - agibot_world_dexhand
    model_status: missing_exact_model
    issue: https://github.com/OpenDriveLab/AgiBot-World/issues/155
    fk_enabled: false

  brainco_revo2:
    datasets:
      - robotacdex
    model_status: exact_official
    model_source: https://github.com/BrainCoTech/revo2_description
    fk_enabled: true
    dataset_available: false
```

---

# 20. FK-ready priority list

## Tier A — start implementation now

### 1. Shadow Hand / RealDex

Why:

- official model;
- rich hand state;
- strong geometric supervision.

Remaining work:

```text
RealDex vector → Shadow named joints
```

### 2. LEAP V2 Advanced / DexWild

Why:

- official exact model;
- explicit 17D motor order;
- rich five-finger configuration.

Remaining work:

```text
DexWild 17D order → official LEAP 17D order
```

### 3. Fourier FDH-6 / ActionNet

Why:

- official Fourier model available.

Remaining work:

```text
ActionNet 6D order → FDH-6 joints
```

---

# 21. Tier B — likely usable after validation

## XHand1 / VITRA

Public candidate URDF is strong, but validate against real images before trusting it.

## Inspire H1 / Humanoid Everyday

The Unitree RH56DFX software strongly suggests the expected control convention, but exact hardware-model matching still needs confirmation.

## RoboMIND RH56BFX

Generic RH56 model is available but exact BFX geometry needs validation.

---

# 22. Tier C — blocked

## Fourier FDH-12

Exact public model not yet confirmed.

## AgiBot G1 dexterous hand

Official exact public model not found; official issue remains relevant.

## HRDexDB RH56F1

Exact F1 model not confirmed.

---

# 23. Tier D — model ready, dataset blocked

## BrainCo Revo2 / RoboTacDex

Official hand description exists, but RoboTacDex itself is not yet publicly released.

---

# 24. Validation checklist for every new hand

Never set:

```text
validation_status = validated
```

without these checks.

## A. Joint order

For each input coordinate:

1. move only that coordinate;
2. render FK;
3. verify the expected finger moves;
4. confirm direction/sign.

## B. Units

Check:

```text
radians?
degrees?
0–1?
0–1000?
manufacturer motor units?
```

## C. Neutral/open hand

Compare FK geometry with real RGB or official hand image.

## D. Closed hand

Check:

- which fingers flex;
- whether thumb opposition is correct;
- whether passive joints bend correctly.

## E. Fingertip order

Fixed universal order must be:

```text
0 thumb
1 index
2 middle
3 ring
4 pinky
```

## F. Palm link

Confirm that the selected palm/base is the rigid body to which fingers attach.

## G. Left/right

Mirror handling must be tested explicitly.

## H. Real image overlay

For datasets with known camera calibration, optionally project FK points into RGB.

Even a rough overlay is a powerful sanity check.

---

# 25. Canonical geometry output expected from every validated model

After mapping and FK, every hand adapter must output:

```python
{
    "hand_model": "...",

    "native_joint_positions": ...,
    "native_joint_names": ...,
    "native_joint_units": ...,

    "state_source": ...,
    "state_confidence": ...,

    "palm_link": "...",

    "fingertips_palm_m": [
        [x, y, z],  # thumb
        [x, y, z],  # index
        [x, y, z],  # middle
        [x, y, z],  # ring
        [x, y, z],  # pinky
    ],

    "fingertips_palm_norm": [...],

    "fingertip_valid_mask": [1, 1, 1, 1, 1],

    "canonicalization_status": "ok",

    "kinematic_model_version": "...",
    "mapping_version": "...",
}
```

---

# 26. Important rule for Claude Code

If the exact model cannot be proven correct:

**do not invent geometry.**

Preferred result:

```text
native state imported correctly
canonical FK = unavailable
```

over:

```text
fake fingertips produced with a similar hand
```

This project depends on scientifically defensible future-grasp labels.

Incorrect FK would silently contaminate all later clustering and Cosmos supervision.

---

# 27. Recommended first implementation sequence

Implement in this order:

```text
1. VITRA / XHand candidate model
2. RealDex / Shadow official model
3. compare canonical geometry
4. DexWild / LEAP V2 Advanced
5. Fourier FDH-6
6. Inspire family datasets
7. blocked models when exact assets become available
```

For the very first proof of concept, VITRA + RealDex are useful because they have very different native dimensionality:

```text
XHand 12D
Shadow ~22D
```

but should produce the same final:

```text
5 fingertips × XYZ
```

representation.

---

# 28. Source index

## Fourier

ActionNet:

- https://action-net.org/
- https://github.com/FFTAI/fourier-lerobot

Official Fourier models:

- https://github.com/FFTAI/Wiki-GRx-Models
- https://github.com/FFTAI/Wiki-GRx-Models/blob/master/CHANGELOG.md

Fourier end-effector docs:

- https://support.fftai.com/en/modules/gr-endeffector

## Inspire / Unitree

Generic RH56 description:

- https://github.com/renesas-rdk/inspire_rh56_hand_description

Generic RH56 bringup:

- https://github.com/renesas-rdk/inspire_rh56_hand_bringup

Unitree RH56DFX controller:

- https://github.com/unitreerobotics/DFX_inspire_service

HRDexDB / ManipTrans:

- https://github.com/hahahataeyun/ManipTrans-HRDexDB

RH56DFTP product/download page:

- https://researchrobots.eu/rh56dftphands.html

## Shadow

Official Shadow models:

- https://github.com/shadow-robot/sr_common

Main hand Xacro:

- https://github.com/shadow-robot/sr_common/blob/noetic-devel/sr_description/robots/sr_hand.urdf.xacro

## LEAP

LEAP Hand V2 Advanced:

- https://github.com/leap-hand/LEAP_Hand_V2_Adv_API

## XHand

VITRA:

- https://github.com/microsoft/VITRA/
- https://huggingface.co/datasets/microsoft/VITRA-TeleData

SPIDER candidate XHand model:

- https://github.com/facebookresearch/spider
- https://github.com/facebookresearch/spider/blob/main/spider/assets/robots/xhand/xhand_right.urdf

## AgiBot

AgiBot World:

- https://github.com/OpenDriveLab/AgiBot-World
- https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

Open exact-model issue:

- https://github.com/OpenDriveLab/AgiBot-World/issues/155

## BrainCo Revo2

BrainCo organization:

- https://github.com/BrainCoTech

Revo2 model:

- https://github.com/BrainCoTech/revo2_description

## Dataset sources

RoboMIND:

- https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND

Humanoid Everyday:

- https://humanoideveryday.github.io/
- https://github.com/physical-superintelligence-lab/Humanoid-Everyday

RealDex:

- https://4dvlab.github.io/RealDex_page/
- https://github.com/4DVLab/RealDex

DexWild:

- https://dexwild.github.io/

HRDexDB:

- https://huggingface.co/datasets/HRDexDB/HRDexDB

---

# 29. Current actionable conclusion

The project does **not** need to wait for every missing URDF.

Start the unified kinematics infrastructure with hands whose models are already available.

Immediately usable / near-usable:

```text
Shadow Hand
LEAP Hand V2 Advanced
Fourier FDH-6
XHand1 candidate (after validation)
```

Inspire family requires careful variant management.

Currently blocked exact models:

```text
Fourier FDH-12
AgiBot G1 5-finger dexterous hand
Inspire RH56F1
```

RoboTacDex:

```text
Revo2 model ready
dataset not yet public
```

Any hand without validated FK must remain in the dataset with:

```text
native_joint_positions = valid
canonical fingertips = unavailable
canonicalization_status = explicit failure reason
```

Never substitute a visually similar / same-DoF hand model silently.

---

# 30. Update 2026-09-23 (implementation findings; supersedes statuses in §4 where they differ)

| Dataset / subset | Hand | Model used | Model status | Mapping source | FK status |
|---|---|---|---|---|---|
| VITRA-TeleData | XHand1 (right) | SPIDER `xhand_right.urdf` | `third_party_candidate` | VITRA README + `XHAND_HUMAN_MAPPING` | ENABLED, visually checked |
| RealDex | Shadow Hand E (right) | RealDex `data_preprocess/assets/bimanual_srhand_ur.urdf` | `dataset_author_model` | RealDex `export_final_data` (22D, WRJ dropped) | ENABLED, visually checked |
| Humanoid Everyday H1 | Inspire DFX (L+R) | Unitree `xr_teleoperate/assets/inspire_hand` (mimic-coupled) | `integrator_official` | DFX_inspire_service order + xr_teleoperate normalize() | ENABLED, side order verified on video |
| HRDexDB inspire_dftp | Inspire RH56DFTP | HRDexDB `assets/robots/xarm_inspire_DFTP.urdf` (ships with the dataset) | `dataset_author_model` | snuvclab/HRDexDB `inspire_action_to_qpos_dof6` | ENABLED, visually checked; thumb yaw exceeds URDF limit in 16% samples |
| HRDexDB inspire_f1 | Inspire RH56F1 | HRDexDB `assets/robots/xarm_inspire_f1_right.urdf` | `dataset_author_model` | snuvclab/HRDexDB `inspire_f1_action_to_qpos_dof6` | ENABLED, visually checked |
| DexWild robot | LEAP V2 Advanced (right) | official `leap_v2_right/robot.urdf` | `exact_official` | dexwild_ros2 `leapv2_node` (17D **command**) | ENABLED as `command_fk`, UNVALIDATED assumptions |

Key corrections to earlier sections:
- §12/§13: HRDexDB itself publishes URDFs for both Inspire DFTP and F1, plus the raw->joint conversion code.
- §8: the Unitree-published Inspire URDF (with mimic coupling) + normalization is the model used by the H1 teleop stack.
- §10: DexWild stores the LEAP **command** stream only (`/leapv2_node/cmd_raw_leap_r`), no measured hand state.
- LEAP URDF root is a CAD origin 13 cm off the palm; wrist point is taken from the palm mesh (see config/hands.yaml).
- Shadow pinky base must be LFJ4 (knuckle), not LFJ5 (metacarpal), otherwise the palm X axis tilts ~30°.

# 31. Update 2026-09-23 (gated datasets inspected)

- **RoboMIND Tien Kung = Inspire RH56DFX, not BFX.** RoboMIND paper (arXiv:2412.13877): "Tien Kung utilizes two
  Inspire-Robots RH56DFX dexterous hands"; "BFX" does not occur in the paper. Mapped with the Unitree DFX model +
  normalization (same hardware as Humanoid Everyday H1); visually checked on plug_extract_from.
  Caveats: `puppet/*` is bit-identical to `master/*` (no separate measured state → `state_source = unknown`);
  no timestamps / recording rate in the release (`t_s = NaN`, frame index only). Gello subsets: 1D closure only.
- **Fourier ActionNet**: 312 tars GR1 + 6-DoF hand (12D), 8 tars GR2 + 12-DoF hand (24D), 2 tars GR2 + 6-DoF hand.
  Hand values are DexHand SDK *motor position params (0..~12), not joint angles* (fourier_dhx `get_angle` docstring).
  FDH-6 URDF exists but no public motor→joint transmission → `missing_joint_mapping`. FDH-12 → `missing_exact_model`.
- **AgiBot World**: dex-hand episodes found by probing proprio tars (effector/position 12D, rad): Alpha task 475 only;
  Beta 18 tasks (475, 536, 549, 554, 577, 578, 595, 608, 620, 622, 660, 679, 705, 710, 730, 731, 749, 753).
  Hand model still not public → native only. 6th value per hand stays in [-1.08, -0.35].
