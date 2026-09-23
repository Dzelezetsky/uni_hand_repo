# DATASETS.md — Selected Dataset Inventory

## Goal

This file defines which datasets/subsets belong to the current real-robot five-finger Stage-1 corpus.

We are not selecting one best dataset. We are combining useful subsets.

---

## 1. Fourier ActionNet — USE SELECTED SUBSETS

Sources:

- https://action-net.org/
- https://github.com/FFTAI/fourier-lerobot

Use:

- Fourier robots;
- five-finger Fourier dexterous hands;
- separate 6-DoF and 12-DoF hand variants.

Useful modalities:

- RGB / depth depending on release;
- robot state/action;
- hand state/action;
- text prompt exists in dataset metadata in current documentation;
- wrist/end-effector-related robot pose fields.

Important:

- do not mix FDH-6 and FDH-12 as one kinematic model;
- exact FDH-12 public model still needs confirmation.

---

## 2. RoboMIND — USE Tien Kung + Xsens ONLY

Source:

- https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND

Use:

```text
Tien Kung + Xsens
```

Do not rely on Gello subset for rich grasp labels if its hand information is effectively low-dimensional closure.

Hand:

- dual Inspire hands;
- 6 active values/hand;
- documented semantics:

```text
little
ring
middle
index
thumb bend
thumb rotation
```

Useful:

- RGB-D;
- robot state;
- hand state;
- language annotations / task text;
- long-horizon and precision tasks.

Need:

- exact Inspire hardware variant validation;
- exact state-to-model mapping.

---

## 3. AgiBot World — USE DEXTEROUS-HAND SUBSET ONLY

Sources:

- https://github.com/OpenDriveLab/AgiBot-World
- https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

Use only episodes/tasks with five-finger dexterous hand.

Do not include gripper episodes in grasp-supervised corpus.

Useful:

- multiple cameras;
- text/subtask annotations;
- 6D/hand dexterous state;
- arm/end-effector information.

Open questions:

- Alpha vs Beta dex-hand coverage/volume;
- exact hand coordinate semantics;
- exact G1 hand kinematic model;
- official exact URDF/USD currently unresolved.

Import native state even when FK is unavailable.

---

## 4. Galaxea Open-World — EXCLUDE FROM PRIMARY CORPUS

Sources:

- https://arxiv.org/pdf/2509.00576v1
- https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset

Reason:

- parallel/two-jaw gripper rather than five-finger dexterous hand.

May be useful for generic world-model pretraining in another project, but not for the current strict five-finger grasp-supervised Stage 1.

---

## 5. Humanoid Everyday — USE H1 ONLY

Sources:

- https://humanoideveryday.github.io/
- https://github.com/physical-superintelligence-lab/Humanoid-Everyday
- https://arxiv.org/pdf/2510.08807

Use:

```text
H1
```

Exclude G1 three-finger subset from strict five-finger corpus.

Hand:

- Inspire 6DoF;
- per-hand values:

```text
pinky
ring
middle
index
thumb bend
thumb rotation
```

Useful:

- RGB-D;
- hand/arm state;
- task annotations;
- diverse manipulation categories.

Need:

- exact Inspire hardware variant confirmation.

---

## 6. RealDex — USE

Sources:

- https://4dvlab.github.io/RealDex_page/
- https://github.com/4DVLab/RealDex
- https://arxiv.org/pdf/2402.13853

Hand:

- Shadow Dexterous Hand;
- rich articulated hand representation.

Useful:

- multi-view RGB-D;
- hand articulation;
- global hand pose;
- object pose;
- object meshes;
- real grasp trajectories.

Weakness:

- essentially one broad task family: grasp;
- no strong native natural-language instruction.

Still extremely useful for:

- hand geometry;
- grasp/posture-space discovery;
- validation of clustering.

Generated language, if added later, must be labeled as generated.

---

## 7. DexWild — USE ROBOT SUBSET ONLY

Sources:

- https://dexwild.github.io/
- https://arxiv.org/pdf/2505.07813v2

Use robot demonstrations only.

Hand:

- LEAP Hand V2 Advanced;
- 17D common joint-angle space.

Useful:

- exact robot hand state;
- hand-mounted cameras;
- scene/ZED view;
- several functional manipulation tasks.

Native rich language is not assumed.

Keep annotation provenance explicit.

---

## 8. VITRA-TeleData — USE

Sources:

- https://github.com/microsoft/VITRA/
- https://huggingface.co/datasets/microsoft/VITRA-TeleData

Robot:

- Realman arm;
- five-finger XHand.

Hand state:

```text
/state/right_hand_joint
```

12D per frame.

Useful:

- RGB;
- exact hand state;
- hand action;
- hand-mount pose;
- text instruction;
- functional grasping tasks.

This is a high-priority PoC dataset.

---

## 9. HRDexDB — USE INSPIRE SUBSETS ONLY

Sources:

- https://arxiv.org/pdf/2604.14944v2
- https://huggingface.co/datasets/HRDexDB/HRDexDB
- https://github.com/hahahataeyun/ManipTrans-HRDexDB

Use:

```text
inspire_dftp
inspire_f1
```

Exclude Allegro under strict five-finger criterion.

Useful:

- hand state/action;
- many synchronized cameras;
- object pose;
- tactile for relevant variants;
- grasp success;
- human↔robot paired structure.

No strong native language instructions assumed.

Important:

- DFTP and F1 must not be treated as identical hand models without proof.

---

## 10. DexScale — EXCLUDE

Not part of the current corpus.

---

## 11. MobileManiDataset-XHand — EXCLUDE FROM PRIMARY CORPUS

Reason:

- simulation.

Can be reconsidered later as synthetic augmentation.

---

## 12. DexVerse — EXCLUDE FROM PRIMARY CORPUS

Reason:

- simulation.

Interesting for later simulation/generalization studies, but not the current real-data Stage-1 corpus.

---

## 13. RoboTacDex — FUTURE CANDIDATE

Source:

- https://roboskin.ai/research/robotacdex-humanoid-visual-tactile-action-dataset-2026

Conceptually excellent:

- real Unitree G1;
- five-finger BrainCo Revo2;
- 6 DoF/hand;
- 4 RGB-D cameras;
- tactile;
- language annotations.

Current blocker:

- official dataset not publicly downloadable as of the latest project check.

Hand model can be prepared now, dataset adapter later.

---

# Dataset-adapter principle

Each adapter should return the same logical schema while preserving native fields.

Adapters must never invent:

- missing text;
- missing depth;
- missing tactile;
- missing hand geometry;
- missing joint values.

Use masks/status fields.
