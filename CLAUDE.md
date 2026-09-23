# CLAUDE.md — Master Instructions for UnifiedDex Project

## 1. Project in one sentence

Build a unified dataset from **real robot demonstrations with five-finger dexterous hands**, convert heterogeneous native hand states into a common geometric representation, later derive grasp/posture classes, and use those labels to train a Cosmos/mimic-video-style world model to predict both the future visual state and the future hand configuration.

---

## 2. Scientific goal

We want a world model to learn more than generic future-video prediction.

The central hypothesis is:

> A world model used for dexterous manipulation should explicitly encode task-relevant future hand configuration / grasp information instead of leaving it implicit in distributed visual latents.

The Stage-1 model should eventually optimize roughly:

\[
L = L_{\text{video}} + \lambda L_{\text{future-hand}}
\]

where:

- `L_video` is the standard video/world-model objective.
- `L_future-hand` predicts a future hand posture / grasp target derived from robot proprioception.

Do **not** hard-code the final grasp taxonomy yet. First build the data infrastructure and a physically meaningful continuous hand representation.

---

## 3. Dataset philosophy

We are **not trying to choose one best dataset**.

We are building a mixture from useful subsets of several datasets.

A dataset is useful if it contributes some combination of:

- real-robot RGB;
- five-finger dexterous hand;
- frame-aligned hand/finger state;
- text task/instruction;
- multiple cameras;
- object state;
- tactile/contact.

The unified corpus must preserve what each source dataset contains without pretending missing modalities exist.

---

## 4. Strict primary corpus rules

Primary Stage-1 grasp-supervised data should be:

- real robot demonstrations;
- five-finger dexterous hand;
- frame-aligned hand/finger state;
- RGB video;
- preferably text instruction.

Simulation is not part of the primary corpus.

Do not include two-jaw/parallel-gripper-only datasets in the primary grasp-supervised corpus.

---

## 5. Current selected source datasets

Use the exact subsets defined in `DATASETS.md`.

High-level:

- Fourier ActionNet — Fourier five-finger hand subsets.
- RoboMIND — Tien Kung + Xsens only.
- AgiBot World — dexterous-hand subset only.
- Humanoid Everyday — H1 only.
- RealDex — Shadow Hand.
- DexWild — robot subset only.
- VITRA-TeleData — XHand real-robot trajectories.
- HRDexDB — Inspire subsets only.
- RoboTacDex — future candidate when officially released.

Exclude from the primary corpus:

- Galaxea Open-World: two-jaw gripper.
- DexScale: not used.
- MobileManiDataset-XHand: simulation.
- DexVerse: simulation.

---

## 6. Critical hand-unification decision

### Never create a fake common joint-angle vector.

These are different mechanical systems:

```text
Inspire     ~ 6 active DoF
XHand       12 DoF
LEAP V2 Adv 17 DoF
Shadow      ~20–22+ articulated values
Fourier     6 or 12 DoF
```

Do not:

- zero-pad one vector to another;
- truncate rich hands;
- assume `q[i]` means the same thing across embodiments;
- silently substitute a "similar" URDF.

Instead preserve:

```text
native_joint_positions
native_joint_actions
native_joint_names
native units
native provenance
```

and derive a common **geometric** representation using each hand's own kinematics.

---

## 7. Canonical hand representation v0.1

First version:

\[
H_t =
[p_{\text{thumb tip}},
 p_{\text{index tip}},
 p_{\text{middle tip}},
 p_{\text{ring tip}},
 p_{\text{pinky tip}}]
\]

Each point is XYZ relative to the hand's own palm frame:

\[
H_t \in \mathbb{R}^{5\times3}
\]

Flattened dimension: 15.

Store two versions:

1. `fingertips_palm_m` — metric coordinates in meters.
2. `fingertips_palm_norm` — normalized by hand/palm scale for cross-embodiment comparison.

Fixed order:

```text
0 thumb
1 index
2 middle
3 ring
4 pinky
```

See `KINEMATICS.md`.

---

## 8. What is the palm frame

Palm = rigid base of the hand to which fingers attach.

The palm frame is a coordinate system fixed to the hand.

We use fingertip positions **relative to the palm**, not world coordinates, so that the representation captures hand shape rather than:

- where the robot stands;
- where the arm is;
- global hand translation;
- global hand rotation.

The same pinch pose performed at different places in the workspace should map to approximately the same canonical hand geometry.

---

## 9. Forward kinematics

For each hand:

```text
dataset-native hand state
      ↓
dataset-specific mapping
      ↓
physical/model joint state
      ↓
URDF/MJCF/USD forward kinematics
      ↓
five fingertip positions
      ↓
canonical palm frame
      ↓
scale normalization
```

If the exact model is unavailable, import native data but leave canonical geometry unavailable.

Never invent FK.

See `HANDS.md` and `KINEMATICS.md`.

---

## 10. Provenance is mandatory

Every derived hand geometry must retain:

```text
dataset_id
source_episode_id
source timestamp/frame
embodiment_id
hand_model
hand_model_version
dataset mapping version
state_source
canonicalization_status
```

`state_source` should distinguish, when possible:

```text
measured_joint
measured_actuator
actuator_fk
command_fk
vision_estimated
unknown
```

Commands are not automatically equivalent to measured poses.

---

## 11. RGB rules

Do not immediately convert all source videos to one permanent resolution.

Keep original video/source references where practical.

Unified camera metadata should expose:

```text
camera_id
camera_role
fps
width
height
intrinsics
extrinsics
rgb path/reference
depth path/reference
timestamps
```

Training-time preprocessing can resize/crop for Cosmos.

Do not destructively normalize source data.

---

## 12. FPS and synchronization

Do not physically resample all datasets to the same FPS during ingestion.

Keep timestamps.

Training windows should be sampled by time:

```text
t
t + Δt
t + 2Δt
...
```

not by assuming all datasets share the same frame rate.

---

## 13. Multi-view handling

Treat one physical robot trajectory as a group:

```text
trajectory_group_id
```

with one or more camera/view episodes.

For example, a single HRDexDB grasp can produce multiple visual training views sharing the same:

- hand state;
- text;
- object state;
- timestamps.

Do not hard-code a fixed global `camera_0...camera_22` schema.

---

## 14. Text provenance

Do not mix native and generated annotations without labels.

Store:

```text
instruction_original
instruction_en
annotation_source
annotation_level
language
```

Possible `annotation_source`:

```text
dataset
human
template
vlm
none
```

Never present a VLM-generated caption as a native dataset annotation.

---

## 15. Grasp labels are a later derived layer

Do not permanently assign a single `grasp_class` during initial conversion.

First build canonical continuous geometry.

Later create versioned derived annotations:

```text
hand_posture_class_v1
grasp_embedding_v1
grasp_class_v1
...
```

The clustering method may change.

---

## 16. Posture is not always grasp

A closed hand may be:

- closed in free space;
- a preshape;
- entering contact;
- stable grasp;
- release.

Long term, distinguish:

```text
hand_posture_class
grasp_contact_phase
functional_grasp_class
```

Do not assume every frame is a grasp.

See `GRASP_LABELING.md`.

---

## 17. Stage-1 model interface

Conceptually:

```text
past RGB clip + language
        ↓
      Cosmos
        ↓
world/video prediction representation
        +
future hand prediction head
```

Targets may be:

- discrete future posture/grasp class;
- continuous future canonical geometry;
- both.

Potential loss:

\[
L =
L_{\text{video}}
+
\lambda_c L_{\text{class}}
+
\lambda_h L_{\text{hand-geometry}}
\]

Do not architect the dataset so that only classification is possible.

---

## 18. Important leakage concern

When training a future-hand classifier inside a video diffusion/flow model, ensure the auxiliary head cannot simply read the clean future frame/latent.

If using mimic-video-style noisy future latents, a clean experiment should support a mode where the grasp head sees:

- past visual context;
- language;
- predictive world representation;

but not ground-truth clean future hand appearance.

See `STAGE1_TRAINING.md`.

---

## 19. First engineering milestone

Do not start by downloading/converting every dataset.

First prove the hand canonicalization idea with two different hands:

- VITRA / XHand 12D.
- RealDex / Shadow Hand rich articulated state.

Pipeline:

```text
native q
→ FK
→ 5 fingertips
→ palm frame
→ scale-normalized 15D geometry
→ visualization
```

Only after manual validation add LEAP, Fourier, Inspire variants, etc.

---

## 20. Scientific correctness over completion

If information is missing:

- do not guess;
- do not invent joint mappings;
- do not silently use a similar hand model;
- do not mark a model validated because dimensions happen to match.

Prefer:

```text
native data imported
canonicalization_status = missing_exact_hand_model
```

over incorrect geometry.

---

## 21. Read these files before implementation

For dataset work:

- `DATASETS.md`
- `UNIFIED_DATASET.md`

For hand/FK work:

- `KINEMATICS.md`
- `HANDS.md`

For training:

- `STAGE1_TRAINING.md`

For clustering:

- `GRASP_LABELING.md`

For engineering milestones:

- `IMPLEMENTATION_PLAN.md`

For research evaluation:

- `EXPERIMENTS.md`
