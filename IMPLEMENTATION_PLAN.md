# IMPLEMENTATION_PLAN.md — Engineering Roadmap

## Phase 0 — Repository skeleton

Create:

```text
unified_dex_dataset/
├── CLAUDE.md
├── README.md
├── config/
├── schema/
├── adapters/
├── hand_models/
├── kinematics/
├── preprocessing/
├── validation/
├── visualization/
├── clustering/
├── training/
└── tests/
```

Implement versioned dataclasses/schema.

---

## Phase 1 — Global manifest

Before downloading/converting everything, implement a manifest that can describe:

- source dataset;
- episode;
- hand model;
- cameras;
- text availability;
- modality availability;
- canonicalization status.

---

## Phase 2 — VITRA/XHand PoC

Implement:

```text
VITRATeleDataAdapter
XHandKinematicsAdapter
```

Tasks:

1. read RGB/video reference;
2. read timestamps;
3. read `state/right_hand_joint`;
4. load candidate XHand model;
5. map joint order;
6. compute FK;
7. produce 5 fingertips;
8. convert to palm frame;
9. normalize scale;
10. visualize.

Do not mass-process until manually validated.

---

## Phase 3 — RealDex/Shadow PoC

Implement:

```text
RealDexAdapter
ShadowKinematicsAdapter
```

Repeat same steps.

Then compare XHand vs Shadow in canonical geometry.

---

## Phase 4 — Representation validation

Build a small balanced set.

Run:

- nearest-neighbor search;
- PCA;
- simple KMeans;
- manual visual inspection.

Goal is not final labels.

Goal is to verify common geometry.

---

## Phase 5 — LEAP V2 Advanced / DexWild

Add exact 17D mapping and official kinematics.

---

## Phase 6 — Fourier FDH-6

Add official FDH-6 model.

Keep FDH-12 native-only until exact model is available.

---

## Phase 7 — Inspire family

Treat each variant separately.

Implement generic infrastructure first.

Do not merge variants without geometry validation.

Add:

- RoboMIND BFX;
- Humanoid Everyday H1/DFX candidate;
- HRDex DFTP;
- HRDex F1 when exact model exists.

---

## Phase 8 — AgiBot

Build native adapter.

Set FK unavailable until exact G1 dex-hand model is found.

---

## Phase 9 — Materialized unified feature store

After schema is stable, write:

- Parquet low-dimensional data;
- references to video;
- derived canonical geometry;
- provenance.

Avoid unnecessary video duplication.

---

## Phase 10 — Posture/grasp mining

Implement separate pipeline:

```text
sample candidate grasp moments
→ features
→ clustering
→ versioned labels
→ visual inspection
```

---

## Phase 11 — Stage-1 model

Integrate dataset loader with Cosmos/world-model training.

Support:

- video loss only;
- classification head;
- geometry head;
- combined.

---

## Phase 12 — Oracle downstream experiment

Before large end-to-end investment, test whether GT future-hand info helps action prediction/control.

---

# Recommended code interfaces

## DatasetAdapter

```python
class DatasetAdapter:
    def list_episodes(self): ...
    def get_metadata(self, episode_id): ...
    def get_timestamps(self, episode_id): ...
    def get_text(self, episode_id): ...
    def get_cameras(self, episode_id): ...
    def get_arm_state(self, episode_id): ...
    def get_hand_state(self, episode_id, side): ...
    def get_hand_action(self, episode_id, side): ...
    def get_object_state(self, episode_id): ...
    def get_tactile(self, episode_id): ...
```

## HandKinematicsAdapter

```python
class HandKinematicsAdapter:
    def raw_to_model_q(self, raw_q): ...
    def forward_kinematics(self, model_q): ...
    def canonical_fingertips(self, model_q, side): ...
    def hand_scale(self, model_q=None): ...
```

---

# Required tests

## Unit tests

- joint-order mapping;
- unit conversion;
- quaternion conventions;
- left/right mirror;
- palm frame orthonormality;
- scale normalization;
- missing-data masks.

## Visual tests

For every new hand:

- neutral;
- open;
- closed;
- thumb opposition;
- index movement;
- random real frames.

## Regression tests

Store a few known input states and expected fingertip outputs.

Changing URDF/mapping should trigger visible diffs/version bump.

---

# First acceptance criteria

The initial milestone is complete when:

1. VITRA XHand loads.
2. RealDex Shadow loads.
3. Native hand states are preserved.
4. FK returns valid five fingertips.
5. Palm-relative metric coordinates exist.
6. Normalized coordinates exist.
7. Visual validation passes.
8. Nearest neighbors across embodiments are qualitatively sensible.
9. Provenance/version metadata is complete.
10. No guessed model is silently labeled as exact.
