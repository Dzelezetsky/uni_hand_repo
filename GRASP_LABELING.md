# GRASP_LABELING.md — Derived posture / grasp layer

Grasp labels are a later, versioned layer on top of the canonical geometry (CLAUDE.md §15–16). This file records
decisions for that layer.

## Decision 2026-09-23: thumb embodiment gap

### Observation

In the canonical space (`fingertips_palm_norm`) the four fingers of different hands are comparable, the thumb is not.
Measured on the sample corpus (`scripts/analysis/embodiment_gap.py`, `unified/validation/embodiment_gap.txt`;
mixing ratio ~1 = hands indistinguishable, >>1 = the feature encodes the hand):

| feature set | median mixing ratio |
|---|---|
| A raw 15D fingertips | 2.24 |
| B 4 fingers only (drop thumb) | 1.90 |
| **C 10 pairwise fingertip distances** | **1.21** |
| E per-hand URDF-reach normalization | 1.14 (semantically unsafe, see below) |

Cause: this is physics, not a bug. Thumb bases sit at similar places (x 19–34 mm, y −55…−86 mm from the knuckle
centroid), but the reachable thumb-tip region differs strongly: XHand / Fourier thumbs sweep a large 3D volume, the
Inspire DFX thumb (2 DoF, one of them coupled) moves on a thin arc (y range 0.36 palm widths vs 2.47 for XHand).
All hands can reach a thumb–index pinch (min distance 0.19–0.33 palm widths ≈ 12–20 mm).

### What this does NOT break

- The stored geometry (`fingertips_palm_m/_norm`) is physically correct and stays the ground truth.
- Stage-1 continuous future-hand regression: the target is still valid per embodiment; give the head the
  embodiment (`hand_model_id`) or predict in per-hand-consistent coordinates. The gap only matters when we want ONE
  shared discrete vocabulary across hands.

### What it WOULD break if ignored

Clustering raw 15D coordinates produces clusters that are mostly "which robot hand" instead of "which posture".

### Rules for `hand_posture_class_v1`

1. Clustering features = embodiment-invariant relational features: the 10 pairwise fingertip distances in palm
   widths (thumb–index, thumb–middle, … are functional: pinch = small thumb–index distance on any hand).
   Optional extras must be justified by a mixing-ratio + semantics check.
2. Do NOT use per-hand normalization to the hand's own range (URDF reach or data statistics) as the primary feature:
   it mixes hands numerically by mapping "50 % of Inspire's small thumb range" onto "50 % of XHand's large range",
   which are different physical postures.
3. Sample candidate grasp moments balanced per embodiment (and per dataset) before clustering; otherwise the
   open-hand-heavy datasets (Humanoid Everyday H1) dominate.
4. Acceptance check per cluster: it must contain ≥ 2 embodiments with non-trivial share; single-embodiment clusters
   are reported as embodiment artefacts, not grasp classes.
5. Some postures are infeasible for some hands (e.g. Inspire thumb arc). Keep a per-embodiment feasibility mask
   instead of forcing every class onto every hand.
6. Every label table carries `feature_version`, `clustering_version`, source `canonical_version`.

### LEAP / DexWild

Superseded by the strict policy (config/verification.yaml): DexWild stores commands only, so no LEAP geometry is
stored or clustered. (The command decoding itself was verified against the glove targets, 21.5 mm median residual.)

## Result 2026-09-23: first grasp-moment clustering (verified hands only)

Layer `grasp_moments_v1` (`unidex/moments.py`): HRDexDB F1 = object lifted > 3 cm (object_6d_pose_v2 in robot frame;
failed grasps max 0.2 cm, successful 6–19 cm); RealDex = authors' contact.txt; H1 excluded (no contact signal).
Hold-level Ward clustering (`scripts/analysis/cluster_grasps.py`): 71 holds (Shadow 63, F1 8) split first by hand
(k=2, silhouette 0.75). Cause is confounded: all Shadow holds are one large cylinder (thumb–finger distances
1.5–2.2 palm widths), F1 holds are 8 small/medium objects (0.3–0.7). The data cannot separate "hand" from "object".
Next: object-category-matched comparison (RealDex and HRDexDB share categories: lotion, sprayer/spray bottle, cup,
watering can/jug, box, can/jar).

## Result 2026-09-23 (2): object-category-matched comparison

Data: RealDex Shadow (cylinder, body_lotion x4 seq, blue_cup, box x2) vs HRDexDB Inspire F1 (58 episodes on 10
matched objects). Three more RealDex objects (sprayer, elephant_watering_can, goji_jar) blocked by Google Drive quota.

1. With v1 moments (RealDex = all contact frames incl. approach/release, F1 = lifted hold) cross-hand nearest-neighbour
   category retrieval was at chance (F1->Shadow 0.10 vs 0.24, Shadow->F1 0.32 vs 0.24). R2(category) 0.16,
   R2(hand) 0.07 — the category signal was within-hand only.
2. The two moment definitions were not comparable. RealDex objects barely lift (max 3–4 cm, tracking-noise level), so
   the lift criterion cannot be used there; RealDex's own grasp annotation is segment.txt (authors' grasp segments,
   body_lotion only in our sample). Using segment ends (178 Shadow lotion grasps): the Shadow-lotion centroid is
   closest to F1-lotion among all 7 F1 categories (0.74 vs next 1.29 palm widths; permutation p = 0.002, F1 n = 5).
   Remaining offset is in the thumb (d_TI 1.77 Shadow vs 1.18 F1) — consistent with the thumb embodiment gap.
3. => `grasp_moments_v2`: RealDex = authors' segment ends only; objects without segment.txt excluded.
   Cross-embodiment grasp structure is visible when moments are defined consistently, but the evidence is one
   category with n = 5 on the F1 side. Needs more RealDex objects that ship segment.txt.

## Result 2026-09-23 (3): choosing k by information about the future (`scripts/analysis/choose_k.py`)

Verified streams with timestamps (100 hand-episodes: HRDexDB F1, H1 L/R, RealDex Shadow), 10 pairwise distances,
anchors every 0.2 s, 5-fold GroupKFold by episode, hand-balanced. Metric = held-out gain over copy-current when the
FUTURE class label is known (per-class ridge from current posture). k = 1 = present posture only.

| k | all, D=1 s | event (top-20% change), D=0.5 / 1 / 2 s | label entropy | smallest class |
|---|---|---|---|---|
| 1 | 0.11 | 0.11 / 0.22 / 0.34 | 0 | — |
| 2 | 0.45 | 0.48 / 0.67 / 0.80 | 0.9 bit | 36% |
| 3 | 0.54 | 0.62 / 0.76 / 0.86 | 1.5 | 22% |
| 5 | 0.63 | 0.71 / 0.82 / 0.90 | 2.2 | 12% |
| 8 | 0.66 | 0.75 / 0.85 / 0.92 | 2.8 | 4.5% |
| 12 | 0.70 | 0.79 / 0.88 / 0.93 | 3.4 | 3% |
| 24 | 0.74 | 0.83 / 0.90 / 0.94 | 4.2 | 1% |

- Knee at k ≈ 4–5: a 5-class future label recovers ~82% of the event change at 1 s (present-only: 22%); beyond k=5
  each doubling adds only 2–3 points while the smallest class drops below 5%.
- Same shape per dataset (event, D=1 s, k=5: F1 0.74, H1 0.86, RealDex 0.87).
- These gains are an UPPER BOUND (perfect future label); the real head predicts the label imperfectly.
- k=5 classes (thumb–index distance): closed/opposed 0.81 (all hands), medium wrap 1.43 (F1 35%, Shadow 45%),
  wide wrap 1.82 (F1 33%, Shadow 41%, H1-L 24%), fully open 2.18 (H1 42–55%, F1/Shadow ≈ 0), H1-only
  "index+middle extended, ring+pinky flexed" 1.37. Two of five classes are essentially H1-only: the open hand of
  F1/Shadow lands in "wide wrap" (thumb embodiment gap) and H1 contains free-space gestures absent elsewhere.
- Candidate: `hand_posture_class_v1` = KMeans k=5 on pairwise distances, hand-balanced. Not final: small data,
  embodiment-split "open" class; re-fit after scaling the verified set.
