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

## Decision 2026-09-29: vocabulary inputs = hand proprioception only; no pad/side attribute

Context: Feix et al. 2016 (GRASP taxonomy) describes grasps by power/intermediate/precision, opposition type
(pad/palm/side), virtual fingers and thumb abduction/adduction. Pad normals were added to the canonical geometry for
this (KINEMATICS.md §5b).

### Rule A — no contact / object information as clustering input

Features of the shared posture vocabulary (`hand_posture_class_*`) must be computable from verified hand
proprioception alone. Object pose/mesh, contact labels and tactile are NOT inputs: each would restrict the corpus to
the few datasets that have them. Consequence (accepted): the vocabulary describes HAND SHAPE, not functional grasp
(fist in free space = palm-opposition grasp; preshape = grasp) — the posture/grasp split of CLAUDE.md §16. Datasets
with object or contact signals may be used only as an EVALUATION layer (do posture classes match real grasps?), never
to define classes or features.

### Rule B — no pad-vs-side opposition attribute in the shared vocabulary

Low-DoF hands (Inspire: 1 coupled DoF per finger, 2 thumb DoF, no finger abduction) meet the thumb on a fixed curve;
which side of the thumb touches the finger is set by the mechanism, not by the operator. Measured
(`scripts/analysis/reachability.py`, `unified/validation/reachability/report.txt`): angle between thumb and index pad
normals in reachable pinch configurations (tip distance < 0.35 palm widths):

| hand | p5 / p50 / p95 | max |
|---|---|---|
| Shadow | 21 / 50 / 98° | 149° |
| Inspire DFX | 17 / 46 / 74° | 82° |
| Inspire F1 | 71 / 90 / 107° | 113° |

The intervals are hand-specific, so a pad/side attribute would encode the embodiment. In addition, with the tip-point
normal the contact surface is not observed (pinch contact is not at the tip point). Pad normals stay stored (physically
correct; per-hand continuous targets, analysis) but are not an axis of the shared vocabulary. Re-open only if a
normal-based feature passes both checks of Rule C.

### Rule C — admission test for any new vocabulary feature

1. Mixing ratio across hands ≈ 1 (as for pairwise distances, 1.21), `scripts/analysis/embodiment_gap.py`.
2. Reachability: each hand's REAL postures must be reproducible by the other hands in that feature
   (`unidex/reachability.py`: native-space sampling through the verified mapping + box-constrained least-squares fit;
   sampling URDF joints independently is wrong for coupled hands). Values/classes reachable by only some hands go to
   the per-embodiment feasibility mask (rule 5 above), not into a shared class.

### Result 2026-09-29: cross-hand reachability of real postures (10 pairwise distances)

Fraction of hand A's real frames (rows) that hand B (columns) reproduces within 0.05 palm widths RMS (~4 mm):

|  | Shadow | DFX-R | DFX-L | F1 |
|---|---|---|---|---|
| Shadow | 1.00 | 0.74 | 0.74 | 0.46 |
| DFX-R | 1.00 | 1.00 | 1.00 | 0.53 |
| DFX-L | 1.00 | 1.00 | 1.00 | 0.38 |
| F1 | 1.00 | 0.98 | 0.95 | 1.00 |

- Shadow reproduces every observed posture of the Inspire hands; F1 postures are reproducible by all hands.
- Inspire DFX misses 26% of Shadow postures, mostly ring/pinky relations (T-P, M-P, R-P): no finger abduction.
- F1 misses about half of the other hands' postures, almost entirely in the thumb distances. CAVEAT: the F1 native
  range is the range OBSERVED in HRDexDB (p0.5..p99.5; e.g. thumb rotation uses 630..1233 of the raw scale), not the
  hardware range, so this is "not used in HRDexDB", not proven "infeasible". Needs an official F1 range.
- => the feasibility mask is real and asymmetric: a class learned mainly from Shadow/H1 data may be unreachable for
  F1 (thumb) and DFX (spread fingers). Evaluate every candidate class against this mask.

## Result 2026-09-29 (2): pad normals fail the Rule C admission test

`scripts/analysis/normals_clustering.py` -> `unified/clustering/normals_rule_c.txt`. Verified hands with normals
(Shadow 5394, F1 7045, DFX-R 1191, DFX-L 212 frames on a 0.2 s grid), hand-balanced, all features z-scored.

| features | dim | mixing ratio (median) | NMI(cluster, hand), k=5 | single-hand clusters |
|---|---|---|---|---|
| D  10 pairwise distances | 10 | **2.17** | **0.23** | 0 |
| D + 10 normal cosines | 20 | 2.75 | 0.33 | 1 |
| D + thumb normal | 13 | 3.93 | 0.30 | 0 |
| D + all normals | 25 | 3.50 | 0.29 | 0 |
| normal cosines only | 10 | 2.55 | 0.35 | 2 |

- Every normal-based set encodes the hand MORE than distances alone; the thumb normal is the worst (thumb gap again).
- Future information (event windows, D=1 s, x = current D+N for all): classes from D give the best future-distance
  gain (0.774 vs 0.743–0.745 with normals); classes from D+N improve future-NORMAL prediction (0.636 -> 0.717), i.e.
  normals add orientation information but only about themselves.
- Reachability in D+cos (dist RMS < 0.05 pw and cos RMS < 0.1): Inspire hands reproduce only 7–8 % of Shadow's real
  frames (0.74 with distances only); F1 reproduces 1–14 % of DFX frames. Inspire fingers have no abduction, so their
  pad normals are almost parallel; normal cosines separate Shadow from Inspire by construction.
- Note: the D mixing ratio here (2.17) is higher than the 1.21 of the 2026-09-23 study (different hand set: that
  study included hands now excluded by the strict policy, and a 0.5 s grid). Compare sets only within one run.
- => Normals are NOT added to the shared vocabulary features (confirms Rule B empirically). Use: per-embodiment
  continuous target (future pad orientation is predictable, gain 0.72) and analysis. Small-sample caveat: 4 hands,
  3 datasets, DFX-L only 212 frames; re-run after scaling.

## Result 2026-09-29 (3): clustering on the scaled verified corpus

Data: Inspire F1 591 episodes, Inspire DFX 506 R + 85 L (H1), Shadow 20 (RealDex, 10 objects); 0.2 s grid,
family-balanced (Shadow / DFX / F1, 4000 each). Features = 10 pairwise distances. Grasp samples = grasp_moments_v1
(F1 object lifted > 3 cm, 587 episodes; Shadow authors' contact.txt), used only to interpret clusters.
`scripts/analysis/cluster_v2.py`, `unified/clustering/v2_*` (report, medoid mesh renders), `choose_k.py` rerun.

- Number of classes: future-information knee again at k = 5 (event windows, D = 1 s: k=3 0.79, k=5 0.87, k=8 0.90;
  per dataset F1 0.75, H1 0.89, RealDex 0.87); KMeans k=5 bootstrap ARI 0.97. => keep k = 5 for
  `hand_posture_class_v1`.
- ALL postures, k = 5 (by thumb-index distance, palm widths):
  1. closed / power (TI 0.96, fingers curled, thumb across) — all three families, Shadow contact 95 %;
  2. index+middle extended, ring+pinky flexed — DFX 88 % (H1 gesture/free-space), F1 borderline reachable;
  3. index separated from the curled fingers (IM 1.0) — Shadow 57 % / DFX 29 %;
  4. semi-flexed fingers, thumb abducted (wide wrap / preshape) — Shadow + F1;
  5. flat open hand — DFX 83 %; NOT reachable by F1 within its HRDexDB range (fit RMS 0.12 pw, thumb).
- GRASP samples: clusters follow OBJECT SIZE, consistent with Feix (size is the main variation within a grasp type):
  k=3: closed wrap on thin objects/handles (banana, whisk, spice mill, frying-pan handle, book; 69 objects, purity
  0.91) / wide wrap on bulky objects (ramen, tuna can, orange, apple, soap tray + Shadow big_eye_toy; 27 objects,
  purity 0.75) / index-separated grasp (mostly Shadow). k=5 adds a thin-object pinch-like class (TM 0.44, TI 0.56,
  F1 95 %) and splits medium (apple, cup, lemon) from large (cans, beer, cylinder) wraps.
- Pinches exist: 22 % of F1 grasp samples have thumb within 0.35 palm widths of index or middle tip; Shadow 0.5 %
  (RealDex objects are large). A pinch class is therefore single-embodiment because of DATA COVERAGE, not kinematics
  (Shadow reproduces every F1 posture, reachability 1.00).
- => Refine rule 4 above: a single-embodiment cluster is an artefact only if the other hands cannot REACH it
  (reachability.py); if they can, it is a coverage gap (keep the class, flag it, get data with small objects).

## Result 2026-09-30: clustering on the 388 h corpus (5 hand families)

Families (balanced, 4000 samples each): Sharpa Wave (T-Rex + Origami), XHand1 (Dexora + VITRA), Inspire DFX (H1),
Inspire F1 (HRDexDB), Shadow (RealDex). `cluster_v2.py` (per-stream sampling), `choose_k.py` (<= 400 streams/family).
Reports `unified/clustering/v3_report_k7.txt`, gallery `v2_gallery_all_k7.png`.

- Future-information knee still k = 5 (event, D = 1 s: k5 0.85, k8 0.88, k12 0.90); Sharpa datasets gain less
  (0.63-0.67 at k5 vs 0.75-0.89 elsewhere) -> the 22-DoF hand's variation is not captured by a few classes of 10 distances.
- With 5 families NO single-family cluster up to k = 8 (3-family data had 2-4). Stability: k5 ARI 0.984, k7 0.987.
- k = 7 clusters are semantically clean against T-Rex labels (NOT used for clustering):
  closed power grasp (wrap/fold/peel/pour) · thumb-index pinch with ring+pinky abducted (twist x8, screw x14) ·
  index separated (coin x8, insert/screw) · index+middle extended, ring+pinky flexed (DFX-heavy; tower of Hanoi x60) ·
  flat hand, fingers together, thumb abducted (press, keyboard, switch, card) · semi-flexed wide wrap / preshape ·
  fully open hand (open, reach).
- F1 cannot reach the open-hand classes (thumb range as used in HRDexDB) -> feasibility mask.
- Candidate: `hand_posture_class_v1` = KMeans k = 7 on pairwise distances, family-balanced (k = 5 remains the
  information knee; k = 7 adds precision classes that the new data populate).

## Result 2026-09-30 (2): clustering on the 736 h corpus (6 hand families, + Ruiyan RY-H2 / EgoSteer)

Families balanced (4000 each): Sharpa, XHand, Inspire DFX (H1 full), Inspire F1 (HRDexDB + OpenArm Banana), Shadow,
Ruiyan RY-H2 (EgoSteer). Same features (10 pairwise distances). Reports `unified/clustering/v4_report_k{7,8}.txt`,
gallery `v4_all_medoids_k7.png`, assignments `v4_assignments_all_k7.parquet`, knee `choose_k.*` (overwritten).

- Future information (event, D = 1 s): k3 0.78, k5 0.82, k6 0.85, k8 0.87, k12 0.89 -> knee k = 6-8. Weakest gain:
  Sharpa (T-Rex 0.57-0.66, Origami 0.64-0.69 at k5-8) and EgoSteer (0.73-0.79), strongest H1/Dexora/Banana (~0.85-0.90).
- Stability (bootstrap ARI): k6 0.86 (unstable), k7 0.983, k8 0.976. No single-family cluster up to k = 12.
- k = 7 reproduces the 388 h classes; the new data populate them and label them (EgoSteer task names / T-Rex
  primitives, NOT clustering inputs):
  c5 thumb-index pinch, other fingers curled (Ruiyan 43 %, Sharpa 29 %): pick up magnet, rope, bow, die; coin, insert ·
  c1 closed power grasp, thumb on index+middle: hammer, pour, wipe, squeeze; wrap, screw, peel ·
  c6 tool grip with pinky out: cut x5, drumsticks x7 ·
  c2 index+middle extended, ring+pinky flexed (DFX 51 %): tower of Hanoi, twist, assemble ·
  c3 flat hand, fingers adducted, thumb abducted (XHand 62 %, Sharpa 33 %): press, keyboard, card — NOT reachable by
    F1 and RY-H2 (fixed finger spacing) -> feasibility mask ·
  c0 wide semi-open wrap / preshape: laptop, globe, trash can; Shadow large-cylinder holds ·
  c4 fully open hand (DFX 57 %): reach, open, lay out placemat — not reachable by F1.
- k = 8 splits c2 into a DFX-dominated (0.69) two-finger class and a small Sharpa tripod (disassemble) -> k = 7 kept.
- => `hand_posture_class_v1` candidate confirmed: KMeans k = 7, family-balanced, on 6 families.
