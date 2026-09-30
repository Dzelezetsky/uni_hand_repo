# Audit of new_datasets.md candidates (2026-09-29)

Question per dataset: can we take the hand's joint values at time t, map them with an OFFICIAL conversion to the
joints of an OFFICIAL hand model, and run FK to get fingertips in the palm frame (config/verification.yaml V1–V6)?
Contact / object signals are not required. Evidence = HF `meta/info.json` + README of each repo, one sampled parquet
where noted, official code of the dataset authors / manufacturer.

Legend: V1 model · V2 raw→joint mapping · V3 measured state · V6 timestamps (LeRobot `timestamp` everywhere below).
V4 (visual FK vs RGB) and V5 (limits) need a downloaded sample and are open for every new dataset.

## Tier A — all criteria look satisfiable from official sources (next: 1 episode + V4 check)

| dataset | hand (per side) | data | V1 model | V2 mapping | V3 | size / license |
|---|---|---|---|---|---|---|
| **T-Rex** `zekaiwang/trex_dataset` | Sharpa Wave, 22 joints, both hands | `observation.state[58]` = L_arm7, L_hand22, R_arm7, R_hand22, "joint positions"; `action` = targets | official `sharpa-robotics/sharpa-urdf-usd-xml` (vendored by the authors) | authors' `dataset_quickstart/.../schema.py` `SHARPA_HAND_JOINT_ORDER` (dim → URDF joint name) + authors' Pinocchio FK in `robot.py` | yes (state vs target stored separately) | 5,464 ep, 5.47 M frames, 30 fps, 207 objects, 22 primitives; MIT; 1.53 TB (select episodes; videos dominate) |
| **Dexora** `Dexora/Dexora_Real-World_Dataset` | XHand1, 12 joints, both hands | `observation.state[39]` = arms 6+6, hands 12+12, head 2, spine 1; values in rad (sample: ≤ 1.49) | RobotEra official STAR1 URDF (`roboterax/models`, BSD-3): its XHand joints are IDENTICAL to the SPIDER XHand URDF (all 16 finger joint origins/axes equal, one limit 1.92 vs 1.94) | official Dexora `deploy/xhand_forwarder.py`: index 0..11 = thumb_bend, thumb_rota1, thumb_rota2, index_bend, index1, index2, mid1, mid2, ring1, ring2, pinky1, pinky2 (= XHand SDK = URDF names) | state from `/observation/*/joint_state` | 11,517 ep, 2.92 M frames, 20 fps, 347 objects; MIT; ~240 GB |
| **VITRA-TeleData** (already imported) | XHand1 right | measured joints, rad | same finding as Dexora → the only failing check (V1) can pass | verified before | yes | small |
| **Vega + XHand1** `tarzanagh/vega-xhand-teleop` (not in the list) | XHand1, both hands | `*_hand_joint_positions` measured rad + `*_hand_target_joint_positions` | as Dexora | names documented | yes, BUT README: "commanded and measured differ by up to ~2 rad on the thumb even at rest" → V4 must decide | ~430 ep, 9 tasks; license "other" |
| **Sharpa RoCo_TaskBoardAssembly** `SharpaIT/RoCo_TaskBoardAssembly` | Sharpa Wave 22, both | `observation.state[65]` names `left_hand_j0..j21` + joint torque + tactile | official Sharpa | order j0..j21 → URDF joint names NOT documented (probably the Sharpa SDK order used by T-Rex; must be confirmed) | state vs action separate | 11 ep, 68 k frames; CC-BY-4.0 |
| **Sharpa Robotic_Origami_Challenge** `SharpaIT/Robotic_Origami_Challenge` | Sharpa Wave 22, both | same schema family as RoCo (gated, not read) | official Sharpa | as RoCo | ? | ~680 ep, 4.8 M frames; CC-BY-4.0; **gated: user must accept terms on HF** |

## Tier B — usable hand + model, but the raw→angle conversion is not (yet) official

| dataset | hand | what is missing |
|---|---|---|
| **PetalDex** `jasonGUself/PetalDex` | Wuji (20 joints/hand, official `wuji-technology/wuji-description`) | state has NO joint names; order of the 20 values and the Wuji hand generation (hand 1 vs hand 2) undocumented. Values look like rad (−0.5…1.64). 525 + 200 ep, 30 fps, Apache-2.0. → ask authors |
| **Unitree G1 WBT Inspire** (`unitreerobotics/G1_WBT_Inspire_*`, ~27 tasks, ~6 k ep, 30 fps, Apache-2.0; not in the list) | Inspire, most likely RH56DFX (Unitree unifolm-vla docs; confirm). Model = Unitree Inspire URDF, already verified for H1 | `hand_state` normalized 0..1 **open→close**, order index, middle, ring, little, thumb bend, thumb rotation (README). Unitree's H1 stack (xr_teleoperate) uses the OPPOSITE convention (1 = open) with ranges [0,1.7],[0,0.5],[-0.1,1.3]; no official formula for the WBT stack found → would be our assumption. Sample: state 0…0.90, thumb rotation ≈ 0. → ask Unitree / find WBT recorder code |
| **Unitree G1 WBT BrainCo** (`unitreerobotics/G1_WBT_Brainco_*`, ~29 tasks, ~6.7 k ep) | BrainCo Revo2 (official URDF `BrainCoTech/revo2_description`, already in registry) | `hand_state` 0..1 open→close (observed 0…0.8); no official normalized→angle mapping (revo2_description documents joint ranges only; 6 motors drive 11 joints) |
| **LET-Dex** `LejuRobotics/LET-Dex-Dataset` | Linker Hand L6 (official `linker-bot/linkerhand-urdf`, L6 v3.1) | hand state in driver units 0..255 (255 = open), 6 active + 5 passive joints; no official unit→angle mapping found in Linker SDKs. ROS bags, 707 GB, CC BY-NC-SA |
| **OpenArm Banana** `June777/openarm_banana_*` | Inspire RH56F1 (we have the HRDexDB F1 URDF) | gated (accept terms); README: 6 driven joints per hand, other 6 follow "the URDF's multipliers" — which URDF and units must be read after access. 406 / 1,072 ep, 30 Hz |
| **DexH2R** (github 4DVLab/DexH2R) | Shadow Hand (same lab as RealDex) | `qpos.pt` dimension / order / measured-vs-command undocumented in README; Google Drive (quota problems as RealDex); CC BY-NC 4.0 → download one sample |
| **RoboCOIN** five-finger subsets | per platform | every subset gated (accept terms per repo); five-finger platforms not listed publicly → inspect after access |

## Tier C — excluded (reason)

| dataset | reason |
|---|---|
| EgoSteer-RealWorld | Ruiyan RY-H2 hand values are normalized driver values, no official conversion / model found → video pretraining only |
| XL-VLA | teleop dataset not released (only `Dataset/demo.npz`); Paxini DexH13 is four-fingered |
| EgoEngine (Aria-Mustard XHand) | tiny, no license, provenance of sequences unclear |
| robot-dex | 5 episodes, gated; same Linker L6 issue as LET-Dex |
| G1 GR00T Inspire DFTP sets (MLeggiero pipette / pick_and_place, birbirll piston, cloudwalk sneaker) | third-party conversions to "rad"; pipette hand constant; DFTP model already failed V5 on HRDexDB |
| ObjectInHand / PoseFusion | not investigated (small, old); revisit only if needed |
| Already decided earlier | ActionNet (motor units), AgiBot (no hand model), RoboMIND (mapping/state unknown), DexWild (commands only) |

## Corrections to new_datasets.md

- Dexora: HF release has 11,517 episodes / 2.92 M frames at **20 fps** (not 12.2 K).
- PetalDex: 525 (robot_auto) + 200 (co-creation) episodes; state has no joint names.
- New large sources not in the list: Unitree UnifoLM WBT (Inspire + BrainCo, ~12.7 k episodes).
- XHand1 now has an official RobotEra model (STAR1 URDF) identical to the SPIDER model.

## Decisions / actions needed

1. User: accept the RobotEra STAR1 hand as the official XHand1 model? (unlocks VITRA, Dexora, Vega-XHand)
2. User (browser): accept HF terms for SharpaIT/Robotic_Origami_Challenge, June777/openarm_banana_*, RoboCOIN subsets.
3. Ask authors: PetalDex (joint order, Wuji generation), Unitree (WBT hand_state formula, Inspire model),
   BrainCo (Revo2 normalized→angle), Linker (L6 0..255→angle).
4. Engineering order: T-Rex adapter (new 22-DoF hand, largest clean source) → Dexora → Sharpa RoCo/Origami.

## Status update 2026-09-29 (after user decisions: XHand = STAR1 official model; gated terms accepted)

| dataset | status |
|---|---|
| VITRA-TeleData | **verified** (V1 now pass with the STAR1 model); re-converted |
| T-Rex | **verified** (V4 coarse: timing/idle-hand consistent, finger detail not resolvable at 640x360); all 5,464 episodes converted, 101 h of hand streams; V5 0.004–0.016 % > 0.15 rad |
| Dexora | adapter done; V4 pass on airbot_dexterous ep 50 (straight-finger book hooking, hand static as in video) and ep 300 (left power grasp on ukulele neck, release); release text: `instruction` matches the video, `action_name` does NOT (kept in extra with a note) |
| Sharpa Origami | access OK; 3,982 ep, 23.6 M frames (~218 h), 30 fps, CC-BY-4.0, proprioception 8 GB (v3). Channel order j0..j21 not documented, but every channel's range fits its URDF joint in the T-Rex/URDF order (AA joints around 0, pinky_CMC 0..0.22 of 0.26, thumb_CMC_FE up to 1.42) -> inferred, needs user acceptance |
| OpenArm Banana (1,072) | access OK; **excluded**: third-party merge whose counts->rad factors were fitted by the uploader ("four fingers still unmeasured"), grafted left-hand tracks, reconstructed actions |
| RoboCOIN | still gated per sub-repo |

## Status update 2026-09-30

- Correction: Origami has **1,813 usable episodes** (129 seasons with LeRobot v3 data; my earlier 3,982 / 23.6 M
  summed the v3 AND v2.1 copies of the same data). 2 seasons exist only as v2.1 and 12 ship no data files -> skipped.
  ~10.8 M frames per hand (~100 h).
- Dexora: **verified and converted** — 11,517 episodes, 80.7 h of XHand streams, V5 0 % > 0.15 rad.
- Origami: **verified** (V2 inferred + user-accepted, V4 coarse), V5 0.017 % (R) / 0.16 % (L) > 0.15 rad; converted.

## Correction 2026-09-30: EgoSteer-RealWorld moves from Tier C to Tier A

The Tier C verdict was wrong (based on the summary in new_datasets.md, not on the authors' code). The official
`egosteer/robot-stack` (commit ba06f62) ships the RY-H2 MJCF (`assets/ruiyan_hand_mjcf/{left,right}/hand.xml`, 6 active
+ mimic joints via `<equality polycoef>`), and `src/hand/hand/hand_fk_node.py` gives the authors' conversion:
state = motor_position/4095; `n = state / [0.6,1,1,1,1,1]`; `q = low + n * (high - low)` per MJCF joint range; mimic
joints from the equality constraints. The dataset's own fingertip columns `[44:74]` are computed with this FK ->
V2 can be cross-checked numerically. V1 dataset_author_model, V3 measured (`observation.state` = motor feedback,
`action` = commands), V6 timestamps (30 Hz nearest-neighbour resampling from native 80 Hz hand stream).
Open: V4, V5, numeric check vs shipped fingertips. Size: 54,454 ep, 192 h, both hands; parquet 17 GB, RGB 510 GB,
depth 2.65 TB (depth not needed). Apache-2.0. Caveat: normalized->angle is linear by the authors' definition.

## Status update 2026-09-30 (evening)

- EgoSteer: **verified** V1-V6 (details in config/verification.yaml; FK replay matches the shipped fingertips to
  0.0001 mm; V4 on three tasks). States-only download (~17 GB) running; adapter next.
- OpenArm Banana 1072: re-opened. The uploader is the rig's own lab (vclab, same URDF multipliers as HRDexDB F1).
  Raw F1 counts recovered exactly from their affines; the official HRDexDB F1 formula fits the recovered counts,
  the uploader's 750 counts/rad for the fingers does not (closure 1.148 vs 1.473 rad at the same mechanical stop).
  Awaiting user decision on V2 = "recovered counts + HRDexDB formula"; then V4 with the videos.
- OpenArm Banana 1072: **verified and converted** (user decision: recovered F1 counts + HRDexDB formula), right hand
  only, 1,072 ep, 3.25 h; V4 coarse on floor2 / american / monkey.
- EgoSteer: RY-H2 MJCF converted to URDF (scripts/models/mjcf_to_urdf_ruiyan.py; URDF FK == MuJoCo == shipped tips),
  adapter unidex/adapters/egosteer.py; full conversion runs after the states download.
- 2026-09-30 final: EgoSteer converted (54,454 ep, 89,095 active verified hand streams, 329.6 h; V5 0 % because the
  authors' mapping clips to the joint range, raw excess <= 0.01 motor units). H1 full converted (4,883 ep).
  Verified corpus: 736.5 h of hand streams, 6 hand families.
