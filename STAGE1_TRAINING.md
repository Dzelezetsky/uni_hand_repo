# STAGE1_TRAINING.md — Stage-1 future-hand target (notes)

## Finding 2026-09-23: is the future hand a non-trivial target? (`scripts/analysis/future_predictability.py`)

Verified streams only (HRDexDB F1 68 ep, Humanoid Everyday H1 20 ep, RealDex Shadow 9 seq); anchors every 0.1 s.
relMSE = error / variance of the future geometry (0 = trivial, 1 = no better than predicting the mean).

| dataset | D=0.5 s: copy relMSE / moving>10mm | D=1 s | D=2 s |
|---|---|---|---|
| HRDexDB F1 | 0.09 / 11% | 0.22 / 22% | 0.51 / 39% |
| H1 | 0.24 / 19% | 0.48 / 30% | 0.98 / 48% |
| RealDex | 0.68 / 59% | 1.53 / 88% | 1.73 / 89% |

- The hand is static most of the time and changes in discrete events (grasp / release). Median change at 1 s is only
  1.4 mm (F1) / 4.4 mm (H1); p90 is 19 / 47 mm. A uniformly sampled loss is dominated by "copy current".
- Constant-velocity extrapolation is WORSE than copying beyond 0.25 s: hand changes are not smooth continuations of
  the current motion; anticipating them needs context (vision/language) — this is exactly what the auxiliary head
  is supposed to add.
- Below ~0.5 s the target is nearly trivial (F1/H1); ~1–2 s is where it becomes informative. RealDex is a special
  protocol (repeated grasp/release every ~2.5 s) and changes much faster.

Consequences for Stage-1:
1. Horizon: predict at 1–2 s (possibly multi-horizon 0.5/1/2 s), not frame-level.
2. Always report the copy-current baseline; evaluate separately on "event" windows (change > 10 mm).
3. Sample / weight training windows so that events are not drowned by static frames.
4. Posture-cluster switch rate at 1 s is 16–25% (F1/H1): a class target changes rarely -> class loss must be read
   against the "same class as now" baseline too.

## Reference: how mimic-video trains (arXiv 2512.15692, checked 2026-09-23)

- Stage 1 (video backbone, LoRA): Cosmos-Predict2 2B latent DiT; input = 5 clean context frames (latent) + T5 language;
  output = the whole window of future latent frames at once (flow matching), not a single next frame. For the mimic
  hand setup the backbone is finetuned on a ~200 h robot video corpus. 480x640 frames.
- Stage 2 (action decoder, backbone frozen): DiT cross-attending to layer-20 hidden states (paper text said 19; the
  released code and all checkpoints use xattn_layer_idx = 20, see below) of the video model at video
  flow time tau_v (+ proprioception) -> action chunk A_t = [a_t .. a_{t+Ha-1}]; mimic: relative EE pose + absolute
  hand joints. Only 1.5–2.2 h of task data per task. At inference tau_v = 1: future latents are pure noise, one
  backbone pass, no video is generated.
- NOT stated in the paper: number of future frames, fps / stride, action chunk length Ha, control rate.
  Base-model default (NVIDIA docs, Cosmos-Predict2 2B Video2World): 1 or 5 conditioning frames, 93 frames generated
  (~5.8 s at 16 fps). Whether mimic-video changed this window is unknown. The lucidrains reimplementation uses
  action_chunk_len=32 — its own choice, not the paper's.

## Decision 2026-09-23: future-hand target definition

1. Target = dense future hand trajectory: canonical 15D geometry (and pairwise distances) at every future LATENT frame
   of the video window (tokenizer compresses time 4x -> one latent frame = 4 video frames ≈ 0.25 s at 16 fps).
   This makes the horizon a property of the video window instead of a single hand-picked Δ.
2. Evaluation always per horizon (0.5 / 1 / 2 s ...), against the copy-current baseline, and separately on event
   windows (change > 10 mm) — see finding above: < 0.5 s is near-trivial, 1–2 s is informative.
3. Horizon of the hand target >= action chunk horizon Ha (chosen by us, ~1–2 s).
4. Leakage (CLAUDE.md §18): at tau_v = 1 the future latents are pure noise, so the hand head cannot read the clean
   future; any experiment with tau_v < 1 needs an explicit leakage control.
5. Sources have different native rates (15 / 30 / 50 Hz): training windows are built by time at the video model's fps
   from the stored per-sample timestamps (never by frame index). Sources without timestamps (RoboMIND) cannot be used.
6. Window size (number of future frames, fps) is fixed together with the Cosmos configuration.

## Design 2026-09-23: how to predict CONTINUOUS future hand geometry (not yet implemented)

Problem with the naive head (pool hidden states -> MLP -> 15D, MSE): the future hand is multimodal at 1–2 s (grasp or
not, which grasp). MSE regresses to the mean of the modes = a half-closed hand that occurs in no real future.
Classification is immune to this, plain regression is not.

Options:
1. RECOMMENDED FIRST — classify, then refine (anchor-style, one hierarchical head):
   outputs per future latent frame: logits over K=5 posture classes (hand_posture_class_v1) and a per-class residual
   Δ_c (either H = centroid_c + Δ_c or H = H_t + Δ_c). Loss = CE(class) + Huber/L1 on the residual of the TRUE class
   only. Inference: argmax class + its residual (or the mixture). Matches the measured structure: the future class
   recovers ~82% of event change at 1 s, the residual the rest (GRASP_LABELING.md, result 3).
2. Generative flow-matching head (same design as mimic-video's action decoder): a small DiT generates the future hand
   trajectory from noise conditioned on the video hidden states. Handles multimodality natively; same architecture
   as Stage 2. Harder to train/evaluate (best-of-K, likelihood); classes must be derived separately.
3. Mixture density / multi-hypothesis (winner-takes-all) head: in between, less interpretable than 1.

Attachment to Cosmos (any option):
- Do not mean-pool: hidden states are spatio-temporal latent patches. Use learned QUERY tokens, one per future latent
  frame (~0.25 s), cross-attending to the chosen layer (mimic-video: layer 19) at video flow time tau_v (sampled
  randomly in training). Output = trajectory over future latent frames (target definition above).
- Feed the CURRENT hand geometry H_t into the head (not into the video model; it is available proprioception at
  inference) and predict the CHANGE, so the head does not relearn copy-current and the loss focuses on events.
- Condition on hand_model_id (learned embedding): geometry differs per hand, especially the thumb.
- Weight event windows up (static frames otherwise dominate the loss).

Two regimes, both needed:
1. Frozen Cosmos + trainable head (probe): is future-hand information already in the representation? Measure before and
   after video finetuning.
2. Joint training L = L_video + λ·L_hand during Stage-1 finetuning: the actual hypothesis (the auxiliary loss reshapes
   the representation). Check with the probe from (1) and with Stage-2 action quality with vs without the loss.

Plan: start with option 1 (K=5 + per-class residual, per-future-frame queries, H_t and hand embedding as inputs).
Keep option 2 in reserve if multimodality WITHIN a class is large — checkable on current data: spread of future
postures within one class given the same current posture.

## Finding 2026-09-30: multimodality of the future posture WITHIN a class (`scripts/analysis/within_class_multimodality.py`)

Question from the design above: given the current posture, is the future posture inside one hand_posture_class_v1
class multimodal (-> option 2) or unimodal (-> option 1 suffices)? Space = 10 pairwise distances (z-scored with the
frozen model), event anchors (top 20 % change per dataset), 5 hand families, 50-neighbour neighbourhoods in the
current posture (other episodes only), 2-mode split along PC1 vs a Gaussian null (selftest: 6 % false positives,
58 % / 98 % detection at 3 / 4 sd mode gap). mm = norm over the 10 pairwise-distance differences (not per fingertip).
Reports `unified/clustering/within_class_multimodality{,_resid}.txt`.

| D = 1 s, family mean | frac bimodal | mode gap | L2-head error (mean -> nearest mode) | within-mode spread |
|---|---|---|---|---|
| A: p(y given x)                            | 0.51 | 125 mm | 38 mm | 42 mm |
| B: p(y given x, future class)              | 0.53 | 64 mm  | 21 mm | 28 mm |
| B, futures residualized on local x (control) | 0.30 | 52 mm  | 20 mm | 26 mm |

- The class removes the large modes (gap 125 -> 52-64 mm, the between-class structure) — as intended.
- Residual within-class bimodality is real but moderate: ~30 % of event neighbourhoods after the control (DFX 52 %,
  F1 42 %, XHand/Ruiyan/Sharpa ~20 %; null 6 %), mode gap ≈ distance between neighbouring class centroids
  (ratio 1.05). An L2/mean residual would add ~20 mm of mode-averaging error on top of ~26 mm irreducible spread
  there: ≈ +25 % RMS error in those neighbourhoods, ≈ +10 % over all event windows.
- Upper bound: conditions on hand proprioception only; video + language disambiguate part of it. Sharpa / Ruiyan
  B estimates rest on few local neighbourhoods (5-11 % of queries) -> noisy.
- Decision proposal: option 1 stays the first head, but the residual is multi-hypothesis — M = 2 residual hypotheses
  per class with winner-takes-all loss + hypothesis logits (K x M = 14 outputs per future latent frame), not a single
  L2 residual. Flow matching (option 2) remains the reserve if the probe shows the 2-hypothesis head saturating.
  (Finer classes are no alternative: choose_k gains only 0.87 -> 0.91 from k = 8 to 16.)

## Finding 2026-09-30: oracle experiment, stage A (no images) — `scripts/oracle/`

Does GROUND-TRUTH future hand information improve action-chunk prediction? Per dataset an MLP predicts arm and hand
deltas over the next 2 s (8 horizons) from arm history (t, t-0.25, t-0.5), current hand state + current posture
class, instruction text (TF-IDF+SVD). Variants add a future-hand input at 0.5/1/1.5/2 s. Episode-level splits, 3 seeds.
Arm target = future MEASURED arm joints (not commands). Hand target is nearly the oracle itself -> leak-prone, only the
ARM numbers test the hypothesis. Windows `build_windows.py` (H1 217 k, Dexora 588 k, T-Rex 846 k, EgoSteer 12 k-episode
subset 621 k anchors), `train_oracle.py`, `summarize.py` (table: unified/oracle/summary_stageA.txt, git-ignored dir).

ARM improvement over the no-oracle model, event windows (hand change in the top 20 %), far horizons 1.25-2 s, %:

| dataset (hand) | true class | true geometry | timing only (class changes?) | class 25 % corrupted | 50 % | random class |
|---|---|---|---|---|---|---|
| H1 (Inspire DFX)   | 19.5 ± 5.5 | 24.5 ± 3.2 | 13.0 ± 0.2 | 8.5 | 2.3 | -3.1 |
| Dexora (XHand)     | 22.3 ± 1.8 | 32.9 ± 3.6 | 17.3 ± 2.9 | 9.5 | 3.3 | -1.9 |
| EgoSteer (RY-H2)   | 21.7 ± 1.0 | 35.5 ± 0.3 | 18.1 ± 1.4 | 10.9 | 4.1 | -1.3 |
| T-Rex (Sharpa 22)  |  8.8 ± 2.3 | 30.2 ± 1.1 |  5.2 ± 1.0 | 5.9 | 3.5 | 0.3 |

(all anchors: 4-11 % for class, 13-25 % for geometry; near horizons roughly half of far.)

- The hypothesis survives stage A: knowing the future hand posture makes the ARM trajectory ~20 % more predictable at
  grasp/release events on three of four datasets; the random-class control is <= 0 -> not an input-size effect.
- Most of the class benefit is TIMING (when the hand will change): timing-only reaches 65-85 % of the class gain; the
  grasp TYPE adds ~3-6 points.
- Continuous future geometry is worth much more than the 7-class label, most of all for the 22-DoF Sharpa hand
  (30 % vs 9 %) — consistent with choose_k (few classes capture Sharpa poorly). => keep the continuous geometry head as
  a first-class target (class + residual design), do not reduce the target to classes.
- Accuracy matters: with 25 % random errors about half of the gain is left, with 50 % almost nothing. Random errors
  are harsher than a model's structured errors, but Stage-1 must predict the future class clearly better than
  "copy the current class" (which already is part of the base input) to help.
- Caveats: upper bound (true future); base has NO vision — images may already carry part of the timing information,
  so stage B (frozen visual features, H1 video is local) must confirm the gain with a visual base; arm target = future
  measured state.

## Reference: mimic-video code facts (vendored in third_party/mimic_video, upstream e3355db; checked 2026-09-30)

Checked by reading the code and `stage1/scripts/inspect_mimic.py` on the released checkpoints (RTX 3060):
- Video backbone `v2w_pretrained_cosmos.pt`: Cosmos-Predict2 2B DiT, 1956 M params, 28 blocks, width 2048, 16 heads,
  patch 2x2, latent 16 channels, bf16. Fixed window `state_t = 16` latent frames (61 video frames), 480x640 ->
  16 x 30 x 40 = 19,200 tokens. Text = cross-attention to T5-11B embeddings (1024-d, 512 tokens).
  Video windows in their configs: 61 frames at 5 fps (Bridge, 12 s) / 10 fps (LIBERO, 6 s), `obs_history` 5.
  Video finetuning = LoRA rank 256 (alpha 32) on q/k/v/output_proj, x_embedder, t_embedder, MLP; lr 1.778e-4.
- Hidden states for the action decoder: `hidden_states[20]` = output of the 20th of 28 blocks
  (`hidden_states[0]` = patch-embedded input); the forward stops after block 20. Shape (B, 16, 30, 40, 2048).
  Video noise level sigma is sampled per sample in training (`draw_video_sigma`, incl. 5 % log-uniform 200..1e5).
  They are stored with `.detach().clone()` -> NO gradient reaches the video model through the action/hand path.
  => for the joint objective L_video + lambda * L_hand the hand head must bypass this detach (our modification).
- Action decoder (Bridge `w2a_..._layer20`): separate DiT, 499 M params, 24 blocks, width 1024, 8 heads;
  cross-attention k/v 2048 -> 1024 after LayerNorm `ctx_norm`; token sequence = 1 obs token + 15 action tokens
  (Bridge action/obs dim 10: pos 3 + rot6d 6 + gripper 1; chunk 15 steps at 5 Hz = 3 s); time embedding `pair`
  takes BOTH flow times (action tau and video sigma); `obs_mask_token` (obs dropout 0.2). Only the embedders /
  final layer depend on the action dimension -> the trunk can initialize our decoders / hand head.
- RTX 3060 12 GB: one denoise pass up to block 20, B = 1, full 480x640 window: 4.5 s (incl. warm-up), peak 6.3 GiB.
- Environment: `third_party/mimic_video/model`, `uv sync --extra cu126` (torch 2.6 + cu126, flash-attn 2.6.3,
  transformer-engine 1.13, apex, megatron-core; prebuilt wheels from the NVIDIA cosmos-dependencies index, no nvcc).
  `source stage1/env.sh` before running (points TE at the pip libnvrtc when no CUDA toolkit is installed).
