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
- Stage 2 (action decoder, backbone frozen): DiT cross-attending to layer-19 hidden states of the video model at video
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
