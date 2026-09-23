# PROJECT.md — Scientific Project Description

## 1. Working research idea

The project studies whether a world model for dexterous robot manipulation benefits from explicitly predicting **future hand configuration / grasp information**.

A standard video/world model may learn task-relevant manipulation structure implicitly, but there is no guarantee that its latent representation cleanly preserves the future dexterous hand state needed by an action policy.

The proposed idea is to add an explicit future-hand prediction objective during world-model adaptation.

---

## 2. High-level architecture

Let:

- \(o_{\le t}\) be past visual observations;
- \(l\) be language/task instruction;
- \(h_t\) be the world-model hidden representation;
- \(c_{t+\Delta}\) be a future grasp/posture class;
- \(H_{t+\Delta}\) be a continuous canonical future hand geometry.

World model:

\[
(o_{\le t},l) \xrightarrow{\text{World Model / Cosmos}} h_t
\]

Auxiliary hand prediction:

\[
h_t \rightarrow \hat c_{t+\Delta}
\]

and/or:

\[
h_t \rightarrow \hat H_{t+\Delta}
\]

Later action policy / ActionDiT may condition on:

\[
(h_t, e_c, q_t)
\]

where \(e_c\) is an embedding of the predicted future hand state.

---

## 3. Why explicit future hand prediction

Dexterous manipulation often requires the robot to anticipate how the hand should be configured before contact.

Examples:

- thumb-index pinch;
- multi-finger enclosure;
- handle grasp;
- tool grasp;
- preshape for insertion;
- bimanual role specialization.

A generic video prediction objective may focus on dominant scene motion while underrepresenting small but task-critical finger geometry.

The auxiliary objective is intended to bias the representation toward these manipulation-critical future variables.

---

## 4. Why use robot proprioception instead of VLM pseudo-labels

The selected datasets frequently contain exact or near-exact:

- hand joint positions;
- actuator positions;
- end-effector poses;
- tactile/contact data.

Therefore future hand labels can be derived from robot state rather than estimated from RGB.

Preferred supervision:

\[
q_{t+\Delta}^{hand}
\rightarrow
\text{kinematics}
\rightarrow
H_{t+\Delta}
\rightarrow
\text{optional class } c_{t+\Delta}
\]

This is much cleaner than:

```text
RGB → VLM / hand tracker → pseudo hand label
```

VLMs may still be used later to create missing text annotations, but not as the primary hand-state ground truth when robot proprioception exists.

---

## 5. Cross-embodiment challenge

The source corpus contains different five-finger hands:

- Inspire;
- XHand;
- LEAP Hand V2 Advanced;
- Shadow Hand;
- Fourier dexterous hands;
- potentially BrainCo Revo2 later.

Their native joint spaces are incompatible.

Therefore the project introduces a common geometric representation based on fingertip positions relative to the palm.

This is described in `KINEMATICS.md`.

---

## 6. Stage 1

Stage 1 adapts a pretrained video/world model to robot manipulation.

Conceptually:

\[
L_{\text{stage1}}
=
L_{\text{video}}
+
\lambda_h L_{\text{future-hand}}
\]

Possible auxiliary variants:

### Classification

\[
L_{\text{future-hand}}
=
CE(\hat p(c_{t+\Delta}), c_{t+\Delta})
\]

### Continuous geometry regression

\[
L_{\text{future-hand}}
=
\|\hat H_{t+\Delta} - H_{t+\Delta}\|_2^2
\]

### Combined

\[
L =
L_{\text{video}}
+
\lambda_c L_{\text{class}}
+
\lambda_g L_{\text{geometry}}
\]

Initial implementation should keep all three options possible.

---

## 7. Stage 2 / downstream action policy

A later ActionDiT / inverse-dynamics action model can condition on:

- world-model hidden representation;
- current robot proprioception;
- predicted future hand embedding.

For example:

\[
(h_t, e_{grasp}, q_t)
\rightarrow
A_{t:t+H}
\]

The project must first show that future-hand information is useful before assuming this is the final architecture.

---

## 8. Key oracle experiment

Before investing heavily in predicted grasp tokens:

Compare:

### Baseline

\[
ActionPolicy(h_t, q_t)
\]

### Oracle

\[
ActionPolicy(h_t, q_t, c^{GT}_{t+\Delta})
\]

If ground-truth future hand information does not improve downstream control, the explicit bottleneck may not be useful.

If oracle helps, then test predicted future-hand labels.

This is a critical scientific de-risking experiment.

---

## 9. Research questions

1. Can heterogeneous five-finger robot hand states be mapped into one meaningful geometric space?
2. Does that space contain stable, interpretable grasp/posture structure?
3. Can a world model predict future hand geometry/classes from past RGB + language?
4. Does auxiliary future-hand supervision improve world-model representations?
5. Does predicted future-hand information improve downstream dexterous action generation?
6. Are gains robust across hand embodiments and datasets?
7. Is a discrete class better than continuous hand geometry?
8. How far into the future should the hand target be predicted?

---

## 10. Current scientific stance on "grasp class"

Do not oversell unsupervised pose clusters as "optimal grasp strategies".

A cluster derived from hand geometry is more defensibly called:

- future hand configuration;
- future hand posture;
- grasp posture;
- interaction-mode token, if contact semantics are included.

Functional "strategy" requires stronger evidence than posture clustering alone.

---

## 11. Expected contribution direction

A plausible contribution is:

> Generic video/world-model adaptation does not necessarily force the latent representation to explicitly encode task-critical future dexterous hand configuration. We introduce a cross-embodiment hand representation and future-hand auxiliary supervision derived from robot proprioception, and study whether this improves downstream dexterous manipulation.

The contribution should be validated empirically, not assumed from the architecture.
