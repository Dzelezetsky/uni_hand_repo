# KINEMATICS.md — Cross-Embodiment Hand Geometry

## 1. Problem

Different five-finger robot hands expose incompatible native states.

Example:

```text
Inspire: 6 values
XHand: 12 values
LEAP V2 Advanced: 17 values
Shadow: ~20–22+ values
```

Joint vectors cannot be directly concatenated into one clustering space.

The solution is to compare **geometry**, not actuator indices.

---

## 2. Canonical representation v0.1

For each hand and timestep compute five fingertip points:

\[
P_t =
\begin{bmatrix}
p_T\\
p_I\\
p_M\\
p_R\\
p_P
\end{bmatrix}
\in \mathbb R^{5\times3}
\]

where:

- \(T\): thumb;
- \(I\): index;
- \(M\): middle;
- \(R\): ring;
- \(P\): pinky.

These points must be expressed relative to a canonical palm frame.

---

## 3. Forward kinematics

For each hand model:

\[
q^{native}
\rightarrow
q^{model}
\rightarrow
FK
\rightarrow
p_{\text{tip}}
\]

The mapping from dataset state to model joints is dataset-specific.

Example:

```python
raw_q = adapter.get_hand_state(...)
model_q = hand_mapping.raw_to_model_q(raw_q)
links = fk(model_q)
```

---

## 4. Why palm-relative coordinates

World coordinates mix hand shape with arm motion.

The same pinch pose performed in two places gives different world XYZ.

We want:

> where are the fingertips relative to the hand itself?

This makes the feature approximately invariant to global translation and rotation.

---

## 5. Palm frame

Palm = rigid hand base.

Canonical axes can be defined from hand geometry.

Let:

- `w` = wrist/base point;
- `b_I,b_M,b_R,b_P` = bases of index/middle/ring/pinky.

Mean finger-base point:

\[
b = (b_I+b_M+b_R+b_P)/4
\]

Finger direction:

\[
y = \frac{b-w}{\|b-w\|}
\]

Across-palm direction:

\[
x_0 = b_I-b_P
\]

Orthogonalize:

\[
x' = x_0 - (x_0^T y)y
\]

\[
x = x'/\|x'\|
\]

Normal:

\[
z=x\times y
\]

Then:

```text
+Y : wrist → fingers
+X : pinky side → index side
+Z : palm normal
```

Rotation:

\[
R=[x\ y\ z]
\]

Palm origin may be:

\[
o=(w+b)/2
\]

or a validated palm-link origin.

Point transform:

\[
p^{palm}=R^T(p-o)
\]

If the FK library directly returns target links relative to the palm link, prefer that.

---

## 6. Left hand

Left/right should not create duplicated grasp classes.

Preserve:

```text
side = left/right
```

but mirror left-hand canonical geometry into the same convention as right-hand geometry before cross-embodiment clustering.

The reflection rule must be explicit and unit-tested.

---

## 7. Scale normalization

Different robot hands have different physical sizes.

Keep metric geometry:

```text
fingertips_palm_m
```

but also normalize for cross-embodiment clustering.

Suggested hand scale:

\[
s=\|b_I-b_P\|
\]

Then:

\[
P^{norm}=P^{palm}/s
\]

Store both.

---

## 8. Derived geometric features

From five fingertips compute 10 unique pairwise distances.

Especially useful:

\[
d_{thumb,index}
\]
\[
d_{thumb,middle}
\]
\[
d_{thumb,ring}
\]
\[
d_{thumb,pinky}
\]

These help distinguish:

- open hand;
- pinch-like posture;
- multi-finger enclosure;
- power-like closure.

Do not assume these features alone define semantic grasp types.

---

## 9. Low-DoF coupled hands

A 6D Inspire state may drive more than 6 physical joints.

Need:

```text
actuator state
→ passive/coupled physical joints
→ FK
```

Use:

- URDF mimic relationships;
- manufacturer mapping;
- dataset mapping.

Never invent passive-joint angles.

If coupling is unknown:

```text
canonicalization_status = missing_joint_coupling
```

---

## 10. State-source quality

Possible inputs:

```text
measured_joint
measured_actuator
command
vision estimate
```

A commanded target does not guarantee the physical hand reached it.

Prefer measured states.

Store state provenance.

---

## 11. Why start with 5 tips, not 21 keypoints

Five tips are:

- available across most five-finger hands;
- easy to interpret;
- low-dimensional;
- enough to test whether cross-embodiment clustering is viable.

If insufficient, expand later to:

```text
finger base
finger middle
finger tip
```

for each finger.

Do not start with unnecessary complexity.

---

## 12. Validation

For every hand:

### Joint-isolation test

Change one native coordinate and see which finger moves.

### Neutral/open pose

Render geometry and compare with real image/official hand pose.

### Closed pose

Verify correct flexion and thumb opposition.

### Left/right

Verify mirror convention.

### Units

Detect degrees/radians/normalized values.

### RGB overlay

When calibration permits, project FK landmarks into camera image.

---

## 13. First PoC

Priority:

1. VITRA / XHand.
2. RealDex / Shadow.

Then compare nearest neighbors in normalized 15D hand geometry.

If open/pinch/power-like poses from both hands become geometrically similar, continue to LEAP/Fourier/Inspire.

---

## 14. Never hide invalid geometry

If the exact hand model is missing:

```text
native state = valid
canonical fingertips = invalid
```

This is scientifically preferable to guessed geometry.
