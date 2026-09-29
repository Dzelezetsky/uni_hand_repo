import numpy as np
import pytest
import yourdfpy

from unidex.hands import REPO, load_hand, registry
from unidex.kinematics.canonical import palm_frame_from_landmarks, to_canonical
from unidex.mappings import MAPPINGS

HAND_IDS = list(registry()["hand_models"])


@pytest.mark.parametrize("hid", HAND_IDS)
def test_palm_frame_orthonormal_and_right_handed(hid):
    R = load_hand(hid).frame.R
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("hid", HAND_IDS)
def test_fk_matches_yourdfpy_reference(hid):
    """Our batched FK (incl. mimic joints) must equal yourdfpy's FK for random joint values."""
    spec = registry()["hand_models"][hid]
    h = load_hand(hid)
    ref = yourdfpy.URDF.load(str(REPO / spec["urdf"]), load_meshes=False, build_collision_scene_graph=False)
    names = list(ref.actuated_joint_names)
    rng = np.random.default_rng(0)
    lo = np.array([h.fk.joints[n].lower if h.fk.joints[n].lower is not None else -1 for n in names])
    hi = np.array([h.fk.joints[n].upper if h.fk.joints[n].upper is not None else 1 for n in names])
    for _ in range(3):
        q = lo + rng.random(len(names)) * (hi - lo)
        ref.update_cfg(q)
        ours = h.fk.link_poses(q[None], names, h.tip_links)
        palm = ref.get_transform(spec["palm_link"])
        for l in h.tip_links:
            expect = np.linalg.inv(palm) @ ref.get_transform(l)
            assert np.allclose(ours[l][0], expect, atol=1e-6), (hid, l)


def test_left_mirror_maps_reflected_right_hand_onto_right():
    rng = np.random.default_rng(1)
    w, bI, bM, bR, bP = rng.normal(size=(5, 3))
    pts = rng.normal(size=(7, 3))
    Rr = palm_frame_from_landmarks(w, bI, bM, bR, bP)
    M = np.diag([-1.0, 1.0, 1.0])  # any reflection turns a right hand into a left hand
    Rl = palm_frame_from_landmarks(*(v @ M for v in (w, bI, bM, bR, bP)))
    assert np.allclose(to_canonical(pts, Rr, "right"), to_canonical(pts @ M, Rl, "left"))


def test_same_hand_both_sides_agree_at_rest():
    """Unitree Inspire left/right URDFs should give (nearly) the same canonical rest geometry."""
    out = []
    for hid in ("inspire_rh56dfx_right_unitree", "inspire_rh56dfx_left_unitree"):
        h = load_hand(hid)
        n = h.fk.actuated_joints
        out.append(h.canonical(np.zeros((1, len(n))), n)[0][0])
    fingers = np.abs(out[0][1:] - out[1][1:]).max()
    thumb = np.abs(out[0][0] - out[1][0]).max()
    assert fingers < 1e-3, fingers
    assert thumb < 0.015, thumb  # known: left URDF thumb differs by ~1 cm


@pytest.mark.parametrize("mid", list(MAPPINGS))
def test_mapping_targets_exist_and_cover_finger_chains(mid):
    m = MAPPINGS[mid]
    h = load_hand(m.hand_model_id)
    assert all(j in h.fk.joints for j in m.model_joints)
    chains = {j.name for l in h.tip_links for j in h.fk.chain(l) if j.type != "fixed" and j.mimic_of is None}
    assert chains <= set(m.model_joints), chains - set(m.model_joints)
    out = m(np.zeros((4, len(m.raw_names))))
    assert out.shape == (4, len(m.model_joints))


def test_inspire_unitree_denormalization_endpoints():
    m = MAPPINGS["humanoid_everyday_h1__inspire_rh56dfx_right"]
    opened, closed = m(np.ones((1, 6)))[0], m(np.zeros((1, 6)))[0]
    assert np.allclose(opened, [0, 0, 0, 0, 0, -0.1])
    assert np.allclose(closed, [1.7, 1.7, 1.7, 1.7, 0.5, 1.3])


@pytest.mark.parametrize("hid", [h for h in HAND_IDS])
def test_flexion_moves_fingertips_to_palmar_side(hid):
    """Closing the four fingers must pull their tips down in Y and keep them on the palmar side (+Z) for every hand."""
    h = load_hand(hid)
    n = h.fk.actuated_joints
    lo = np.array([h.fk.joints[j].lower or 0 for j in n])
    hi = np.array([h.fk.joints[j].upper or 0 for j in n])
    q0 = np.clip(np.zeros(len(n)), lo, hi)
    flex = [j for j in n if any(k in j.lower() for k in ("proximal", "_1_joint", "joint1", "mcpf", "j3", "ffj3",
                                                          "mfj3", "rfj3", "lfj3"))
            and "thumb" not in j.lower() and "th" != j.lower()[3:5]]
    q1 = q0.copy()
    for j in flex:
        i = n.index(j)
        q1[i] = hi[i] if abs(hi[i]) >= abs(lo[i]) else lo[i]
    a, b = h.canonical(np.stack([q0, q1]), n)[0]
    moved = np.linalg.norm(b[1:] - a[1:], axis=1) > 0.01
    assert moved.any(), hid
    assert (b[1:, 1][moved] < a[1:, 1][moved]).all(), (hid, a[1:, 1], b[1:, 1])
    assert (b[1:, 2][moved] > -0.005).all(), (hid, b[1:, 2])


def test_only_verified_mappings_get_geometry():
    """Strict gate: excluded / unreviewed mappings must never produce fingertips."""
    import pandas as pd
    from unidex.convert import OUT
    from unidex.hands import mapping_verification
    hs = pd.read_parquet(OUT / "hand_streams.parquet")
    ok = hs[hs.canonicalization_status == "ok"]
    assert len(ok)
    assert all(mapping_verification(m)["status"] == "verified" for m in ok.mapping_id)
    for p in hs[hs.canonicalization_status != "ok"].path:
        assert "fingertips_palm_m" not in pd.read_parquet(OUT / p).columns, p


@pytest.mark.parametrize("hid", HAND_IDS)
def test_pad_normals_unit_perpendicular_and_palmar_at_rest(hid):
    h = load_hand(hid)
    n = h.fk.actuated_joints
    rng = np.random.default_rng(2)
    lo = np.array([h.fk.joints[j].lower for j in n]); hi = np.array([h.fk.joints[j].upper for j in n])
    q = np.r_[np.clip(np.zeros((1, len(n))), lo, hi), lo + rng.random((5, len(n))) * (hi - lo)]
    N = h.pad_normals_root(q, n)
    assert np.allclose(np.linalg.norm(N, axis=2), 1.0)
    poses = h.fk.link_poses(q, n, h.tip_links)
    for i, s in enumerate(h.pad_normal_specs()):  # normal is perpendicular to the distal flexion axis
        j = h.fk.joints[s["joint"]]
        a = h.fk.link_poses(q, n, [j.child])[j.child][:, :3, :3] @ (j.axis / np.linalg.norm(j.axis))
        assert np.abs(np.einsum("ti,ti->t", a, N[:, i])).max() < 1e-9
    assert (h.canonical_pad_normals(q[:1], n)[0, 1:, 2] > 0.5).all()  # four fingers: pads face the palm side


def test_pad_normals_left_right_agree_at_rest():
    out = []
    for hid in ("inspire_rh56dfx_right_unitree", "inspire_rh56dfx_left_unitree"):
        h = load_hand(hid)
        n = h.fk.actuated_joints
        out.append(h.canonical_pad_normals(np.zeros((1, len(n))), n)[0])
    assert np.abs(out[0][1:] - out[1][1:]).max() < 1e-3
    assert np.degrees(np.arccos(np.clip((out[0][0] * out[1][0]).sum(), -1, 1))) < 10  # left URDF thumb differs a bit


def test_reachability_self_fit_recovers_own_postures():
    """best_fit must reproduce postures the hand itself produced (solver sanity)."""
    from itertools import combinations

    from unidex.reachability import best_fit, sample
    pairs = list(combinations(range(5), 2))

    def pw(p):
        return np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in pairs], 1)

    from scipy.spatial import cKDTree
    mid = "humanoid_everyday_h1__inspire_rh56dfx_right"
    S = sample(mid, n=20000, seed=1)
    T = sample(mid, n=50, seed=2)  # independent postures of the same hand
    targets = pw(T["tips_norm"])
    nn = cKDTree(pw(S["tips_norm"])).query(targets)[1]  # same init scheme as scripts/analysis/reachability.py
    _, r = best_fit(mid, pw, targets, S["raw"][nn])
    rms = np.sqrt((r ** 2).mean(1))
    assert np.median(rms) < 0.005 and (rms < 0.05).mean() > 0.95
