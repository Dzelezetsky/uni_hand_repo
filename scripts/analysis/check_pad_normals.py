"""Manual-check renders for fingertip pad normals (PAD_NORMAL_VERSION).

Per hand model -> unified/validation/pad_normals/<hand_model_id>.png
  row 1: whole hand seen from the palm side (canonical +Z towards the viewer), poses rest / half / near-full flexion
  row 2: whole hand seen from the radial side (index side, looking along -X)
  row 3: per finger, distal + previous phalanx at half flexion, viewed along the distal flexion axis
         (the concave side of the bent finger is the pad side)
Red arrows = pad normals (25 mm), black dot = fingertip point.

What to check by eye: in every panel each arrow must leave the finger through the PAD (palmar skin side), never the
nail/back side, and for the four fingers it must point towards the palm when flexed.

Also prints the rule-consistency table: for index..pinky the flexion-sign rule (joint limits) must give a palmar
(+Z) normal at rest; this is an internal check, the visual one is the acceptance criterion.

Usage: python scripts/analysis/check_pad_normals.py [hand_model_id ...]
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from unidex.hands import REPO, load_hand, registry  # noqa: E402
from unidex.kinematics.canonical import FINGERS  # noqa: E402
from unidex.viz import COLORS  # noqa: E402
from unidex.viz_mesh import draw_mesh, hand_triangles, link_meshes, project  # noqa: E402

OUT = REPO / "unified" / "validation" / "pad_normals"
ARROW_M = 0.025


def pose(hand, frac):
    """Every actuated joint at `frac` of the way from 0 (clipped into limits) towards its larger-magnitude limit."""
    names = hand.fk.actuated_joints
    q = []
    for n in names:
        j = hand.fk.joints[n]
        lo, hi = (j.lower if j.lower is not None else -1.0), (j.upper if j.upper is not None else 1.0)
        start = min(max(0.0, lo), hi)
        end = hi if abs(hi) >= abs(lo) else lo
        q.append(start + frac * (end - start))
    return np.array(q)[None], names


def arrows(ax, tips, normals, right, up2, fingers=range(5)):
    for i in fingers:
        a = project(tips[i], right, up2)
        b = project(tips[i] + ARROW_M * normals[i], right, up2)
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="red", lw=2), zorder=10)
        ax.plot(*a, "o", color="k", ms=4, zorder=11)
        ax.text(*b, FINGERS[i][:2], color=COLORS[FINGERS[i]], fontsize=8, fontweight="bold", zorder=12)


def finish(ax, title, pts=None):
    ax.set_aspect("equal")
    ax.autoscale_view()
    if pts is not None:  # make sure arrow heads are inside the frame
        lo, hi = pts.min(0) - 10, pts.max(0) + 10
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(min(x0, lo[0]), max(x1, hi[0])); ax.set_ylim(min(y0, lo[1]), max(y1, hi[1]))
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2)


def render(hid):
    hand = load_hand(hid)
    specs = hand.pad_normal_specs()
    if not link_meshes(hid):
        print(f"{hid}: no meshes resolved, skipped")
        return None
    fracs = (0.0, 0.5, 0.9)
    fig, axes = plt.subplots(3, 5, figsize=(22, 14))
    views = [(np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), "palm side (+Z to viewer), X right = index"),
             (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), "radial side (from index side), +Z (palmar) to the LEFT")]
    for r, (view, up, vlab) in enumerate(views):
        for c, fr in enumerate(fracs):
            ax = axes[r, c]
            q, names = pose(hand, fr)
            tips = hand.canonical(q, names)[0][0]
            nrm = hand.canonical_pad_normals(q, names)[0]
            right, up2 = draw_mesh(ax, hand_triangles(hid, hand, q, names), view, up)
            arrows(ax, tips, nrm, right, up2)
            pts = np.r_[project(tips, right, up2), project(tips + ARROW_M * nrm, right, up2)]
            finish(ax, f"{vlab}\nflex fraction {fr}", pts)
        axes[r, 3].axis("off"); axes[r, 4].axis("off")
    # per-finger close-ups along the distal flexion axis
    q, names = pose(hand, 0.5)
    tips = hand.canonical(q, names)[0][0]
    nrm = hand.canonical_pad_normals(q, names)[0]
    for i, (f, s) in enumerate(zip(FINGERS, specs)):
        ax = axes[2, i]
        j = hand.fk.joints[s["joint"]]
        P = hand.fk.link_poses(q, names, [j.child])[j.child][0]
        axis_c = P[:3, :3] @ (j.axis / np.linalg.norm(j.axis)) @ hand.frame.R
        if hand.side == "left":
            axis_c = axis_c * np.array([1, 1, -1.0])
        chain = hand.fk.chain(hand.tip_links[i])
        k = [jj.name for jj in chain].index(j.name)
        links = [j.parent, j.child] + [jj.child for jj in chain[k + 1:]]
        prev = hand_triangles(hid, hand, q, names, [j.parent])
        dist = hand_triangles(hid, hand, q, names, links[1:])
        up = np.cross(axis_c, nrm[i])  # distal long axis roughly vertical in the image
        right, up2 = draw_mesh(ax, prev, axis_c, up, color=(0.55, 0.65, 0.85))
        draw_mesh(ax, dist, axis_c, up, color=(0.8, 0.8, 0.8))
        arrows(ax, tips, nrm, right, up2, [i])
        pts = np.r_[project(tips[i:i + 1], right, up2), project(tips[i:i + 1] + ARROW_M * nrm[i:i + 1], right, up2)]
        finish(ax, f"{f}: {s['joint']}\nsign {s['sign']:+.0f}, limits [{s['lower']:.2f}, {s['upper']:.2f}], "
                   f"half flex; blue = {j.parent}", pts)
    # text box: normals at rest
    q0, names = pose(hand, 0.0)
    n0 = hand.canonical_pad_normals(q0, names)[0]
    txt = "\n".join(f"{f:6s} n_rest = ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})" for f, v in zip(FINGERS, n0))
    axes[0, 3].text(0, 1, f"{hid}\n{registry()['hand_models'][hid]['model_status']}\n\n{txt}",
                    family="monospace", fontsize=9, va="top", transform=axes[0, 3].transAxes)
    fig.suptitle(f"pad normals check | {hid} | red arrow must leave the finger PAD", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{hid}.png"
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out, n0


def main(ids):
    ids = ids or list(registry()["hand_models"])
    print(f"{'hand_model_id':32s} four fingers palmar (+Z) at rest | thumb n_rest")
    for hid in ids:
        res = render(hid)
        if res is None:
            continue
        out, n0 = res
        ok = bool((n0[1:, 2] > 0.5).all())
        print(f"{hid:32s} {'OK ' if ok else 'FAIL'} min z={n0[1:, 2].min():+.2f} | thumb "
              f"({n0[0, 0]:+.2f},{n0[0, 1]:+.2f},{n0[0, 2]:+.2f})  -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1:])
