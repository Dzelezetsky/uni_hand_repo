"""V4 visual check for a (dataset, hand) mapping BEFORE it is verified: camera frame next to a mesh render of the hand
pose obtained by mapping + FK at the same timestamp. Works directly on the adapter output (nothing is stored).

usage: python scripts/analysis/check_v4.py <dataset_id> <camera_id> <episode_id>[@t1,t2,...] [...]
  times default: 4 frames spread over the episode, including the most-closed hand moment.
output: unified/validation/v4/<dataset>__<episode>.png
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
from unidex.convert import adapters  # noqa: E402
from unidex.hands import REPO, load_hand  # noqa: E402
from unidex.mappings import MAPPINGS  # noqa: E402
from unidex.viz_mesh import draw_mesh, hand_triangles  # noqa: E402

OUT = REPO / "unified" / "validation" / "v4"


def frame_at(cam, t):
    import av
    i = int(np.argmin(np.abs(cam.frame_t - t))) + cam.frame_index_offset
    with av.open(cam.rgb_ref) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        c.seek(int(max(i / fps - 1.0, 0) / s.time_base), stream=s)  # seek to a keyframe before, then decode forward
        for fr in c.decode(s):
            if int(round(float(fr.pts * s.time_base) * fps)) >= i:
                return fr.to_ndarray(format="rgb24"), i
    return None, i


def main(ds, cam_id, specs):
    mod = adapters()[ds]
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        eid, _, ts = spec.partition("@")
        e = mod.load_episode(eid)
        cam = next(c for c in e.cameras if c.camera_id == cam_id)
        hands = [h for h in e.hands if h.mapping_id and h.status_override is None]
        mapped = {h.side: (MAPPINGS[h.mapping_id], h) for h in hands}
        if ts:
            times = [float(x) for x in ts.split(",")]
        else:
            h0 = hands[0]
            q = MAPPINGS[h0.mapping_id](h0.native_q)
            tips = load_hand(h0.hand_model_id).canonical(q, list(MAPPINGS[h0.mapping_id].model_joints))[1]
            closure = np.linalg.norm(tips[:, 1:, :2], axis=2).mean(1)
            times = sorted({float(h0.t[0] + 0.5), float(h0.t[int(np.argmin(closure))]),
                            float(h0.t[len(h0.t) // 2]), float(h0.t[-1] - 0.5)})
        fig, axes = plt.subplots(1 + 2 * len(mapped), len(times), figsize=(4.2 * len(times), 3.4 * (1 + 2 * len(mapped))))
        axes = np.atleast_2d(axes).reshape(1 + 2 * len(mapped), len(times))
        for k, t in enumerate(times):
            img, fi = frame_at(cam, t)
            ax = axes[0, k]
            if img is not None:
                ax.imshow(img)
            ax.set_title(f"t={t:.2f}s  video frame {fi}", fontsize=8); ax.axis("off")
            for j, (side, (m, h)) in enumerate(sorted(mapped.items())):
                i = int(np.argmin(np.abs(h.t - t)))
                q = m(h.native_q[i:i + 1])
                hand = load_hand(h.hand_model_id)
                for v, (view, up, lab) in enumerate([(np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), "palm view"),
                                                     (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), "side view")]):
                    ax = axes[1 + 2 * j + v, k]
                    draw_mesh(ax, hand_triangles(h.hand_model_id, hand, q, list(m.model_joints)), view, up)
                    ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xticks([]); ax.set_yticks([])
                    if k == 0:
                        ax.set_ylabel(f"{side} {lab}", fontsize=9)
        fig.suptitle(f"V4 check | {ds} | {eid} | cam {cam_id} | {e.instruction_original}", fontsize=10)
        fig.tight_layout()
        out = OUT / f"{ds}__{eid.replace('/', '__')}.png"
        fig.savefig(out, dpi=60); plt.close(fig)
        print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
