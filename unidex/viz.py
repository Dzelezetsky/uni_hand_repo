"""Visual validation: canonical hand skeleton (from FK) next to the RGB frame at the same time."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import av
import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .hands import REPO, load_hand  # noqa: E402
from .kinematics.canonical import FINGERS, to_canonical  # noqa: E402

UNI = REPO / "unified"
COLORS = dict(thumb="tab:red", index="tab:orange", middle="tab:green", ring="tab:blue", pinky="tab:purple")


def skeleton(hand, model_q, names):
    """{finger: (T, K, 3)} canonical coordinates of link origins from palm to tip (+ tip offset point)."""
    out = {}
    for f, tip, off in zip(FINGERS, hand.tip_links, hand.tip_offsets):
        chain = hand.fk.chain(tip)
        links = [j.child for j in chain]
        P = hand.fk.link_poses(model_q, names, links)
        pts = [P[l][:, :3, 3] for l in links]
        pts.append(P[tip][:, :3, 3] + np.einsum("tij,j->ti", P[tip][:, :3, :3], off))
        first = hand.fk.joints[chain[0].name]
        base = np.broadcast_to(first.origin[:3, 3], pts[0].shape)
        arr = np.stack([base] + pts, axis=1)
        out[f] = to_canonical(arr, hand.frame, hand.side)
    return out


def read_frame(cam_row, frame_index):
    ref = cam_row.rgb_ref
    if ref.startswith("zip://"):
        zpath, member = ref[len("zip://"):].split("!")
        z = zipfile.ZipFile(zpath)
        names = sorted((n for n in z.namelist() if n.startswith(member) and n.endswith((".png", ".jpg"))),
                       key=lambda n: int(Path(n).stem) if Path(n).stem.isdigit() else Path(n).stem)
        img = cv2.imdecode(np.frombuffer(z.read(names[min(frame_index, len(names) - 1)]), np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if ref.startswith("hdf5://"):
        import h5py
        fpath, grp = ref[len("hdf5://"):].split("!")
        g = h5py.File(fpath, "r", locking=False)[grp.rstrip("/")]
        if isinstance(g, h5py.Dataset):  # RoboMIND: dataset of encoded frames
            buf = np.asarray(g[min(frame_index, len(g) - 1)], np.uint8)
            return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        names = sorted(g.keys())
        a = np.asarray(g[names[min(frame_index, len(names) - 1)]][()], np.uint8)
        if a.ndim == 3:  # DexWild stores decoded frames (H, W, 3), channel order as recorded (BGR, cv2 pipeline)
            return a[..., ::-1]
        return cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with av.open(ref) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == frame_index:
                return fr.to_ndarray(format="rgb24")
    return None


def _nearest(t_arr, t):
    return int(np.nanargmin(np.abs(t_arr - t)))


def panel(dataset_id, episode_key, side="right", n=4, camera_id=None, out=None, times=None, pad_normals=False):
    """pad_normals=True: also draw fingertip pad normals (20 mm red arrows) in both skeleton views."""
    hs = pd.read_parquet(UNI / "hand_streams.parquet")
    row = hs[(hs.dataset_id == dataset_id) & (hs.source_episode_id == episode_key) & (hs.side == side)].iloc[0]
    frame = pd.read_parquet(UNI / row.path)
    hand = load_hand(row.hand_model_id)
    names = list(row.model_joint_names)
    cams = pd.read_parquet(UNI / "cameras.parquet")
    cams = cams[(cams.dataset_id == dataset_id) & (cams.source_episode_id == episode_key) & cams.local]
    cam = cams[cams.camera_id == camera_id].iloc[0] if camera_id else cams.iloc[0]
    ep_dir = UNI / Path(row.path).parent
    cf = pd.read_parquet(ep_dir / "camera_frames.parquet") if (ep_dir / "camera_frames.parquet").exists() else None
    tips = np.stack(frame.fingertips_palm_m.to_numpy())
    closure = np.linalg.norm(tips.reshape(-1, 5, 3)[:, 1:, :2], axis=2).mean(1)  # smaller = more flexed
    if times is None:
        idx = np.unique(np.r_[np.argmax(closure), np.argmin(closure),
                              np.linspace(0, len(frame) - 1, max(n - 2, 1)).astype(int)])[:n]
    else:
        idx = [_nearest(frame.t_s.to_numpy(), t) for t in times]
    q = np.stack(frame.model_q.to_numpy())[idx]
    sk = skeleton(hand, q, names)
    nrm = hand.canonical_pad_normals(q, names) if pad_normals else None
    fig, axes = plt.subplots(3, len(idx), figsize=(4 * len(idx), 11))
    axes = np.atleast_2d(axes).reshape(3, len(idx))
    for k, i in enumerate(idx):
        t = frame.t_s.iloc[i]
        img, fi = None, None
        if cf is not None and (cf.camera_id == cam.camera_id).any():
            c1 = cf[cf.camera_id == cam.camera_id]
            fi = int(c1.frame_index.iloc[_nearest(c1.t_s.to_numpy(), t)])
        elif dataset_id in ("realdex", "robomind_tienkung"):
            fi = i  # hand rows are one-to-one with RGB frames
        if fi is not None:
            img = read_frame(cam, fi)
        ax = axes[0, k]
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"t={t:.2f}s row={i} frame={fi}", fontsize=9); ax.axis("off")
        for r, (a, b, lab) in enumerate([(0, 1, "front: X (pinky->index) / Y (wrist->fingers)"),
                                         (2, 1, "side: Z (palm normal) / Y")]):
            ax = axes[r + 1, k]
            for f in FINGERS:
                p = sk[f][k] * 1000
                ax.plot(p[:, a], p[:, b], "-o", color=COLORS[f], ms=3, lw=2, label=f)
                if nrm is not None:
                    d = nrm[k, FINGERS.index(f)] * 20
                    ax.annotate("", xy=(p[-1, a] + d[a], p[-1, b] + d[b]), xytext=(p[-1, a], p[-1, b]),
                                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
            lm = hand.frame.landmarks
            base = to_canonical(np.stack([lm[f"base_{f}"] for f in ("index", "middle", "ring", "pinky")]),
                                hand.frame, hand.side) * 1000
            ax.plot(base[:, a], base[:, b], "k--", lw=1)
            ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlim(-120, 180) if a == 0 else ax.set_xlim(-120, 120)
            ax.set_ylim(-140, 200); ax.set_title(lab, fontsize=8)
            if k == 0 and r == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{dataset_id} | {episode_key} | {side} | {row.hand_model_id} | {row.state_source} | cam {cam.camera_id}")
    fig.tight_layout()
    out = Path(out or UNI / "validation" / f"{dataset_id}__{episode_key.replace('/', '__')}__{side}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out
