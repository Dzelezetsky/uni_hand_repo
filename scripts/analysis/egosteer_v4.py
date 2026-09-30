"""EgoSteer V4: project the FK fingertips (state[44:74], head-camera frame = world; reproduced exactly by
egosteer_fk_check.py) and wrist positions into the head and chest RGB frames at the same timestamp.
usage: egosteer_v4.py <episode_index> [n_frames]    output: unified/validation/v4/egosteer__ep<idx>.png"""
import sys

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = "raw_data/egosteer"
COL = ["red", "orange", "lime", "cyan", "magenta"]  # thumb..pinky


def frame(path, t):
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 1.0, 0) / s.time_base), stream=s)
        for fr in c.decode(s):
            if float(fr.pts * s.time_base) >= t - 1e-3:
                return fr.to_ndarray(format="rgb24")


def main(ep, n):
    E = pd.concat(pd.read_parquet(f"{ROOT}/meta/episodes/chunk-000/file-{i:03d}.parquet") for i in range(6))
    e = E[E.episode_index == ep].iloc[0]
    d = pd.read_parquet(f"{ROOT}/data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet")
    d = d[d.episode_index == ep]
    S, w2c_chest = np.stack(d["observation.state"].to_numpy()), np.stack(d["observation.camera.chest_world2cam"].to_numpy())
    closure = S[:, [15, 16, 17, 21, 22, 23]].sum(1)
    ks = sorted(set(np.linspace(0, len(S) - 1, n - 1).astype(int)) | {int(np.argmax(closure))})
    fig, ax = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 6.6))
    for col, k in enumerate(ks):
        for row, cam in enumerate(("head", "chest")):
            K = np.array(e[f"calibration/{cam}_intrinsics"]).reshape(3, 3)
            T = np.eye(4) if cam == "head" else w2c_chest[k].reshape(4, 4)
            vid = f"{ROOT}/videos/observation.images.{cam}/chunk-{e[f'videos/observation.images.{cam}/chunk_index']:03d}/" \
                  f"file-{e[f'videos/observation.images.{cam}/file_index']:03d}.mp4"
            img = frame(vid, e[f"videos/observation.images.{cam}/from_timestamp"] + k / 30)
            a = ax[row, col]
            a.imshow(img)
            for tips, wr in ((S[k, 44:59], S[k, 26:29]), (S[k, 59:74], S[k, 35:38])):
                P = np.c_[np.r_[wr, tips].reshape(-1, 3), np.ones(6)] @ T.T
                uv = (P[:, :3] @ K.T)
                uv = uv[:, :2] / uv[:, 2:]
                for f in range(5):
                    a.plot(*uv[[0, f + 1]].T, "-", c=COL[f], lw=1)
                    a.plot(*uv[f + 1], "o", c=COL[f], ms=4, mec="k", mew=0.5)
            a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0); a.axis("off")
            a.set_title(f"{cam} f{k} t={k / 30:.1f}s  L{S[k, 14:20].round(2)}\nR{S[k, 20:26].round(2)}", fontsize=6)
    fig.suptitle(f"EgoSteer ep {ep}: {e['tasks'][0]} | lines wrist->tip (red thumb, orange index, lime middle, cyan ring, magenta pinky)", fontsize=8)
    fig.tight_layout()
    out = f"unified/validation/v4/egosteer__ep{ep:06d}.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5)


def crops(ep, ks, side="right", cam="chest", half=90):
    """Zoomed crops around one hand (V4 at finger level): raw crop | crop + projected wrist->tip lines."""
    E = pd.concat(pd.read_parquet(f"{ROOT}/meta/episodes/chunk-000/file-{i:03d}.parquet") for i in range(6))
    e = E[E.episode_index == ep].iloc[0]
    d = pd.read_parquet(f"{ROOT}/data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet")
    d = d[d.episode_index == ep]
    S, W = np.stack(d["observation.state"].to_numpy()), np.stack(d[f"observation.camera.{cam}_world2cam"].to_numpy())
    K = np.array(e[f"calibration/{cam}_intrinsics"]).reshape(3, 3)
    vid = f"{ROOT}/videos/observation.images.{cam}/chunk-{e[f'videos/observation.images.{cam}/chunk_index']:03d}/" \
          f"file-{e[f'videos/observation.images.{cam}/file_index']:03d}.mp4"
    tip_sl, wr_sl = (slice(59, 74), slice(35, 38)) if side == "right" else (slice(44, 59), slice(26, 29))
    fig, ax = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 6.6))
    for c, k in enumerate(ks):
        img = frame(vid, e[f"videos/observation.images.{cam}/from_timestamp"] + k / 30)
        P = np.c_[np.r_[S[k, wr_sl], S[k, tip_sl]].reshape(-1, 3), np.ones(6)] @ W[k].reshape(4, 4).T
        uv = P[:, :3] @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        cx, cy = uv[1:].mean(0)
        for r in (0, 1):
            a = ax[r, c]
            a.imshow(img)
            if r:
                for f in range(5):
                    a.plot(*uv[[0, f + 1]].T, "-", c=COL[f], lw=1.5)
                    a.plot(*uv[f + 1], "o", c=COL[f], ms=6, mec="k")
            a.set_xlim(cx - half, cx + half); a.set_ylim(cy + half, cy - half); a.axis("off")
        h = S[k, tip_sl.start - 30 - (0 if side == "right" else 0):][:0]
        ax[0, c].set_title(f"f{k} {side} hand {S[k, 20:26] if side == 'right' else S[k, 14:20]}".replace("\n", ""), fontsize=6)
    out = f"unified/validation/v4/egosteer__ep{ep:06d}_{cam}_{side}_crops.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(out)
