"""Stage-1 dataset: video windows for mimic-video / Cosmos-Predict2 video finetuning + future-hand targets.

Returns exactly the keys of mimic-video's video Dataset (cosmos_predict2/data/dataset_video.py) so its
Video2World training code runs unchanged, plus `hand/*` targets:
  video                  uint8 [3, 61, 480, 640]  frames at t + (k - 4) / FPS, k = 0..60 (5 past incl. t, 56 future)
  obs/language_embedding float [512, 1024]        T5-11B embedding of the instruction (stage1_data/t5/<sha>.safetensors)
  t5_text_mask           int64 [512]
  fps                    float                    10
  padding_mask           float [1, 480, 640]      1 = letterbox padding (16:9 sources are fitted, not stretched)
  num_conditional_frames int64                    2 latent frames (= 5 video frames)
  hand/geo_cur           float [2, 15]            canonical fingertips (palm widths) at t, sides (left, right)
  hand/geo_fut           float [2, 14, 15]        at the end of each future latent frame: t + 0.4, 0.8, ... 5.6 s
  hand/cls_cur           int64 [2]                hand_posture_class_v1 at t (-1 = unknown)
  hand/cls_fut           int64 [2, 14]
  hand/mask_cur          bool  [2]
  hand/mask_fut          bool  [2, 14]            False: side not verified / outside the hand stream / invalid sample
  hand/family            int64 [2]                index into FAMILIES (-1 = no verified hand on that side)
Time-based sampling (CLAUDE.md §12): frames are chosen by the stored per-frame timestamps, never by index arithmetic.
Latent/time mapping of the Wan-style tokenizer: latent 0 = frame 0, latent m >= 1 = frames 4m-3 .. 4m; the hand target
of latent m is taken at the time of its last frame 4m, i.e. t + 0.4 (m - 1) s at 10 fps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from safetensors.numpy import load_file

FAMILIES = ["Shadow", "DFX", "F1", "XHand", "Sharpa", "Ruiyan"]
SIDES = ("left", "right")
NUM_FRAMES, OBS_HISTORY, FPS, H, W = 61, 5, 10.0, 480, 640
T5_TOKENS, T5_DIM = 512, 1024


def latent_target_offsets(num_frames=NUM_FRAMES, obs_history=OBS_HISTORY, fps=FPS):
    """Seconds after t (the last observed frame) of the end of every FUTURE latent frame."""
    n_lat = 1 + (num_frames - 1) // 4
    n_cond = 1 + (obs_history - 1) // 4
    return np.array([(4 * m - (obs_history - 1)) / fps for m in range(n_cond, n_lat)])


class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, repo: str | Path, datasets: list[str] | None = None, is_val: bool = False,
                 val_ratio: float = 0.01, allow_dummy_t5: bool = False, seed: int | None = None,
                 t5_subdir: str = "t5", size: tuple[int, int] = (H, W)):
        self.root, self.repo = Path(root), Path(repo)
        rows = [json.loads(l) for l in (self.root / "index.jsonl").read_text().splitlines()]
        if datasets:
            rows = [r for r in rows if r["dataset"] in datasets]
        key = lambda r: int.from_bytes(f"{r['dataset']}/{r['episode']}".encode()[-8:].ljust(8, b"0"), "little")
        rows = [r for r in rows if ((key(r) * 2654435761) % 10000 < val_ratio * 10000) == is_val]
        self.rows = rows
        self.allow_dummy_t5 = allow_dummy_t5
        self.t5_subdir = t5_subdir
        self.h, self.w = size  # (480, 640) for training; smaller only for local smoke tests
        self.offsets = latent_target_offsets()
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def episode_weights(self, dataset_weights: dict[str, float] | None = None) -> np.ndarray:
        """Sampling weight per episode: every DATASET gets equal total probability (or `dataset_weights`), shared
        by its episodes in proportion to their duration (so every second of a dataset is equally likely)."""
        dur = np.array([max(r["t1"] - r["t0"], 0.1) for r in self.rows])
        ds = np.array([r["dataset"] for r in self.rows])
        w = np.zeros(len(self.rows))
        for d in np.unique(ds):
            m = ds == d
            w[m] = dur[m] / dur[m].sum() * (dataset_weights or {}).get(d, 1.0)
        return w / w.sum()

    # ------------------------------------------------------------------ video
    def _frames(self, r, ep, t):
        times = t + (np.arange(NUM_FRAMES) - (OBS_HISTORY - 1)) / FPS
        ft = ep["frame_t"]
        idx = np.clip(np.searchsorted(ft, times), 1, len(ft) - 1)
        idx -= (times - ft[idx - 1]) < (ft[idx] - times)          # nearest frame in time (pads by repeating ends)
        uniq, inv = np.unique(idx, return_inverse=True)
        vr = VideoReader(str(self.repo / r["rgb_ref"]), ctx=cpu(0), num_threads=0)
        x = torch.from_numpy(vr.get_batch((uniq + int(ep["frame_offset"])).tolist()).asnumpy())  # [U,h,w,3]
        del vr
        x = x.permute(3, 0, 1, 2).float()                                                # [3,U,h,w]
        h, w = x.shape[-2:]
        HH, WW = self.h, self.w
        s = min(HH / h, WW / w)
        nh, nw = round(h * s), round(w * s)
        x = F.interpolate(x.reshape(-1, 1, h, w), size=(nh, nw), mode="bilinear", align_corners=False,
                          antialias=True).reshape(3, -1, nh, nw)
        top, left = (HH - nh) // 2, (WW - nw) // 2
        out = torch.zeros(3, x.shape[1], HH, WW)
        out[:, :, top:top + nh, left:left + nw] = x
        pad = torch.ones(1, HH, WW)
        pad[:, top:top + nh, left:left + nw] = 0
        video = out.round().clamp_(0, 255).to(torch.uint8)[:, torch.from_numpy(inv)]
        return video, pad

    # ------------------------------------------------------------------ hand
    def _hand(self, ep, side, times):
        if f"{side}_t" not in ep:
            return (np.zeros((len(times), 15), np.float32), np.full(len(times), -1), np.zeros(len(times), bool))
        ht, geo, valid, cls = ep[f"{side}_t"], ep[f"{side}_geo"], ep[f"{side}_valid"].astype(bool), ep[f"{side}_cls"]
        i = np.clip(np.searchsorted(ht, times), 1, len(ht) - 1)
        a, b = i - 1, i
        wgt = np.clip((times - ht[a]) / np.maximum(ht[b] - ht[a], 1e-9), 0, 1)[:, None]
        g = (1 - wgt) * geo[a] + wgt * geo[b]
        near = np.where(wgt[:, 0] < 0.5, a, b)
        ok = (times >= ht[0]) & (times <= ht[-1]) & valid[a] & valid[b] & np.isfinite(g).all(1)
        c = np.where(ok, cls[near], -1)
        return g.astype(np.float32), c.astype(np.int64), ok & (c >= 0)

    # ------------------------------------------------------------------ text
    def _t5(self, sha):
        p = self.root / self.t5_subdir / f"{sha}.safetensors"
        emb = np.zeros((T5_TOKENS, T5_DIM), np.float32)
        mask = np.zeros(T5_TOKENS, np.int64)
        if p.exists():
            e = load_file(str(p))["encoded_text"].astype(np.float32)[:T5_TOKENS]
            emb[: len(e)], mask[: len(e)] = e, 1
        elif not self.allow_dummy_t5:
            raise FileNotFoundError(f"missing T5 embedding {p} (run stage1/precompute_t5.py)")
        return torch.from_numpy(emb), torch.from_numpy(mask)

    def sample(self, index: int, t: float | None = None) -> dict:
        r = self.rows[index]
        ep = load_file(str(self.root / r["file"]))
        if t is None:
            t = float(self.rng.uniform(r["t0"], r["t1"]))
        video, pad = self._frames(r, ep, t)
        times = np.r_[t, t + self.offsets]
        geo, cls, mask, fam = [], [], [], []
        for side in SIDES:
            g, c, m = self._hand(ep, side, times)
            geo.append(g); cls.append(c); mask.append(m)
            fam.append(r["families"][r["sides"].index(side)] if side in r["sides"] else -1)
        geo, cls, mask = np.stack(geo), np.stack(cls), np.stack(mask)
        emb, tmask = self._t5(r["instruction_sha"])
        return {
            "video": video, "obs/language_embedding": emb, "t5_text_mask": tmask, "fps": FPS, "padding_mask": pad,
            "num_conditional_frames": torch.tensor(1 + (OBS_HISTORY - 1) // 4),
            "hand/geo_cur": torch.from_numpy(geo[:, 0]), "hand/geo_fut": torch.from_numpy(geo[:, 1:]),
            "hand/cls_cur": torch.from_numpy(cls[:, 0]), "hand/cls_fut": torch.from_numpy(cls[:, 1:]),
            "hand/mask_cur": torch.from_numpy(mask[:, 0]), "hand/mask_fut": torch.from_numpy(mask[:, 1:]),
            "hand/family": torch.tensor(fam), "meta/t": t, "meta/episode": f"{r['dataset']}/{r['episode']}",
        }

    def __getitem__(self, index: int) -> dict:
        return self.sample(index)
