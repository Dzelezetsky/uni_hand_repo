"""Inspect the mimic-video checkpoints and the video backbone's hidden states (run with the mimic-video venv).

  cd third_party/mimic_video/model
  .venv/bin/python ../../../stage1/scripts/inspect_mimic.py [--forward] [--latent-frames 16] [--height 480 --width 640]

Part 1 (CPU): parameter counts / key groups / shapes of the video DiT and the Bridge action decoder, which inputs the
decoder expects (action, state, cross-attention context dims).
Part 2 (--forward, GPU): build Video2WorldPipeline from mimic-video's 2B/480p config with the pretrained Cosmos DiT
(no T5: zero language embedding), run ONE denoise pass on a random latent at video noise sigma and return the hidden
states up to layer 20 — the exact call World2ActionModel.get_crossattn_emb makes. Prints shapes, timing, peak memory.
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict

import torch

CKPT = "checkpoints"
VIDEO = f"{CKPT}/video_backbone/v2w_pretrained_cosmos.pt"
DECODER = f"{CKPT}/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt"


def summarize(path, name):
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        print(f"[{name}] top-level keys: {list(sd.keys())}")
        sd = sd["model"]
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    other = {k: v for k, v in sd.items() if not torch.is_tensor(v)}
    groups = Counter()
    for k, v in tensors.items():
        g = re.sub(r"\.(\d+)\.", ".N.", k).split(".")
        groups[".".join(g[:3])] += v.numel()
    total = sum(v.numel() for v in tensors.values())
    print(f"\n[{name}] {path}\n  tensors {len(tensors)}, params {total / 1e6:.1f} M, dtypes "
          f"{Counter(str(v.dtype) for v in tensors.values())}")
    if other:
        print(f"  non-tensor entries: { {k: type(v).__name__ for k, v in list(other.items())[:10]} }")
    for g, n in groups.most_common(25):
        print(f"  {g:60s} {n / 1e6:8.2f} M")
    blocks = defaultdict(int)
    for k in tensors:
        m = re.search(r"blocks\.(\d+)\.", k)
        if m:
            blocks[int(m.group(1))] += 1
    if blocks:
        print(f"  blocks: {len(blocks)} (indices {min(blocks)}..{max(blocks)})")
    return tensors


def part1():
    v = summarize(VIDEO, "video DiT")
    for k in sorted(v):
        if k.split(".")[0] in ("net", "") and ("blocks.20." in k or not "blocks." in k):
            if "blocks.20." in k and not any(s in k for s in ("q_proj", "k_proj", "cross_attn", "mlp.layer1")):
                continue
            print(f"    {k:70s} {tuple(v[k].shape)}")
    d = summarize(DECODER, "Bridge action decoder")
    for k in sorted(d):
        if "blocks." not in k or "blocks.0." in k:
            print(f"    {k:70s} {tuple(d[k].shape)}")


def part2(args):
    from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

    cfg = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    cfg.guardrail_config.enabled = False
    t0 = time.time()
    pipe = Video2WorldPipeline.from_config(cfg, dit_path=VIDEO, use_text_encoder=False)
    print(f"\npipeline built in {time.time() - t0:.0f}s; DiT params "
          f"{sum(p.numel() for p in pipe.dit.parameters()) / 1e6:.0f} M; state_t {cfg.state_t}")
    B, T, H, W = 1, args.latent_frames, args.height // 8, args.width // 8
    lat = torch.randn(B, 16, T, H, W, device="cuda")
    batch = {"obs/workspace_rgb_embedding": lat, "obs/num_conditional_frames": torch.tensor([2]),
             "obs/language_embedding": torch.zeros(B, 512, 1024, device="cuda", dtype=torch.bfloat16)}
    latent, condition = pipe.get_mimic_data_and_condition(batch)
    sigma = torch.full((B, 1), 1.0, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        t0 = time.time()
        out = pipe.denoise(latent + torch.randn_like(latent) * sigma[:, :, None, None, None].transpose(1, 2),
                           sigma, condition, return_only_hidden_states_up_to=args.layer)
        torch.cuda.synchronize()
    hs = out.hidden_states
    h = hs[args.layer] if isinstance(hs, (list, tuple, dict)) else hs
    print(f"hidden_states type {type(hs).__name__}, len {len(hs) if hasattr(hs, '__len__') else '-'}; "
          f"layer {args.layer} shape {tuple(h.shape)} dtype {h.dtype}")
    print(f"denoise (up to layer {args.layer}) {time.time() - t0:.2f}s, peak GPU memory "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--latent-frames", type=int, default=16)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--layer", type=int, default=20)
    a = ap.parse_args()
    part1()
    if a.forward:
        part2(a)
