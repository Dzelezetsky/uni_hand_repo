"""Local smoke test of Stage-1 training WITHOUT mimic-video's trainer/hydra (mimic venv, one GPU):
    source stage1/env.sh; cd $MIMIC_MODEL; $MIMIC_PY $UNIDEX_REPO/stage1/scripts/smoke_train.py --steps 4 --height 240 --width 320
Builds Stage1Model (pretrained Cosmos 2B + LoRA + hand head), runs a few optimizer steps on real Stage-1 samples and
checks: finite losses, gradients on LoRA and hand-head parameters (none on frozen base weights), peak memory, s/step.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(os.environ["UNIDEX_REPO"])
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--lora-rank", type=int, default=8)
ap.add_argument("--hand-mode", default="clean")
ap.add_argument("--t5-subdir", default="t5_TESTONLY")
ap.add_argument("--video-loss-scale", type=float, default=100.0,
                help="0 -> any LoRA gradient must come from the hand loss (checks the joint objective)")
ap.add_argument("--truncate-blocks", type=int, default=0,
                help="keep only the first N DiT blocks and read the hand head from block N (12 GB GPUs; code-path test)")
a = ap.parse_args()

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline  # noqa: E402
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig  # noqa: E402

from stage1.dataset import Stage1Dataset  # noqa: E402
from stage1.model import Stage1Model, Stage1ModelConfig  # noqa: E402

pipe_cfg = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
pipe_cfg.guardrail_config.enabled = False
cfg = Stage1ModelConfig(
    pipe_config=pipe_cfg,
    model_manager_config=Predict2ModelManagerConfig(dit_path="checkpoints/video_backbone/v2w_pretrained_cosmos.pt",
                                                    text_encoder_path=""),
    train_architecture="lora", lora_rank=a.lora_rank, lora_alpha=a.lora_rank,
    lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
    fsdp_shard_size=0, high_sigma_ratio=0.05, loss_scale=a.video_loss_scale, hand_mode=a.hand_mode, hand_fraction=0.5,
    hand_layer=a.truncate_blocks or 20)
model = Stage1Model(cfg)
if a.truncate_blocks:
    model.net.blocks = torch.nn.ModuleList(list(model.net.blocks)[: a.truncate_blocks])
    print(f"DiT truncated to {len(model.net.blocks)} blocks (smoke test only)")
model.on_train_start()
params = [p for p in model.net.parameters() if p.requires_grad]
print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f} M "
      f"(hand head {sum(p.numel() for p in model.net.hand_head.parameters()) / 1e6:.1f} M)")
opt = torch.optim.AdamW(params, lr=1e-4)
ds = Stage1Dataset(REPO / "stage1_data", REPO, t5_subdir=a.t5_subdir, val_ratio=0.0, seed=0, size=(a.height, a.width))
dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=2)
torch.cuda.reset_peak_memory_stats()
it = iter(dl)
for step in range(a.steps):
    batch = next(it)
    batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    t0 = time.time()
    out, loss = model.training_step(batch, step)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    lora_g = [p.grad.norm().item() for n, p in model.net.named_parameters() if "lora" in n and p.grad is not None]
    head_g = [p.grad.norm().item() for p in model.net.hand_head.parameters() if p.grad is not None]
    frozen_g = [n for n, p in model.net.named_parameters() if not p.requires_grad and p.grad is not None]
    opt.step()
    torch.cuda.synchronize()
    keys = ["loss", "video_loss", "hand/loss", "hand/acc", "hand/acc_copy_current", "hand/geo_err_pw", "hand/n"]
    print(f"step {step}: " + " ".join(f"{k}={out[k]:.3f}" for k in keys if k in out) +
          f" | grad LoRA {len(lora_g)} tensors (mean norm {sum(lora_g) / max(len(lora_g), 1):.2e}), "
          f"head {len(head_g)} tensors, frozen-with-grad {len(frozen_g)} | {time.time() - t0:.1f}s", flush=True)
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB at {a.height}x{a.width}, batch {a.batch}")
