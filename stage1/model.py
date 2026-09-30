"""Stage-1 model: Cosmos-Predict2 video LoRA finetuning (mimic-video) + future-hand head, L = L_video + lambda * L_hand.

The head reads the output of video-DiT block `hand_layer` (20 = the block mimic-video's action decoder uses) through a
forward hook, WITHOUT detaching (mimic-video detaches its hidden states; here the hand loss must reach the LoRA
weights). The head lives inside the DiT (`net.hand_head`) so mimic-video's optimizer, checkpointer and DDP wrapper
pick it up unchanged.

Leakage control (CLAUDE.md §18): in hand_mode "clean" a random `hand_fraction` of every batch is noised at
sigma = sigma_max (80, the video noise level at which mimic-video reads hidden states at inference, i.e. pure-noise
future latents), and the hand loss is computed ONLY on those samples; the video loss is computed on all samples as
usual. hand_mode "leaky" (ablation): hand loss on every sample at its training sigma.
"""
from __future__ import annotations

import attrs
import torch

from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from imaginaire.utils import log

from stage1.hand_head import HandHead

HAND_KEYS = ("hand/geo_cur", "hand/geo_fut", "hand/cls_cur", "hand/cls_fut", "hand/mask_cur", "hand/mask_fut",
             "hand/family")


@attrs.define(slots=False)
class Stage1ModelConfig(Predict2Video2WorldModelConfig):
    hand_layer: int = 20
    hand_weight: float = 1.0          # lambda
    hand_fraction: float = 0.5        # share of each batch used for the hand loss (clean mode)
    hand_mode: str = "clean"          # clean | leaky | off (video-only baseline, head not trained)
    hand_sigma: float = 80.0          # sigma_max of the video scheduler = inference read-out level
    hand_dim: int = 512
    hand_layers: int = 4
    hand_heads: int = 8
    hand_hyp: int = 2
    hand_w_res: float = 1.0
    hand_w_hyp: float = 0.1
    debug_truncate_blocks: int = 0    # >0: keep only the first N DiT blocks (local 12 GB smoke tests ONLY)


class Stage1Model(Predict2Video2WorldModel):
    def __init__(self, config: Stage1ModelConfig):
        super().__init__(config)
        if config.debug_truncate_blocks:
            assert config.hand_layer <= config.debug_truncate_blocks
            self.pipe.dit.blocks = torch.nn.ModuleList(list(self.pipe.dit.blocks)[: config.debug_truncate_blocks])
            log.warning(f"DEBUG: DiT truncated to {config.debug_truncate_blocks} blocks (smoke test only)")
        assert config.fsdp_shard_size == 0, "hand head is attached after construction: use DDP (fsdp_shard_size=0)"
        dit = self.pipe.dit
        dit.hand_head = HandHead(ctx_dim=dit.model_channels, dim=config.hand_dim, layers=config.hand_layers,
                                 heads=config.hand_heads, n_hyp=config.hand_hyp, w_res=config.hand_w_res,
                                 w_hyp=config.hand_w_hyp).to(device="cuda", dtype=torch.float32)
        dit.hand_head.requires_grad_(config.hand_mode != "off")
        if self.pipe.dit_ema is not None:
            self.pipe.dit_ema.hand_head = HandHead(ctx_dim=dit.model_channels, dim=config.hand_dim,
                                                   layers=config.hand_layers, heads=config.hand_heads,
                                                   n_hyp=config.hand_hyp).to(device="cuda", dtype=torch.float32)
            self.pipe.dit_ema.hand_head.load_state_dict(dit.hand_head.state_dict())
            self.pipe.dit_ema.hand_head.requires_grad_(False)
        self._hidden, self._want_hidden = None, False
        dit.blocks[config.hand_layer - 1].register_forward_hook(self._capture)
        n = sum(p.numel() for p in dit.hand_head.parameters())
        log.info(f"Stage-1 hand head: {n / 1e6:.1f} M params on block {config.hand_layer}, mode {config.hand_mode}")

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format)          # casts the whole DiT to bf16 ...
        self.net.hand_head.to(dtype=torch.float32)     # ... keep the (small) hand head in fp32
        if self.pipe.dit_ema is not None:
            self.pipe.dit_ema.hand_head.to(dtype=torch.float32)

    def _capture(self, module, args, output):
        if self._want_hidden:
            self._hidden = output

    def training_step(self, data_batch: dict, data_batch_idx: int, force_hand_sigma: bool = False):
        """force_hand_sigma (validation): every sample at sigma = hand_sigma and in the hand loss."""
        cfg = self.config
        self.pipe.device = self.device
        # the DiT concatenates padding_mask with the latents and projects the T5 context -> both in the model dtype
        data_batch["padding_mask"] = data_batch["padding_mask"].to(**self.tensor_kwargs)
        data_batch["obs/language_embedding"] = data_batch["obs/language_embedding"].to(**self.tensor_kwargs)
        x0, condition = self.pipe.get_data_and_condition(data_batch)
        sigma_B_T, eps = self.draw_training_sigma_and_epsilon(x0.size(), condition)
        B = x0.shape[0]
        hand_sel = torch.zeros(B, dtype=torch.bool, device=x0.device)
        if force_hand_sigma:
            hand_sel[:] = True
            sigma_B_T = torch.full_like(sigma_B_T, cfg.hand_sigma)
        elif cfg.hand_mode == "clean":
            hand_sel = torch.rand(B, device=x0.device) < cfg.hand_fraction
            if not hand_sel.any():
                hand_sel[torch.randint(B, (1,))] = True
            sigma_B_T = torch.where(hand_sel[:, None], torch.full_like(sigma_B_T, cfg.hand_sigma), sigma_B_T)
        elif cfg.hand_mode == "leaky":
            hand_sel[:] = True
        x0, condition, eps, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(x0, condition, eps, sigma_B_T)
        self._hidden, self._want_hidden = None, cfg.hand_mode != "off" or force_hand_sigma
        output_batch, video_loss = self.compute_loss_with_epsilon_and_sigma(x0, condition, eps, sigma_B_T)
        self._want_hidden = False
        video_loss = video_loss.mean() * self.loss_scale if self.loss_reduce == "mean" else \
            video_loss.sum(dim=1).mean() * self.loss_scale
        loss = video_loss
        output_batch["video_loss"] = video_loss.item()
        if cfg.hand_mode != "off" or force_hand_sigma:
            h = self._hidden
            assert h is not None, "hand hook did not fire"
            h = h.reshape(B, -1, h.shape[-1])[hand_sel]
            sub = {k: data_batch[k].to(self.device)[hand_sel] for k in HAND_KEYS}
            pred = self.net.hand_head(h, sub["hand/geo_cur"].float(), sub["hand/cls_cur"].long(),
                                      sub["hand/mask_cur"].bool(), sub["hand/family"].long(), sigma_B_T[hand_sel, 0])
            hand_loss, metrics = self.net.hand_head.loss(pred, sub)
            if cfg.hand_mode != "off":
                loss = loss + cfg.hand_weight * hand_loss
            output_batch.update(metrics)
            output_batch["hand/weighted_over_video"] = cfg.hand_weight * metrics.get("hand/loss", 0.0) / max(
                video_loss.item(), 1e-8)
        self._hidden = None
        output_batch["loss"] = loss.item()
        return output_batch, loss

    @torch.no_grad()
    def validation_step(self, data_batch: dict, data_batch_idx: int):
        """Video loss at the training sigma distribution + hand metrics at the inference read-out sigma (all
        samples). In "off" mode the hand metrics measure an untrained head (sanity floor only)."""
        video = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data_batch.items()}
        out_v, loss = self.training_step(video, data_batch_idx)
        out = {"loss": out_v["loss"], "video_loss": out_v["video_loss"]}  # "loss": read by mimic-video callbacks
        out_h, _ = self.training_step(data_batch, data_batch_idx, force_hand_sigma=True)
        out.update({k: v for k, v in out_h.items() if k.startswith("hand/")})
        return out, loss
