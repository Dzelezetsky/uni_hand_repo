"""Future-hand head for Stage-1 (STAGE1_TRAINING.md: class + multi-hypothesis residual, per future latent frame).

Queries: one per (side, future latent frame) = 2 x 14. Each query = learned (frame, side) embedding + hand-family
embedding + embedding of the CURRENT hand (geometry 15D + posture class) of that side + video-noise embedding.
Queries self-attend and cross-attend to the video DiT hidden states of one block (all T*H*W tokens, 2048-d).
Per query outputs: K class logits; M residual hypotheses per class (change of the canonical fingertips relative to
the current geometry, 15D); M hypothesis logits per class.
Loss (masked by hand/mask_fut): CE(class) + w_res * min_m L1(true-class hypothesis m, target change)
                                + w_hyp * CE(hypothesis logits of the true class, argmin m)   (winner-takes-all).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class HandHead(nn.Module):
    def __init__(self, ctx_dim=2048, dim=512, layers=4, heads=8, n_future=14, n_sides=2, n_classes=7,
                 n_hyp=2, n_families=6, geo_dim=15, w_res=1.0, w_hyp=0.1):
        super().__init__()
        self.K, self.M, self.G, self.nf, self.ns = n_classes, n_hyp, geo_dim, n_future, n_sides
        self.w_res, self.w_hyp = w_res, w_hyp
        self.ctx = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, dim))
        self.query = nn.Parameter(torch.randn(n_sides * n_future, dim) * 0.02)
        self.family = nn.Embedding(n_families + 1, dim)          # last index = no verified hand on this side
        self.cur_cls = nn.Embedding(n_classes + 1, dim)          # last index = unknown
        self.cur_geo = nn.Sequential(nn.Linear(geo_dim + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.sigma = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_classes + n_classes * n_hyp * (geo_dim + 1))

    def forward(self, ctx_B_N_D, geo_cur_B_S_G, cls_cur_B_S, mask_cur_B_S, family_B_S, sigma_B):
        B = ctx_B_N_D.shape[0]
        ctx = self.ctx(ctx_B_N_D.float())
        cls_cur = torch.where(mask_cur_B_S, cls_cur_B_S.clamp(min=0), torch.full_like(cls_cur_B_S, self.K))
        fam = torch.where(family_B_S >= 0, family_B_S, torch.full_like(family_B_S, self.family.num_embeddings - 1))
        geo = torch.cat([geo_cur_B_S_G * mask_cur_B_S[..., None], mask_cur_B_S[..., None].float()], -1)
        per_side = self.family(fam) + self.cur_cls(cls_cur) + self.cur_geo(geo)                    # [B,S,D]
        q = self.query.view(self.ns, self.nf, -1)[None] + per_side[:, :, None]                     # [B,S,F,D]
        q = q + self.sigma(torch.log(sigma_B.float().clamp(min=1e-3))[:, None])[:, None, None]
        h = self.norm(self.decoder(q.reshape(B, self.ns * self.nf, -1), ctx))
        o = self.out(h).view(B, self.ns, self.nf, -1)
        logits = o[..., : self.K]
        rest = o[..., self.K:].reshape(B, self.ns, self.nf, self.K, self.M, self.G + 1)
        return logits, rest[..., : self.G], rest[..., self.G]              # [B,S,F,K], [B,S,F,K,M,G], [B,S,F,K,M]

    def loss(self, pred, batch):
        logits, res, hyp = pred
        cls = batch["hand/cls_fut"].long()
        m = batch["hand/mask_fut"].bool() & (cls >= 0)
        out = {}
        if not m.any():
            z = logits.sum() * 0.0
            return z, {"hand/n": 0.0}
        target = (batch["hand/geo_fut"] - batch["hand/geo_cur"][:, :, None]).float()           # change, [B,S,F,G]
        c = cls.clamp(min=0)
        idx = c[..., None, None, None].expand(*c.shape, 1, self.M, self.G)
        res_c = res.gather(3, idx).squeeze(3)                                                   # [B,S,F,M,G]
        hyp_c = hyp.gather(3, c[..., None, None].expand(*c.shape, 1, self.M)).squeeze(3)       # [B,S,F,M]
        l1 = (res_c - target[..., None, :]).abs().mean(-1)                                      # [B,S,F,M]
        best = l1.argmin(-1)
        l_cls = F.cross_entropy(logits[m].float(), c[m])
        l_res = l1.min(-1).values[m].mean()
        l_hyp = F.cross_entropy(hyp_c[m].float(), best[m])
        total = l_cls + self.w_res * l_res + self.w_hyp * l_hyp
        with torch.no_grad():
            acc = (logits.argmax(-1) == c)[m].float().mean()
            copy = (batch["hand/cls_cur"].long()[:, :, None].expand_as(c) == c)[m].float().mean()
            geo_err = (res_c.gather(3, hyp_c.argmax(-1)[..., None, None].expand(*c.shape, 1, self.G)).squeeze(3)
                       - target).norm(dim=-1)[m].mean()
        out.update({"hand/loss": total.item(), "hand/ce": l_cls.item(), "hand/res_l1": l_res.item(),
                    "hand/acc": acc.item(), "hand/acc_copy_current": copy.item(), "hand/geo_err_pw": geo_err.item(),
                    "hand/n": float(m.sum().item())})
        return total, out
