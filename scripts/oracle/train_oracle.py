"""Oracle experiment, stage A: does GROUND-TRUTH future hand information improve action-chunk prediction?

Per dataset (unified/oracle/<ds>.npz from build_windows.py), an MLP predicts the arm and hand deltas over the next
2 s (8 horizons) from: arm history, current hand state + current posture class, instruction text. Variants add a
FUTURE-hand input at 0.5/1/1.5/2 s:
  base       nothing
  cls        true future hand_posture_class_v1 (one-hot per side and horizon)
  geo        true future canonical fingertips (15D per side and horizon)
  cls_p25/50 true class replaced by a random class (drawn from the marginal) with probability 0.25 / 0.5
  cls_rand   random class from the marginal (p = 1): same input size, no information -> capacity control
  chg        only WHETHER the class at 0.5/1/1.5/2 s differs from the current one (timing, no grasp type)
Main metric: ARM prediction (the hand target is nearly the oracle itself, so hand metrics are leak-prone and only
reported). Error = MSE on per-dim standardized deltas; skill = 1 - MSE / MSE(zero delta); improvement of a variant
= 1 - MSE_variant / MSE_base on the same test anchors. Groups: near (0.25-1 s) / far (1.25-2 s) horizons; all test
anchors / event anchors (hand change over the next 2 s in the top 20 %). Episode-level splits 80/10/10, SEEDS seeds.
Output: unified/oracle/results_<ds>.csv and printed table.
usage: python scripts/oracle/train_oracle.py <dataset> [variants...]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, ".")
from unidex.hands import REPO  # noqa: E402

OUT = REPO / "unified" / "oracle"
VARIANTS = ["base", "cls", "geo", "cls_p25", "cls_p50", "cls_rand", "chg"]
SEEDS, NCLS, EPOCHS, BATCH, MAX_TRAIN = (0, 1, 2), 9, 40, 2048, 1_500_000
DEV = "cuda"


def onehot(c, n=NCLS):
    return np.eye(n, dtype=np.float32)[c.astype(int)]


def corrupt(cls_fut, p, marg, rng):
    c = cls_fut.copy()
    m = rng.random(c.shape) < p
    c[m] = rng.choice(len(marg), m.sum(), p=marg)
    return c


def features(D, variant, rng):
    n = len(D["group"])
    parts = [D["arm_hist"], D["text"]]
    for s in ("left", "right"):
        parts += [D[f"{s}_mask"][:, None], D[f"{s}_q"], onehot(D[f"{s}_cls"])]
    for s in ("left", "right"):
        cf = D[f"{s}_cls_fut"]
        act = D[f"{s}_mask"] > 0
        marg = np.bincount(cf[act].ravel(), minlength=NCLS).astype(float)
        marg = marg / marg.sum() if marg.sum() else np.eye(NCLS)[-1]
        if variant == "cls":
            parts.append(onehot(cf).reshape(n, -1))
        elif variant.startswith("cls_p") or variant == "cls_rand":
            p = 1.0 if variant == "cls_rand" else int(variant[5:]) / 100
            c = corrupt(cf, p, marg, rng)
            c[~act] = cf[~act]
            parts.append(onehot(c).reshape(n, -1))
        elif variant == "chg":
            parts.append(((cf != D[f"{s}_cls"][:, None]) & act[:, None]).astype(np.float32))
        elif variant == "geo":
            parts.append(D[f"{s}_geo_fut"].reshape(n, -1))
    return np.concatenate([np.asarray(x, np.float32).reshape(n, -1) for x in parts], 1)


def targets(D):
    n = len(D["group"])
    arm = D["arm_tgt"].reshape(n, -1)
    hands, masks = [], []
    for s in ("left", "right"):
        q = D[f"{s}_q_tgt"].reshape(n, -1)
        hands.append(q)
        masks.append(np.repeat(D[f"{s}_mask"][:, None], q.shape[1], 1))
    return arm.astype(np.float32), np.concatenate(hands, 1).astype(np.float32), np.concatenate(masks, 1)


class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1024), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, d_out))

    def forward(self, x):
        return self.net(x)


def run(ds, variants):
    D = dict(np.load(OUT / f"{ds}.npz"))
    TH = D["target_h"]
    G = D["group"]
    arm, hand, hmask = targets(D)
    na, nh, nt = arm.shape[1] // len(TH), hand.shape[1], len(TH)
    event_thr = np.quantile(D["change"], 0.8)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(G.max() + 1)
        split = np.zeros(G.max() + 1, int)
        split[perm[int(0.8 * len(perm)):int(0.9 * len(perm))]] = 1
        split[perm[int(0.9 * len(perm)):]] = 2
        sp = split[G]
        tr, va, te = (np.where(sp == k)[0] for k in range(3))
        if len(tr) > MAX_TRAIN:
            tr = np.sort(rng.choice(tr, MAX_TRAIN, replace=False))
        Y = np.concatenate([arm, hand], 1)
        W = np.concatenate([np.ones_like(arm), hmask], 1)
        ymu = np.average(Y[tr], axis=0, weights=W[tr] + 1e-9)
        ysd = np.sqrt(np.average((Y[tr] - ymu) ** 2, axis=0, weights=W[tr] + 1e-9)) + 1e-6
        Yz = (Y - ymu) / ysd
        zero_z = (0 - ymu) / ysd  # "no change" prediction in z units
        test_err = {}
        for v in variants:
            torch.manual_seed(seed)
            X = features(D, v, np.random.default_rng(seed + 100))
            xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xz = torch.tensor((X - xmu) / xsd, device=DEV)
            Yt, Wt = torch.tensor(Yz, device=DEV), torch.tensor(W, device=DEV)
            model = MLP(X.shape[1], Y.shape[1]).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
            steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
            trt, vat = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
            best, best_state, t0 = 1e9, None, time.time()
            for ep in range(EPOCHS):
                model.train()
                for b in trt[torch.randperm(len(trt), device=DEV)].split(BATCH):
                    loss = (((model(Xz[b]) - Yt[b]) ** 2) * Wt[b]).sum() / Wt[b].sum()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                model.eval()
                with torch.no_grad():
                    vl = sum(((((model(Xz[b]) - Yt[b]) ** 2) * Wt[b]).sum()).item() for b in vat.split(8192))
                if vl < best:
                    best, best_state = vl, {k: x.clone() for k, x in model.state_dict().items()}
            model.load_state_dict(best_state)
            with torch.no_grad():
                P = torch.cat([model(Xz[b]) for b in torch.tensor(te, device=DEV).split(8192)]).cpu().numpy()
            test_err[v] = (P - Yz[te]) ** 2
            print(f"{ds} seed {seed} {v:8s} val {best / len(va):.4f}  {time.time() - t0:.0f}s", flush=True)
        ev = D["change"][te] >= event_thr
        Wte = W[te]
        zero_err = (zero_z[None] - Yz[te]) ** 2
        for v in variants:
            for block, sl in (("arm", slice(0, na * nt)), ("hand", slice(na * nt, None))):
                e = test_err[v][:, sl]; eb = test_err["base"][:, sl]; ez = zero_err[:, sl]; w = Wte[:, sl]
                dim = e.shape[1] // nt if block == "arm" else None
                for hname, hm in (("near", TH <= 1.0), ("far", TH > 1.0)):
                    if block == "arm":
                        cols = np.concatenate([np.arange(i * na, (i + 1) * na) for i in np.where(hm)[0]])
                    else:  # hand columns: per side blocks of (nt x q)
                        cols, off = [], 0
                        for s in ("left", "right"):
                            q = D[f"{s}_q"].shape[1]
                            cols += [off + i * q + j for i in np.where(hm)[0] for j in range(q)]
                            off += nt * q
                        cols = np.array(cols)
                    for scope, m in (("all", np.ones(len(te), bool)), ("event", ev)):
                        ww = w[m][:, cols]
                        mse = (e[m][:, cols] * ww).sum() / ww.sum()
                        rows.append(dict(dataset=ds, seed=seed, variant=v, target=block, horizon=hname, scope=scope,
                                         mse=mse, skill=1 - mse / ((ez[m][:, cols] * ww).sum() / ww.sum()),
                                         improvement=1 - mse / ((eb[m][:, cols] * ww).sum() / ww.sum())))
    R = pd.DataFrame(rows)
    f = OUT / f"results_{ds}.csv"
    if f.exists():  # keep earlier variants (their improvement is relative to the base of their own run)
        old = pd.read_csv(f)
        R = pd.concat([old[~old.variant.isin(R.variant)], R], ignore_index=True)
    R.to_csv(f, index=False)
    S = R.groupby(["target", "horizon", "scope", "variant"]).improvement.agg(["mean", "std"]).unstack("variant")
    pd.set_option("display.width", 220)
    print(f"\n{ds}: improvement over base (1 - MSE/MSE_base), mean ± std over {len(SEEDS)} seeds")
    print((S["mean"] * 100).round(1).astype(str) + " ± " + (S["std"] * 100).round(1).astype(str))
    print("\nskill of base (1 - MSE/MSE_zero):")
    print(R[R.variant == "base"].groupby(["target", "horizon", "scope"]).skill.mean().round(3))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or VARIANTS)
