"""V5 (joint limits) over ALL episodes of a dataset, straight from the adapter + mapping (nothing stored).

Prints per mapping: fraction of samples beyond a URDF limit by > 0.05 / 0.15 (V5_TOL) / 0.3 rad (gross), the worst
joints, and |action - state| as a sanity check that state is not just the command.
usage: python scripts/analysis/check_v5.py <dataset_id> [max_episodes]
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from unidex.convert import GROSS_TOL, LIMIT_TOL, V5_TOL, adapters  # noqa: E402
from unidex.hands import load_hand  # noqa: E402
from unidex.mappings import MAPPINGS  # noqa: E402


def main(ds, max_eps=None):
    mod = adapters()[ds]
    eps = mod.list_episodes()
    if max_eps:
        eps = eps[:: max(1, len(eps) // max_eps)][:max_eps]
    V, D, n_eps = defaultdict(list), defaultdict(list), 0
    for eid in eps:
        e = mod.load_episode(eid)
        n_eps += 1
        for h in e.hands:
            if not h.mapping_id or h.status_override:
                continue
            m = MAPPINGS[h.mapping_id]
            hand = load_hand(m.hand_model_id)
            q = m(h.native_q)
            lo = np.array([hand.fk.joints[j].lower for j in m.model_joints])
            hi = np.array([hand.fk.joints[j].upper for j in m.model_joints])
            V[h.mapping_id].append(np.maximum(lo - q, 0) + np.maximum(q - hi, 0))
            if h.native_action is not None and h.native_action.shape == h.native_q.shape:
                D[h.mapping_id].append(np.abs(h.native_action - h.native_q))
    print(f"{ds}: {n_eps} episodes")
    for mid, vs in V.items():
        v = np.concatenate(vs)
        names = MAPPINGS[mid].model_joints
        worst = np.argsort(-(v > LIMIT_TOL).mean(0))[:4]
        print(f"  {mid}: n={len(v)}  >{LIMIT_TOL} {np.mean((v > LIMIT_TOL).any(1)):.4%}  >{V5_TOL} (V5) "
              f"{np.mean((v > V5_TOL).any(1)):.4%}  >{GROSS_TOL} (gross) {np.mean((v > GROSS_TOL).any(1)):.4%}  "
              f"max {v.max():.3f}")
        print("    worst joints:", [(names[i], f"{(v[:, i] > LIMIT_TOL).mean():.4f}", round(float(v[:, i].max()), 3))
                                    for i in worst])
        if D[mid]:
            d = np.concatenate(D[mid])
            print(f"    |action-state| median {np.median(d):.3f}  p99 {np.percentile(d, 99):.3f} rad")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
