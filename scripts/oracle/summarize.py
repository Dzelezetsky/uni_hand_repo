"""Summarize oracle stage-A results: ARM improvement over base (%), mean ± std over seeds, per dataset."""
import glob

import pandas as pd

R = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("unified/oracle/results_*.csv")))
order = ["cls", "geo", "chg", "cls_p25", "cls_p50", "cls_rand"]
for tgt in ("arm", "hand"):
    S = R[R.target == tgt].groupby(["dataset", "horizon", "scope", "variant"]).improvement.agg(["mean", "std"])
    T = (S["mean"] * 100).round(1).astype(str) + "±" + (S["std"] * 100).round(1).astype(str)
    T = T.unstack("variant")[[v for v in order if v in set(R.variant)]]
    print(f"\n{tgt.upper()} improvement over base, % (1 - MSE/MSE_base){' — leak-prone' if tgt == 'hand' else ''}")
    print(T.to_string())
B = R[(R.variant == "base") & (R.target == "arm")].groupby(["dataset", "horizon", "scope"]).skill.mean().unstack()
print("\nARM skill of base (1 - MSE/MSE_zero):\n" + B.round(3).to_string())
