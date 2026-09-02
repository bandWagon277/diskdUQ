#!/usr/bin/env python
"""Per-horizon bias-vs-variance diagnostic for the two CIF-coverage cases.

Reads horizon_*.npz from responses_horizon_diag/ (best = K5 N500, worst = K5 N8000)
and plots, per horizon: posterior variance, coverage, and the bias/PostSD/RMSE
decomposition that tells variance apart from bias.
"""
import os, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(os.environ.get("SRC", "responses_horizon_diag"))
OUT = Path(os.environ.get("OUT", "responses_writeup/figs/horizon_bias_variance.png"))
OUT.parent.mkdir(parents=True, exist_ok=True)
FN = os.environ.get("FN", "cif")                                 # cif or haz

cases = []
for tag, lab in [("N500", "best coverage (N=500)"), ("N8000", "worst coverage (N=8000)")]:
    hits = [f for f in glob.glob(str(SRC/"horizon_*.npz")) if f"_{tag}_" in f]
    if hits: cases.append((lab, np.load(hits[0])))
if not cases:
    raise SystemExit(f"no horizon npz found in {SRC}")

fig, ax = plt.subplots(2, 2, figsize=(12, 8)); ax = ax.ravel()
colors = ["#1f77b4", "#d62728"]
for (lab, d), col in zip(cases, colors):
    K = len(d[f"{FN}_cov"]); h = np.arange(1, K+1)
    ax[0].plot(h, d[f"{FN}_post_var"], "o-", color=col, label=lab)
    ax[1].plot(h, d[f"{FN}_cov"], "o-", color=col, label=lab)
    # variance-vs-bias: PostSD (claimed) vs |bias| (systematic) vs RMSE (total point error)
    ax[2].plot(h, d[f"{FN}_post_sd"], "o-", color=col, label=f"PostSD — {lab}")
    ax[2].plot(h, d[f"{FN}_rmse"], "s--", color=col, alpha=0.7, label=f"RMSE — {lab}")
    ax[3].plot(h, d[f"{FN}_bias"], "o-", color=col, label=lab)

ax[0].set_title(f"{FN.upper()} posterior variance vs horizon"); ax[0].set_ylabel("mean posterior variance")
ax[1].set_title(f"{FN.upper()} 95% coverage vs horizon"); ax[1].axhline(0.95, ls="--", c="gray", lw=.8); ax[1].set_ylabel("coverage"); ax[1].set_ylim(0, 1)
ax[2].set_title("Claimed spread (PostSD) vs total error (RMSE)"); ax[2].set_ylabel("CIF units")
ax[3].set_title("Signed bias vs horizon (center error)"); ax[3].axhline(0, ls=":", c="gray", lw=.8); ax[3].set_ylabel("E[postmean - truth]")
for a in ax:
    a.set_xlabel("horizon k"); a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle(f"{FN.upper()}: variance vs bias by horizon (K=5).  PostSD<<RMSE => variance/under-dispersion;  large |bias| => bias problem", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
