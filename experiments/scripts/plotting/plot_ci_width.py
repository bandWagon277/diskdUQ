#!/usr/bin/env python
"""CIF range vs credible-interval width, eta=0 vs eta=1 (simplest well-specified case).

Reads linear_well_K*_N*_T{none,oracle}_eta{0,1}_om1.npz from responses_ci_width/.
Left: 95% credible-interval width of CIF vs horizon. Middle: CIF cohort-mean (the "range" of
CIF, 0->~1) vs horizon. Right: credible width vs CIF level (binomial inverted-U check).
"""
import os, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(os.environ.get("SRC", "responses_ci_width"))
OUT = Path(os.environ.get("OUT", "responses_writeup/figs/ci_width_eta.png"))
OUT.parent.mkdir(parents=True, exist_ok=True)

def find(tag):
    hits = [f for f in glob.glob(str(SRC/"linear_*.npz")) if tag in f]
    return np.load(hits[0]) if hits else None

cases = [("eta0 (student only)", find("_Tnone_eta0_"), "#1f77b4"),
         ("eta1 (oracle teacher)", find("_Toracle_eta1_"), "#d62728")]
cases = [(l, d, c) for l, d, c in cases if d is not None]
if not cases: raise SystemExit(f"no npz found in {SRC}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for lab, d, col in cases:
    K = len(d["cif_width"]); h = np.arange(1, K+1)
    ax[0].plot(h, d["cif_width"], "o-", color=col, label=lab)
    ax[1].plot(h, d["cif_truth_mean"], "o-", color=col, label=lab)
    ax[2].scatter(d["cif_truth_mean"], d["cif_width"], color=col, label=lab, s=30)
ax[0].set_title("CIF 95% credible-interval width vs horizon"); ax[0].set_xlabel("horizon k"); ax[0].set_ylabel("CI width")
ax[1].set_title("True CIF (cohort mean) vs horizon = the CIF range"); ax[1].set_xlabel("horizon k"); ax[1].set_ylabel("CIF")
ax[2].set_title("CI width vs CIF level (binomial check: widest near 0.5)"); ax[2].set_xlabel("CIF level"); ax[2].set_ylabel("CI width")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print(f"saved {OUT}")
