#!/usr/bin/env python
"""Competing-risk Question-B: CIF & lambda credible-interval width, eta=0 vs eta=1 (CR->CR, overall->CR).
Reads cr_T*_eta*_K*_N500_*.npz from responses_competing/."""
import os, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(os.environ.get("SRC", "responses_competing"))
OUT = Path(os.environ.get("OUT", "responses_writeup/figs/ci_width_cr.png"))
OUT.parent.mkdir(parents=True, exist_ok=True)
def find(tag):
    h = [f for f in glob.glob(str(SRC/"cr_*_N500_*.npz")) if tag in f]
    return np.load(h[0]) if h else None
cases = [("eta0 (no teacher)", find("_Tnone_eta0_"), "#1f77b4"),
         ("eta1 CR->CR", find("_Tcr_cr_eta1_"), "#d62728"),
         ("eta1 overall->CR", find("_Toverall_eta1_"), "#2ca02c")]
cases = [(l, d, c) for l, d, c in cases if d is not None]
if not cases: raise SystemExit(f"no npz in {SRC}")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for lab, d, col in cases:
    h = np.arange(1, len(d["cif_width"])+1)
    ax[0].plot(h, d["cif_width"], "o-", color=col, label=lab)
    ax[1].plot(h, d["haz_width"], "o-", color=col, label=lab)
    ax[2].plot(d["cif_level"], d["cif_width"], "o-", color=col, label=lab)
ax[0].set_title("CIF 95% CI width vs horizon"); ax[0].set_xlabel("horizon"); ax[0].set_ylabel("CI width")
ax[1].set_title("lambda (hazard) 95% CI width vs horizon"); ax[1].set_xlabel("horizon"); ax[1].set_ylabel("CI width")
ax[2].set_title("CIF CI width vs CIF level (binomial check)"); ax[2].set_xlabel("CIF level"); ax[2].set_ylabel("CI width")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print(f"saved {OUT}")
