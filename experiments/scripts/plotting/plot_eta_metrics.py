#!/usr/bin/env python
"""CIF coverage / RMSE / (LPML,WAIC,DIC) vs eta, from responses_eta_metrics/eta_summary.csv."""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(os.environ.get("SRC", "responses_eta_metrics/eta_summary.csv"))
OUT = Path(os.environ.get("OUT", "responses_writeup/figs/eta_metrics.png"))
OUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SRC).groupby("eta").mean(numeric_only=True).reset_index().sort_values("eta")
e = df["eta"].values

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(e, df["cif_cov"], "o-", color="#1f77b4", label="CIF coverage")
ax[0].plot(e, df["haz_cov"], "s--", color="#17becf", label="λ coverage")
ax[0].axhline(0.95, ls=":", c="gray"); ax[0].set_ylim(0.5, 1.02); ax[0].set_title("coverage vs η"); ax[0].legend(fontsize=8)
ax[1].plot(e, df["cif_rmse"], "o-", color="#d62728"); ax[1].set_title("CIF RMSE vs η")
ax[2].plot(e, df["lpml"], "o-", color="#2ca02c", label="LPML (↑ better)")
ax2 = ax[2].twinx()
ax2.plot(e, df["waic"], "s--", color="#9467bd", label="WAIC (↓)")
ax2.plot(e, df["dic"], "^--", color="#8c564b", label="DIC (↓)")
ax[2].set_title("LPML / WAIC / DIC vs η"); ax[2].set_ylabel("LPML"); ax2.set_ylabel("WAIC / DIC")
l1,la1 = ax[2].get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
ax[2].legend(l1+l2, la1+la2, fontsize=8, loc="center right")
for a in ax: a.set_xlabel("η"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print(f"saved {OUT}")
