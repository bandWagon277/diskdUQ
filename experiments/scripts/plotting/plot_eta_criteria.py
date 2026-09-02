#!/usr/bin/env python
"""Plot the four selection criteria + effective-df vs eta from eta_criteria.csv.

If a 'post' column is present (sgld vs laplace), one row of panels per posterior.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(os.environ.get("SRC", "responses_eta_criteria/eta_criteria.csv"))
OUT = Path(os.environ.get("OUT", "responses_writeup/figs/eta_criteria.png"))
OUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SRC)
posts = list(df["post"].unique()) if "post" in df.columns else [None]
etas = sorted(df["eta"].unique()); x = np.arange(len(etas))

panels = [("lpml", "LPML (argmax)", "max"), ("waic", "WAIC (argmin)", "min"),
          ("dic", "DIC (argmin)", "min"), ("gbic", "GBIC (argmin)", "min")]
nrow = len(posts)
fig, axes = plt.subplots(nrow, 5, figsize=(19, 3.4*nrow), squeeze=False)
for ri, post in enumerate(posts):
    d = df[df["post"] == post] if post is not None else df
    g = d.groupby("eta")
    for ci, (col, title, how) in enumerate(panels):
        ax = axes[ri][ci]; y = g[col].mean().reindex(etas).values
        ax.plot(x, y, "o-", color="#1f77b4")
        star = int(np.nanargmax(y) if how == "max" else np.nanargmin(y))
        ax.scatter([x[star]], [y[star]], color="crimson", zorder=5,
                   label=fr"$\hat\eta$={etas[star]:g}")
        ax.set_title((f"[{post}] " if post else "") + title); ax.legend(fontsize=8)
    axp = axes[ri][4]
    axp.plot(x, g["df_eff"].mean().reindex(etas).values, "s-", color="teal", label="df_eff (GBIC/Laplace)")
    axp.plot(x, g["p_waic"].mean().reindex(etas).values, "^-", color="orange", label="p_WAIC")
    axp.set_title((f"[{post}] " if post else "") + "effective # params"); axp.legend(fontsize=8)
    for ax in axes[ri]:
        ax.set_xticks(x); ax.set_xticklabels([f"{e:g}" for e in etas]); ax.set_xlabel(r"$\eta$"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT, dpi=140); print(f"saved {OUT}")
