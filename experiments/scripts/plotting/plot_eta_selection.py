#!/usr/bin/env python
"""Plot CV-A eta-selection curves from responses_eta_selection/eta_selection.csv."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(os.environ.get("SRC", "responses_eta_selection/eta_selection.csv"))
OUT = Path(os.environ.get("OUT", "responses_writeup/figs/eta_cv_a.png"))
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)
etas = sorted(df["eta"].unique())
g = df.groupby("eta")
cv = g["cv_dev"].mean().reindex(etas)
td = g["test_dev"].mean().reindex(etas)
c1 = g["cidx1"].mean().reindex(etas)
c2 = g["cidx2"].mean().reindex(etas)

x = np.arange(len(etas))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

ax[0].plot(x, cv.values, "o-", color="#1f77b4")
ax[0].set_title("CV-A held-out deviance (sum)")
ax[0].set_xlabel(r"$\eta$"); ax[0].set_ylabel("summed held-out deviance")
amin = int(np.nanargmin(cv.values))
ax[0].scatter([x[amin]], [cv.values[amin]], color="crimson", zorder=5,
              label=fr"$\hat\eta$={etas[amin]:g} (grid max)")
ax[0].legend(fontsize=8)

ax[1].plot(x, td.values, "s-", color="#2ca02c")
ax[1].set_title("test-set deviance (mean)")
ax[1].set_xlabel(r"$\eta$"); ax[1].set_ylabel("test deviance")
tmin = int(np.nanargmin(td.values))
ax[1].scatter([x[tmin]], [td.values[tmin]], color="crimson", zorder=5,
              label=fr"min at $\eta$={etas[tmin]:g}")
ax[1].legend(fontsize=8)

ax[2].plot(x, c1.values, "^-", color="#9467bd", label="cause 1")
ax[2].plot(x, c2.values, "v-", color="#ff7f0e", label="cause 2")
ax[2].set_title("competing-risk C-index")
ax[2].set_xlabel(r"$\eta$"); ax[2].set_ylabel("C-index")
ax[2].legend(fontsize=8)

for a in ax:
    a.set_xticks(x); a.set_xticklabels([f"{e:g}" for e in etas]); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
