#!/usr/bin/env python
"""Per-horizon figures for E1 (eta,sigma) grid and E3 (sandwich vs raw)."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
E1 = ROOT / os.environ.get("E1_DIR", "responses_exp1m")   # manuscript-matched R=15 runs
E3 = ROOT / os.environ.get("E3_DIR", "responses_exp3m")
OUT = ROOT / "responses_writeup" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ETAS = [0.0, 0.5, 1.0]
SIGMAS = [1.0, 5.0, 10.0, 30.0]
SIG_COL = dict(zip(SIGMAS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))


def load_e1(eta, sg):
    d = np.load(E1 / f"exp1_eta{eta:g}_sig{sg:g}.npz")
    return d["cif_cov"], d["cif_width"]


def fig_e1():
    K = len(load_e1(0.0, 1.0)[0]); x = np.arange(1, K + 1)
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for j, eta in enumerate(ETAS):
        for sg in SIGMAS:
            cov, w = load_e1(eta, sg)
            ax[0, j].plot(x, cov, "-o", ms=3.5, color=SIG_COL[sg], label=f"$\\sigma$={sg:g}")
            ax[1, j].plot(x, w, "-o", ms=3.5, color=SIG_COL[sg])
        ax[0, j].axhline(0.95, ls="--", lw=1, color="0.5")
        ax[0, j].set_title(f"$\\eta$ = {eta:g}", fontsize=12, fontweight="bold")
        ax[0, j].set_ylim(0.90, 1.01); ax[1, j].set_ylim(0.03, 0.16)
        ax[1, j].set_xlabel("time interval $t_k$"); ax[1, j].set_xticks(x)
        for r in (0, 1):
            ax[r, j].grid(alpha=0.25)
    ax[0, 0].set_ylabel("95% CIF coverage")
    ax[1, 0].set_ylabel("CIF interval width")
    ax[0, 2].legend(fontsize=9, framealpha=0.9, loc="lower left")
    fig.suptitle("E1 — per-horizon CIF coverage (top) & interval width (bottom), by $(\\eta,\\sigma)$",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "E1_perhorizon_cov_width.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_e3():
    files = sorted(E3.glob("perhz_eta*.npz"), key=lambda f: float(f.stem.split("eta")[1]))
    etas = [float(f.stem.split("eta")[1]) for f in files]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(files)))
    d0 = np.load(files[0]); K = len(d0["cov_raw"]); x = np.arange(1, K + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for f, eta, col in zip(files, etas, cmap):
        d = np.load(f)
        ax[0].plot(x, d["cov_raw"], "-o", ms=3.5, color=col, label=f"$\\eta$={eta:g}")
        ax[0].plot(x, d["cov_sand"], "--s", ms=3.5, color=col, mfc="none")
        ax[1].plot(x, d["width_raw"], "-o", ms=3.5, color=col)
        ax[1].plot(x, d["width_sand"], "--s", ms=3.5, color=col, mfc="none")
    ax[0].axhline(0.95, ls=":", lw=1.2, color="k")
    ax[0].set_ylabel("95% CIF coverage"); ax[0].set_ylim(0.90, 1.01)
    ax[1].set_ylabel("CIF interval width")
    for a in ax:
        a.set_xlabel("time interval $t_k$"); a.set_xticks(x); a.grid(alpha=0.25)
    # legends: color = eta ; linestyle = raw vs sandwich
    leg1 = ax[0].legend(fontsize=9, ncol=2, framealpha=0.9, loc="lower right", title="$\\eta$")
    ax[0].add_artist(leg1)
    from matplotlib.lines import Line2D
    style = [Line2D([0], [0], color="0.3", ls="-", marker="o", label="raw posterior"),
             Line2D([0], [0], color="0.3", ls="--", marker="s", mfc="none", label="sandwich")]
    ax[1].legend(handles=style, fontsize=9, loc="upper right", framealpha=0.9)
    fig.suptitle("E3 — per-horizon CIF coverage (left) & interval width (right): raw vs sandwich",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / "E3_perhorizon_raw_vs_sandwich.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


if __name__ == "__main__":
    p1 = fig_e1(); p3 = fig_e3()
    print("saved:", p1); print("saved:", p3)
