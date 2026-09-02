#!/usr/bin/env python
"""Per-horizon coverage & width figures (same style as E1/E3) for:
   1. small-sigma arm (responses_exp1m_smallsig)
   2. exp7 eta x student-N sweep (responses_exp7/N*)
   3. expMLP MLP-student (responses_expMLP), vs GLM E1m reference.
Run any subset via WHICH=smallsig,exp7,mlp (default all-that-have-data).
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "responses_writeup" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
ETAS3 = [0.0, 0.5, 1.0]


def _grid(nrow, ncol, w=4.4, h=3.4):
    fig, ax = plt.subplots(nrow, ncol, figsize=(w*ncol, h*nrow), squeeze=False)
    return fig, ax


def fig_smallsigma():
    d0 = ROOT / "responses_exp1m_smallsig"
    sig = [0.1, 0.2, 0.5]; col = dict(zip(sig, ["#d7191c", "#fdae61", "#2c7bb6"]))
    x = np.arange(1, 11)
    fig, ax = _grid(2, 3)
    for j, eta in enumerate(ETAS3):
        for s in sig:
            f = d0 / f"exp1_eta{eta:g}_sig{s:g}.npz"
            if not f.exists(): continue
            z = np.load(f)
            ax[0, j].plot(x, z["cif_cov"], "-o", ms=3.5, color=col[s], label=f"$\\sigma$={s:g}")
            ax[1, j].plot(x, z["cif_width"], "-o", ms=3.5, color=col[s])
        ax[0, j].axhline(0.95, ls="--", lw=1, color="0.5"); ax[0, j].set_title(f"$\\eta$ = {eta:g}", fontweight="bold")
        ax[0, j].set_ylim(-0.03, 1.05); ax[1, j].set_xlabel("time interval $t_k$")
        for r in (0, 1): ax[r, j].set_xticks(x); ax[r, j].grid(alpha=0.25)
    ax[0, 0].set_ylabel("95% CIF coverage"); ax[1, 0].set_ylabel("CIF interval width")
    ax[0, 2].legend(fontsize=9, loc="center right", framealpha=0.9)
    fig.suptitle("Small-$\\sigma$ (strong prior): per-horizon CIF coverage (top) & width (bottom), by $(\\eta,\\sigma)$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = OUT / "smallsigma_perhorizon.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_exp7():
    Ns = [250, 500, 1000, 2000, 4000, 8000]; cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(Ns)))
    etas = [0.0, 1.0]; x = np.arange(1, 11)
    fig, ax = _grid(2, 2, w=5.2)
    for j, eta in enumerate(etas):
        for N, c in zip(Ns, cmap):
            f = ROOT / f"responses_exp7/N{N}/exp1_eta{eta:g}_sig10.npz"
            if not f.exists(): continue
            z = np.load(f)
            ax[0, j].plot(x, z["cif_cov"], "-o", ms=3.5, color=c, label=f"N={N}")
            ax[1, j].plot(x, z["cif_width"], "-o", ms=3.5, color=c)
        ax[0, j].axhline(0.95, ls="--", lw=1, color="0.5"); ax[0, j].set_title(f"$\\eta$ = {eta:g}", fontweight="bold")
        ax[0, j].set_ylim(0.88, 1.01); ax[1, j].set_yscale("log"); ax[1, j].set_xlabel("time interval $t_k$")
        for r in (0, 1): ax[r, j].set_xticks(x); ax[r, j].grid(alpha=0.25, which="both")
    ax[0, 0].set_ylabel("95% CIF coverage"); ax[1, 0].set_ylabel("CIF interval width (log)")
    ax[0, 1].legend(fontsize=8, ncol=2, loc="lower center", framealpha=0.9)
    fig.suptitle("E7 ($\\eta\\times$ student-$N$, oracle teacher): per-horizon CIF coverage (top) & width (bottom)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = OUT / "exp7_perhorizon.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_mlp():
    dM = ROOT / "responses_expMLP"; dG = ROOT / "responses_exp1m"
    if not (dM / f"expMLP_eta0.npz").exists(): return None
    x = np.arange(1, 11)
    fig, ax = _grid(2, 3)
    for j, eta in enumerate(ETAS3):
        zM = np.load(dM / f"expMLP_eta{eta:g}.npz"); zG = np.load(dG / f"exp1_eta{eta:g}_sig10.npz")
        ax[0, j].plot(x, zM["cif_cov"], "-o", ms=3.5, color="#c0392b", label="MLP student")
        ax[0, j].plot(x, zG["cif_cov"], "--s", ms=3.5, color="#2c7bb6", mfc="none", label="GLM student")
        ax[1, j].plot(x, zM["cif_width"], "-o", ms=3.5, color="#c0392b")
        ax[1, j].plot(x, zG["cif_width"], "--s", ms=3.5, color="#2c7bb6", mfc="none")
        ax[0, j].axhline(0.95, ls="--", lw=1, color="0.5"); ax[0, j].set_title(f"$\\eta$ = {eta:g}", fontweight="bold")
        ax[0, j].set_ylim(0.55, 1.03); ax[1, j].set_xlabel("time interval $t_k$")
        for r in (0, 1): ax[r, j].set_xticks(x); ax[r, j].grid(alpha=0.25)
    ax[0, 0].set_ylabel("95% CIF coverage"); ax[1, 0].set_ylabel("CIF interval width")
    ax[0, 2].legend(fontsize=9, loc="lower left", framealpha=0.9)
    fig.suptitle("MLP vs GLM student (linear DGP, oracle teacher): per-horizon CIF coverage (top) & width (bottom)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = OUT / "expMLP_perhorizon.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


if __name__ == "__main__":
    which = os.environ.get("WHICH", "smallsig,exp7,mlp").split(",")
    fns = {"smallsig": fig_smallsigma, "exp7": fig_exp7, "mlp": fig_mlp}
    for w in which:
        p = fns[w.strip()]()
        print("saved:", p) if p else print(f"skipped {w} (no data yet)")
