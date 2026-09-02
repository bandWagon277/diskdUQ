"""SGLD analog of the ICML paper's Figure 2 (distillation-temperature sweep).

Paper Figure 2 (Sec. 5.3): for DiSKD-C under a high-quality teacher, sweep the
distillation temperature T in {1,2,3,4,5} and report cause-specific C_td (cause 1,
cause 2) and predictive deviance on the test set. Error bars there are summarized
over **20 random seeds** (i.e. 20 independent retrains per temperature).

This script reproduces the same 3-panel layout and x-axis
    {Internal, T=1, T=2, T=3, T=4, T=5}
but the error bars come from the **SGLD posterior** of a SINGLE warm-start fit
per temperature (the proposal's main configuration: warm-start from the AdamW
MAP, fixed step eps=2e-4, noise sigma=1, 5 chains x 500 draws, pooled). Each
posterior draw yields one (C_td1, C_td2, deviance) triple, so the spread of
those draws is the credible band -- the Bayesian replacement for the 20-seed
replicate band.

For context/validation we also (optionally) compute the paper-style 20-seed
AdamW band (one AdamW fit per seed per temperature) and overlay it, so the figure
literally shows: "the posterior band from ONE fit recovers what 20 retrains give,
at ~1/20 the compute."

Outputs (OUT_DIR, default responses_temperature/):
  temperature_figure_sgld.{pdf,png}     # clean SGLD-posterior version (deliverable)
  temperature_figure_compare.{pdf,png}  # SGLD posterior vs 20-seed AdamW overlay
  temperature_figure_data.npz           # per-draw + per-seed metric arrays
  temperature_figure_report.md
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    WarmStartMultiChainSampler,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd.uncertainty import _iter_samples


# ---------- Configuration ----------
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
TEST_N = int(os.environ.get("TEST_N", 500))
# "full" = high-quality teacher (matches the paper's high-quality-teacher Fig 2).
TEACHER_FEATURE_QUALITY = os.environ.get("TEACHER_FEATURE_QUALITY", "full")
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 32
BATCH_SIZE = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))

# Distillation temperatures to sweep for DiSKD-C (paper uses {1,2,3,4,5}).
TEMPS = [float(t) for t in os.environ.get("TEMPS", "1,2,3,4,5").split(",")]

# SGLD posterior band: single seed, proposal main config (warm-start, fixed eps).
SGLD_SEED = int(os.environ.get("SGLD_SEED", 42))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 2e-4))
EPS_T = float(os.environ.get("EPS_T", 2e-4))
GAMMA = float(os.environ.get("GAMMA", 0.55))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", 1.0))

# Paper-style 20-seed AdamW reference band (set SEED_BAND_N=0 to skip).
SEED_BAND_N = int(os.environ.get("SEED_BAND_N", 20))
SEED_BAND_START = int(os.environ.get("SEED_BAND_START", 42))

# Credible / replicate band percentiles for the whiskers.
Q_LO = float(os.environ.get("Q_LO", 5.0))
Q_HI = float(os.environ.get("Q_HI", 95.0))
# Test-set bootstrap replicates for the deployed-estimate CI.
N_BOOT = int(os.environ.get("N_BOOT", 400))

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses_temperature"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))

# x-axis configs in display order.
CONFIGS = ["Internal"] + [f"T={int(t) if t == int(t) else t}" for t in TEMPS]


# ---------- Builders ----------

def setup_cohorts(seed):
    cohorts = simulate_competing_risk_cohorts(
        n_teacher=TEACHER_N, n_student=STUDENT_N, n_test=TEST_N,
        seed=seed, teacher_feature_quality=TEACHER_FEATURE_QUALITY,
    )
    combined_dur = np.concatenate([cohorts.teacher["duration"].values,
                                    cohorts.student["duration"].values])
    time_grid = fit_time_grid(combined_dur, NUM_DURATIONS)
    return cohorts, time_grid


def train_cr_teacher(cohorts, time_grid):
    return DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(cohorts.teacher, feature_cols=cohorts.teacher_features)


def build_adamw(config, teacher, time_grid):
    common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid, optimizer="adamw",
    )
    if config == "Internal":
        return DiscreteSurvivalModel(**common)
    temp = float(config.split("=")[1])
    return DiSKDStudent(teacher_model=teacher, teacher_type="competing",
                        eta=1.0, temperature=temp, **common)


def build_sgld(config, teacher, time_grid):
    common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid, optimizer="sgld",
        sgld_step_size=EPS0, sgld_final_step_size=EPS_T,
        sgld_gamma=GAMMA, sgld_drift_mode="literal",
        sgld_noise_scale=NOISE_SCALE,
        sgld_burnin_epochs=BURNIN_EPOCHS,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )
    if config == "Internal":
        return DiscreteSurvivalModel(**common)
    temp = float(config.split("=")[1])
    return DiSKDStudent(teacher_model=teacher, teacher_type="competing",
                        eta=1.0, temperature=temp, **common)


def fit_adamw_once(config, teacher, cohorts, time_grid, seed):
    """One AdamW fit; return (ctd1, ctd2, dev) on the test set + MAP state dict."""
    test = cohorts.test
    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = build_adamw(config, teacher, time_grid)
    m.fit(cohorts.student, feature_cols=cohorts.student_features)
    cif = np.asarray(m.predict_cif(test))
    ctd = competing_risk_c_index(cif, test_dur, test_ev)
    dev = float(predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_ev))
    pretrained = {k: v.clone() for k, v in m.net.state_dict().items()}
    return float(ctd[0]), float(ctd[1]), dev, pretrained


def run_sgld_perdraw(config, teacher, cohorts, time_grid, pretrained, seed):
    """Warm-start SGLD; return per-draw, per-chain-BMA, and pooled-BMA metrics.

    Each *chain* is one independent sampler run, so its BMA prediction (metric of
    that chain's median CIF / mean interval-prob surface) is the SGLD analog of a
    single replicate fit. The spread across chains is the "replicate band"; the
    pooled-BMA point is the deployed estimate. All BMA metrics are computed
    consistently as metric-of-averaged-prediction (not average-of-per-draw-metric).
    """
    test = cohorts.test
    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    base = build_sgld(config, teacher, time_grid)
    chain_seeds = [100 * seed + ci for ci in range(N_CHAINS)]
    sampler = WarmStartMultiChainSampler(
        base, pretrained, n_chains=N_CHAINS, seeds=chain_seeds,
    )
    sampler.fit(cohorts.student, feature_cols=cohorts.student_features)
    lead = sampler.lead_chain
    c1s, c2s, devs = [], [], []            # per-draw (pooled) scalar metrics
    cif_accum = []                          # all draws' CIF, for the pooled median
    chain_bma_c1, chain_bma_c2, chain_bma_dev = [], [], []  # per-chain BMA
    pooled_ip_sum = None                    # running sum of interval probs (pooled BMA dev)
    n_draws = 0
    for chain in sampler.chains:
        chain_cifs, chain_ips = [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            ip = m.predict_interval_probs(test).numpy()
            chain_cifs.append(cif); chain_ips.append(ip)
            cif_accum.append(cif)
            ctd = competing_risk_c_index(cif, test_dur, test_ev)
            c1s.append(float(ctd[0])); c2s.append(float(ctd[1]))
            devs.append(float(predictive_deviance(ip, test_idx, test_ev)))
        chain_cifs = np.stack(chain_cifs, axis=0)
        chain_ips = np.stack(chain_ips, axis=0)
        # Per-chain BMA: metric of this chain's averaged prediction.
        chain_med_cif = np.median(chain_cifs, axis=0)
        cctd = competing_risk_c_index(chain_med_cif, test_dur, test_ev)
        chain_bma_c1.append(float(cctd[0])); chain_bma_c2.append(float(cctd[1]))
        chain_bma_dev.append(float(predictive_deviance(
            chain_ips.mean(axis=0), test_idx, test_ev)))
        s = chain_ips.sum(axis=0)
        pooled_ip_sum = s if pooled_ip_sum is None else pooled_ip_sum + s
        n_draws += chain_ips.shape[0]
    # Pooled-BMA point estimate: metric of the all-chains averaged prediction.
    pooled = np.stack(cif_accum, axis=0)
    med_cif = np.median(pooled, axis=0)
    pooled_ip = pooled_ip_sum / n_draws
    ctd_point = competing_risk_c_index(med_cif, test_dur, test_ev)
    pooled_bma_dev = float(predictive_deviance(pooled_ip, test_idx, test_ev))

    # Test-set bootstrap CI of the deployed pooled-BMA metric: resample test
    # subjects with replacement and re-score the fixed pooled prediction. This is
    # the sampling uncertainty of the reported number, centred on the point.
    rng = np.random.default_rng(20240601 + seed)
    n = len(test_dur)
    bc1, bc2, bdev = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ctd = competing_risk_c_index(med_cif[idx], test_dur[idx], test_ev[idx])
        bc1.append(float(ctd[0])); bc2.append(float(ctd[1]))
        bdev.append(float(predictive_deviance(pooled_ip[idx], test_idx[idx], test_ev[idx])))
    return {
        "ctd1": np.array(c1s), "ctd2": np.array(c2s), "dev": np.array(devs),
        "ctd1_point": float(ctd_point[0]), "ctd2_point": float(ctd_point[1]),
        "dev_point": pooled_bma_dev,
        "chain_bma_ctd1": np.array(chain_bma_c1),
        "chain_bma_ctd2": np.array(chain_bma_c2),
        "chain_bma_dev": np.array(chain_bma_dev),
        "boot_ctd1": np.array(bc1), "boot_ctd2": np.array(bc2), "boot_dev": np.array(bdev),
    }


# ---------- Band helpers ----------

def band(arr):
    """Return (median, lo, hi) for the whisker percentiles."""
    a = np.asarray(arr, dtype=float)
    return float(np.median(a)), float(np.percentile(a, Q_LO)), float(np.percentile(a, Q_HI))


# ---------- Figures ----------

PANELS = [("ctd1", "$C^{td}$ cause 1"), ("ctd2", "$C^{td}$ cause 2"), ("dev", "Predictive deviance")]


def _errbar(ax, x, stats, color, marker, label, ls="-"):
    med = np.array([s[0] for s in stats])
    lo = np.array([s[1] for s in stats])
    hi = np.array([s[2] for s in stats])
    yerr = np.vstack([med - lo, hi - med])
    ax.errorbar(x, med, yerr=yerr, color=color, marker=marker, ls=ls,
                capsize=3, lw=1.4, ms=5, label=label)


def fig_sgld_only(sgld_stats):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(CONFIGS))
    for ax, (key, ylab) in zip(axes, PANELS):
        _errbar(ax, x, sgld_stats[key], "#1f77b4", "o", "SGLD posterior")
        ax.set_xticks(x); ax.set_xticklabels(CONFIGS, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(ls=":", alpha=0.4)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Distillation-temperature sweep (DiSKD-C, high-quality teacher) — "
        f"error bars = SGLD posterior {int(Q_LO)}–{int(Q_HI)}% credible interval "
        f"from ONE warm-start fit ({N_CHAINS} chains × {SAMPLES_PER_CHAIN} draws)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "temperature_figure_sgld.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_compare(sgld_stats, seed_stats):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(CONFIGS))
    for ax, (key, ylab) in zip(axes, PANELS):
        if seed_stats is not None:
            _errbar(ax, x - 0.08, seed_stats[key], "#ff7f0e", "s",
                    f"AdamW, {SEED_BAND_N} seeds", ls="--")
        _errbar(ax, x + 0.08, sgld_stats[key], "#1f77b4", "o",
                "SGLD posterior, 1 fit")
        ax.set_xticks(x); ax.set_xticklabels(CONFIGS, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(ls=":", alpha=0.4)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "ICML Fig. 2 reproduced: 20-seed AdamW replicate band vs SGLD posterior "
        "band from a single fit", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "temperature_figure_compare.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_grant(sgld_boot, sgld_points, seed_stats):
    """Grant headline: AdamW 20-seed band vs SGLD deployed pooled-BMA point with a
    test-set bootstrap CI.

    Blue line = the deployed one-fit estimate (metric of the all-chains averaged
    prediction). Blue band = its test-set bootstrap Q_LO-Q_HI CI -- the sampling
    uncertainty of the reported number, centred on the point.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    x = np.arange(len(CONFIGS))
    BLUE, ORANGE = "#1f77b4", "#ff7f0e"
    for ax, (key, ylab) in zip(axes, PANELS):
        if seed_stats is not None:
            sm = np.array([s[0] for s in seed_stats[key]])
            sl = np.array([s[1] for s in seed_stats[key]])
            sh = np.array([s[2] for s in seed_stats[key]])
            ax.fill_between(x, sl, sh, color=ORANGE, alpha=0.15)
            ax.plot(x, sm, color=ORANGE, ls="--", marker="s", ms=5, lw=1.4,
                    label=f"AdamW, {SEED_BAND_N} seeds (median + 5–95%)")
        pt = np.array(sgld_points[key])
        blo = np.array([np.percentile(b, Q_LO) for b in sgld_boot[key]])
        bhi = np.array([np.percentile(b, Q_HI) for b in sgld_boot[key]])
        ax.fill_between(x, blo, bhi, color=BLUE, alpha=0.18)
        ax.plot(x, pt, color=BLUE, marker="o", ms=5, lw=1.5,
                label=f"SGLD one fit, pooled BMA (+ {int(Q_HI-Q_LO)}% bootstrap CI)")
        ax.set_xticks(x); ax.set_xticklabels(CONFIGS, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(ls=":", alpha=0.4)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        f"Distillation-temperature sweep (DiSKD-C, high-quality teacher): the deployed "
        f"one-fit SGLD estimate vs {SEED_BAND_N}-seed AdamW; band = test-set bootstrap CI",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "temperature_figure_grant.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------- Report ----------

def write_report(sgld_stats, sgld_chain, sgld_boot, sgld_points, seed_stats):
    md = ["# Distillation-temperature figure (SGLD analog of ICML Fig. 2)", ""]
    md.append(f"**Cell:** teacher_n={TEACHER_N}, student_n={STUDENT_N}, test_n={TEST_N}, "
              f"teacher_feature_quality=\"{TEACHER_FEATURE_QUALITY}\" (high-quality teacher).")
    md.append(f"**SGLD:** seed {SGLD_SEED}, warm-start from AdamW MAP, "
              f"fixed ε={EPS0:.0e}, noise σ={NOISE_SCALE} (T≈N·σ²={STUDENT_N*NOISE_SCALE**2:.0f}), "
              f"{N_CHAINS} chains × {SAMPLES_PER_CHAIN} draws. All SGLD metrics are BMA "
              f"(metric of the averaged prediction), consistent across Ctd and deviance.")
    if seed_stats is not None:
        md.append(f"**AdamW reference band:** {SEED_BAND_N} random seeds "
                  f"(start {SEED_BAND_START}), one AdamW fit per seed. "
                  f"Whiskers = {int(Q_LO)}–{int(Q_HI)}% across seeds.")
    md.append("")
    md.append("## Deployed SGLD pooled-BMA point with test-set bootstrap CI (figure)")
    md.append("")
    md.append(f"Point = metric of the all-chains averaged prediction; "
              f"CI = {int(Q_LO)}–{int(Q_HI)}% over {N_BOOT} test-set bootstrap resamples.")
    md.append("")
    header = "| Config | SGLD Ctd1 [boot CI] | SGLD Ctd2 [boot CI] | SGLD Dev [boot CI] |"
    sep = "|---|---|---|---|"
    md.append(header); md.append(sep)
    for i, cfg in enumerate(CONFIGS):
        def bfmt(key):
            lo, hi = np.percentile(sgld_boot[key][i], [Q_LO, Q_HI])
            return f"{sgld_points[key][i]:.3f} [{lo:.3f}, {hi:.3f}]"
        md.append(f"| {cfg} | {bfmt('ctd1')} | {bfmt('ctd2')} | {bfmt('dev')} |")
    md.append("")
    md.append("Per-chain BMA spread (median [min, max] over chains) is reported for "
              "reference; it sits below the pooled point because pooling diverse "
              "(non-converged) chains is where the gain appears:")
    md.append("")
    header = "| Config | Ctd1 chains | Ctd2 chains | Dev chains |"
    sep = "|---|---|---|---|"
    md.append(header); md.append(sep)
    for i, cfg in enumerate(CONFIGS):
        def cfmt(key):
            a = sgld_chain[key][i]
            return f"{np.median(a):.3f} [{np.min(a):.3f}, {np.max(a):.3f}]"
        md.append(f"| {cfg} | {cfmt('ctd1')} | {cfmt('ctd2')} | {cfmt('dev')} |")
    md.append("")
    md.append("Note: SGLD deviance here is the BMA deviance (deviance of the averaged "
              "interval-prob surface). The earlier per-draw-average deviance is reported "
              "for reference below; it is an upper bound (convex score).")
    md.append("")
    md.append("| Config | per-draw mean Dev (legacy) | BMA Dev (consistent) |")
    md.append("|---|---|---|")
    for i, cfg in enumerate(CONFIGS):
        md.append(f"| {cfg} | {sgld_stats['dev'][i][0]:.3f} | {sgld_points['dev'][i]:.3f} |")
    if seed_stats is not None:
        md.append("")
        md.append("## AdamW 20-seed band (median [5–95%])")
        md.append("")
        md.append("| Config | AdamW Ctd1 | AdamW Ctd2 | AdamW Dev |")
        md.append("|---|---|---|---|")
        for i, cfg in enumerate(CONFIGS):
            def afmt(key):
                m, lo, hi = seed_stats[key][i]
                return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
            md.append(f"| {cfg} | {afmt('ctd1')} | {afmt('ctd2')} | {afmt('dev')} |")
    md.append("")
    md.append("## Figures")
    md.append("![grant](temperature_figure_grant.png)")
    md.append("")
    md.append("Legacy per-draw-cloud figures (kept for reference):")
    md.append("![sgld](temperature_figure_sgld.png)")
    if seed_stats is not None:
        md.append("![compare](temperature_figure_compare.png)")
    (OUT_DIR / "temperature_figure_report.md").write_text("\n".join(md) + "\n")


# ---------- Main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print("=== Temperature-sweep figure (SGLD analog of ICML Fig. 2) ===")
    print(f"Configs: {CONFIGS}")
    print(f"SGLD band: seed {SGLD_SEED}, warm-start, eps={EPS0:.0e} sigma={NOISE_SCALE} "
          f"{N_CHAINS}x{SAMPLES_PER_CHAIN} draws")
    print(f"Seed band: {SEED_BAND_N} seeds" if SEED_BAND_N else "Seed band: skipped")

    save = {}

    # ---- SGLD posterior band (single seed) ----
    print(f"\n--- SGLD posterior band (seed {SGLD_SEED}) ---", flush=True)
    cohorts, time_grid = setup_cohorts(SGLD_SEED)
    teacher = train_cr_teacher(cohorts, time_grid)
    sgld_stats = {"ctd1": [], "ctd2": [], "dev": []}        # per-draw band (legacy figs)
    sgld_chain = {"ctd1": [], "ctd2": [], "dev": []}        # per-chain BMA arrays (5 each)
    sgld_boot = {"ctd1": [], "ctd2": [], "dev": []}         # test-set bootstrap of pooled BMA
    sgld_points = {"ctd1": [], "ctd2": [], "dev": []}       # pooled-BMA point estimate
    for cfg in CONFIGS:
        t0 = time.time()
        _, _, _, pretrained = fit_adamw_once(cfg, teacher, cohorts, time_grid, 1000 * SGLD_SEED)
        d = run_sgld_perdraw(cfg, teacher, cohorts, time_grid, pretrained, SGLD_SEED)
        for key in ("ctd1", "ctd2", "dev"):
            sgld_stats[key].append(band(d[key]))
            save[f"sgld_{cfg}_{key}"] = d[key]
        chain_map = {"ctd1": "chain_bma_ctd1", "ctd2": "chain_bma_ctd2", "dev": "chain_bma_dev"}
        boot_map = {"ctd1": "boot_ctd1", "ctd2": "boot_ctd2", "dev": "boot_dev"}
        for key in ("ctd1", "ctd2", "dev"):
            sgld_chain[key].append(d[chain_map[key]])
            sgld_boot[key].append(d[boot_map[key]])
            save[f"sgld_{cfg}_chainbma_{key}"] = d[chain_map[key]]
            save[f"sgld_{cfg}_boot_{key}"] = d[boot_map[key]]
        sgld_points["ctd1"].append(d["ctd1_point"])
        sgld_points["ctd2"].append(d["ctd2_point"])
        sgld_points["dev"].append(d["dev_point"])
        bl, bh = np.percentile(d["boot_ctd1"], [Q_LO, Q_HI])
        print(f"  {cfg:9s} pooled-BMA Ctd1={d['ctd1_point']:.3f} boot[{bl:.3f},{bh:.3f}] "
              f"BMA-dev {d['dev_point']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    save["sgld_points_ctd1"] = np.array(sgld_points["ctd1"])
    save["sgld_points_ctd2"] = np.array(sgld_points["ctd2"])
    save["sgld_points_dev"] = np.array(sgld_points["dev"])

    # ---- 20-seed AdamW reference band ----
    seed_stats = None
    if SEED_BAND_N > 0:
        print(f"\n--- AdamW {SEED_BAND_N}-seed reference band ---", flush=True)
        per_seed = {cfg: {"ctd1": [], "ctd2": [], "dev": []} for cfg in CONFIGS}
        for s in range(SEED_BAND_N):
            seed = SEED_BAND_START + s
            co, tg = setup_cohorts(seed)
            tch = train_cr_teacher(co, tg)
            for cfg in CONFIGS:
                c1, c2, dev, _ = fit_adamw_once(cfg, tch, co, tg, seed)
                per_seed[cfg]["ctd1"].append(c1)
                per_seed[cfg]["ctd2"].append(c2)
                per_seed[cfg]["dev"].append(dev)
            print(f"  seed {seed} done", flush=True)
        seed_stats = {"ctd1": [], "ctd2": [], "dev": []}
        for cfg in CONFIGS:
            for key in ("ctd1", "ctd2", "dev"):
                arr = np.array(per_seed[cfg][key])
                seed_stats[key].append(band(arr))
                save[f"seed_{cfg}_{key}"] = arr

    # ---- Save / plot / report ----
    save["configs"] = np.array(CONFIGS)
    save["temps"] = np.array(TEMPS)
    np.savez(OUT_DIR / "temperature_figure_data.npz", **save)
    print(f"\nData saved to {OUT_DIR / 'temperature_figure_data.npz'}", flush=True)

    fig_grant(sgld_boot, sgld_points, seed_stats)
    fig_sgld_only(sgld_stats)
    if seed_stats is not None:
        fig_compare(sgld_stats, seed_stats)
    write_report(sgld_stats, sgld_chain, sgld_boot, sgld_points, seed_stats)
    print("Figures + report saved.", flush=True)


if __name__ == "__main__":
    main()
