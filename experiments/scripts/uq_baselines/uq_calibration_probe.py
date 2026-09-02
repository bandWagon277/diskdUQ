"""UQ + calibration plots at a fixed cell, replicated across seeds.

Goal: deliver Jian's UQ-quality probes (width-vs-error correlation,
reliability diagrams, width-stratified diagnostics) with proper seed-level
error bars, at one or a few target cells from the lift matrix.

Default cell: TEACHER_N=5000, STUDENT_N=500, quality="reduced".
Seeds: 5.

For each (seed, method):
  - 5 AdamW restarts (Adam baseline + warm-start state from restart 0)
  - Warm-start SGLD: 5 chains x 500 draws, locked e3_g75 config
  - Per-entry (subject, cause, time) width / |err| / inside / median
  - Reliability-diagram inputs: predicted vs. closed-form true CIF

Outputs (responses/):
  uq_calibration_report.md
  uq_calibration_data.npz
  uq_calibration_reliability.{pdf,png}   # reliability diagram per method
  uq_calibration_width_stratified.{pdf,png}  # mean |err| + coverage per width bin
  uq_calibration_correlations.{pdf,png}  # Spearman rho per method per seed
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    MultiChainSampler,
    WarmStartMultiChainSampler,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import (
    _iter_samples,
    effective_sample_size,
    gelman_rubin_rhat,
)


# ---------- Configuration ----------
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
TEST_N = int(os.environ.get("TEST_N", 500))
TEACHER_FEATURE_QUALITY = os.environ.get("TEACHER_FEATURE_QUALITY", "reduced")
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 32
BATCH_SIZE = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))
N_ADAM_RESTARTS = int(os.environ.get("N_ADAM_RESTARTS", 5))
HORIZON_INDEX = int(os.environ.get("HORIZON_INDEX", 7))

SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 1e-3))
EPS_T = float(os.environ.get("EPS_T", 1e-5))
GAMMA = float(os.environ.get("GAMMA", 0.75))
# Temperature knob: noise-std multiplier sigma. Operating temperature
# T = N * sigma^2 / c (c=1 in literal mode), so sigma>1 -> hotter -> wider intervals.
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", 1.0))
# COLD_START=1 -> random-init chains (no warm-start from AdamW MAP); pair with a
# decaying schedule (EPS0>EPS_T, GAMMA) to reproduce the classic SGLD setting.
COLD_START = int(os.environ.get("COLD_START", 0))

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]

METHODS_ALL = ["internal", "cr_to_cr", "overall_to_cr", "binary1_to_cr", "binary2_to_cr"]
METHODS = [m.strip() for m in os.environ.get(
    "METHODS", ",".join(METHODS_ALL)).split(",") if m.strip() in METHODS_ALL]
METHOD_LABEL = {
    "internal":      "Internal CR",
    "cr_to_cr":      "CR -> CR",
    "overall_to_cr": "Overall -> CR",
    "binary1_to_cr": "Binary-1 -> CR",
    "binary2_to_cr": "Binary-2 -> CR",
}

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))


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


def train_teachers(cohorts, time_grid, methods):
    teachers = {}
    if "cr_to_cr" in methods:
        teachers["cr"] = DiscreteSurvivalModel(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
            epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
        ).fit(cohorts.teacher, feature_cols=cohorts.teacher_features)
    if "overall_to_cr" in methods:
        td = cohorts.teacher.copy()
        td["event_any"] = (td["event"] > 0).astype(int)
        teachers["overall"] = DiscreteSurvivalModel(
            num_risks=1, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
            epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
        ).fit(td, feature_cols=cohorts.teacher_features, event_col="event_any")
    if "binary1_to_cr" in methods or "binary2_to_cr" in methods:
        horizon_time = float(time_grid.cuts[HORIZON_INDEX])
        for risk_label in [1, 2]:
            method_key = f"binary{risk_label}_to_cr"
            if method_key not in methods:
                continue
            td = cohorts.teacher
            y = ((td["event"] == risk_label) & (td["duration"] <= horizon_time)).astype(int)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
            tf = list(cohorts.teacher_features)
            clf.fit(td[tf], y)
            # Closure capture of tf
            def make_predict(tf=tf, clf=clf):
                def predict(frame):
                    return clf.predict_proba(frame[tf])[:, 1]
                return predict
            teachers[f"bin{risk_label}"] = make_predict()
    return teachers


def _model_from(method, teachers, common):
    if method == "internal":
        return DiscreteSurvivalModel(**common)
    if method == "cr_to_cr":
        return DiSKDStudent(teacher_model=teachers["cr"], teacher_type="competing",
                            eta=1.0, temperature=2.0, **common)
    if method == "overall_to_cr":
        return DiSKDStudent(teacher_model=teachers["overall"], teacher_type="overall",
                            eta=1.0, **common)
    if method == "binary1_to_cr":
        return DiSKDStudent(teacher_model=teachers["bin1"], teacher_type="binary_horizon",
                            binary_risk_index=0, binary_horizon_index=HORIZON_INDEX,
                            eta=1.0, **common)
    if method == "binary2_to_cr":
        return DiSKDStudent(teacher_model=teachers["bin2"], teacher_type="binary_horizon",
                            binary_risk_index=1, binary_horizon_index=HORIZON_INDEX,
                            eta=1.0, **common)
    raise ValueError(method)


def build_adamw(method, teachers, time_grid):
    common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid, optimizer="adamw",
    )
    return _model_from(method, teachers, common)


def build_sgld(method, teachers, time_grid):
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
    return _model_from(method, teachers, common)


def run_adamw(method, teachers, cohorts, time_grid, seed):
    test = cohorts.test
    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    ctd1, ctd2, dev, cifs = [], [], [], []
    pretrained = None
    for r in range(N_ADAM_RESTARTS):
        torch.manual_seed(1000 * seed + r)
        np.random.seed(1000 * seed + r)
        m = build_adamw(method, teachers, time_grid)
        m.fit(cohorts.student, feature_cols=cohorts.student_features)
        cif = np.asarray(m.predict_cif(test))
        ctd = competing_risk_c_index(cif, test_dur, test_ev)
        d = predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_ev)
        ctd1.append(float(ctd[0]))
        ctd2.append(float(ctd[1]))
        dev.append(float(d))
        cifs.append(cif)
        if r == 0:
            pretrained = {k: v.clone() for k, v in m.net.state_dict().items()}
    return {
        "ctd1": np.array(ctd1), "ctd2": np.array(ctd2), "dev": np.array(dev),
        "cifs": np.stack(cifs, axis=0), "pretrained": pretrained,
    }


def run_sgld(method, teachers, cohorts, time_grid, pretrained, seed):
    test = cohorts.test
    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    base = build_sgld(method, teachers, time_grid)
    chain_seeds = [100 * seed + ci for ci in range(N_CHAINS)]
    if COLD_START:
        # Classic SGLD: random-init chains, no warm-start from the AdamW MAP.
        sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=chain_seeds)
    else:
        sampler = WarmStartMultiChainSampler(
            base, pretrained, n_chains=N_CHAINS, seeds=chain_seeds,
        )
    sampler.fit(cohorts.student, feature_cols=cohorts.student_features)
    lead = sampler.lead_chain
    chains = []
    for chain in sampler.chains:
        cifs, c1s, c2s, devs = [], [], [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            cifs.append(cif)
            ctd = competing_risk_c_index(cif, test_dur, test_ev)
            c1s.append(float(ctd[0])); c2s.append(float(ctd[1]))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_ev))
        chains.append({
            "cifs": np.stack(cifs, axis=0),
            "ctd1": np.array(c1s), "ctd2": np.array(c2s), "dev": np.array(devs),
        })
    return chains


def per_entry(cif_samples, true_cif):
    q = np.quantile(cif_samples, [0.025, 0.5, 0.975], axis=0)
    return {
        "median": q[1], "width": q[2] - q[0],
        "abs_err": np.abs(q[1] - true_cif),
        "inside": (true_cif >= q[0]) & (true_cif <= q[2]),
        "q025": q[0], "q975": q[2],
    }


def chain_diagnostics(chains):
    """Gelman-Rubin R-hat and ESS across warm-start chains.

    The scalar functional per posterior draw is the test-cohort mean CIF at the
    final horizon, evaluated separately per cause. Returns ``(max R-hat, min
    ESS)`` over the causes so the worst-mixing functional drives the reported
    convergence diagnostic. NaN-safe if a chain is degenerate.
    """
    rhats, esss = [], []
    for cause in range(NUM_RISKS):
        # [M chains, N draws]: per-chain, per-draw cohort-mean CIF at last horizon.
        fn = np.stack(
            [c["cifs"][:, :, cause, -1].mean(axis=1) for c in chains], axis=0
        )
        if fn.shape[0] >= 2 and fn.shape[1] >= 2:
            try:
                rhats.append(gelman_rubin_rhat(fn))
                esss.append(effective_sample_size(fn))
            except (ValueError, ZeroDivisionError):
                pass
    rhat = float(np.nanmax(rhats)) if rhats else float("nan")
    ess = float(np.nanmin(esss)) if esss else float("nan")
    return rhat, ess


# ---------- Calibration / reliability ----------

def reliability_data(pred_median, true_cif, n_bins=10):
    """Bin predicted CIF values, compute mean predicted vs mean true within each bin.

    pred_median, true_cif: arrays of equal shape (flattened to 1D).
    Returns (bin_mid_pred, bin_mean_true, bin_count).
    """
    p = pred_median.flatten()
    t = true_cif.flatten()
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bin_pred = np.full(n_bins, np.nan)
    bin_true = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_pred[b] = float(np.mean(p[mask]))
            bin_true[b] = float(np.mean(t[mask]))
            bin_count[b] = int(mask.sum())
    return bin_pred, bin_true, bin_count


def width_stratified(pe, cause, n_bins=10):
    """Bin entries by CI width (per cause). Returns (bin_w, mean_err, coverage, count)."""
    w = pe["width"][:, cause, :].flatten()
    e = pe["abs_err"][:, cause, :].flatten()
    inside = pe["inside"][:, cause, :].flatten().astype(float)
    edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(w, edges[1:-1]), 0, n_bins - 1)
    bin_w = np.full(n_bins, np.nan)
    bin_e = np.full(n_bins, np.nan)
    bin_cov = np.full(n_bins, np.nan)
    bin_n = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_w[b] = float(np.mean(w[mask]))
            bin_e[b] = float(np.mean(e[mask]))
            bin_cov[b] = float(np.mean(inside[mask]))
            bin_n[b] = int(mask.sum())
    return bin_w, bin_e, bin_cov, bin_n


# ---------- Figures ----------

def fig_reliability(results):
    """Reliability diagram per method, aggregated across seeds (pool entries).

    Bin entries by SGLD posterior median; plot mean true CIF (closed-form ground
    truth) per bin against the bin's mean predicted value. Identity line = perfect calibration.
    """
    fig, axes = plt.subplots(1, len(METHODS), figsize=(3.6 * len(METHODS), 4.0))
    if len(METHODS) == 1:
        axes = [axes]
    for ax, method in zip(axes, METHODS):
        # Pool across seeds and across causes
        all_pred, all_true = [], []
        for seed in SEEDS:
            pe = results[(seed, method)]["per_entry_sgld"]
            all_pred.append(pe["median"].flatten())
            all_true.append(results[(seed, method)]["true_cif"].flatten())
        p = np.concatenate(all_pred)
        t = np.concatenate(all_true)
        bp, bt, bn = reliability_data(p, t, n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
        ax.plot(bp, bt, "o-", color="#1f77b4", label="SGLD")
        # Adam reliability for comparison
        all_pred_a, all_true_a = [], []
        for seed in SEEDS:
            pe_a = results[(seed, method)]["per_entry_adam"]
            all_pred_a.append(pe_a["median"].flatten())
            all_true_a.append(results[(seed, method)]["true_cif"].flatten())
        p_a = np.concatenate(all_pred_a)
        t_a = np.concatenate(all_true_a)
        bp_a, bt_a, _ = reliability_data(p_a, t_a, n_bins=10)
        ax.plot(bp_a, bt_a, "s--", color="#ff7f0e", label="Adam")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(METHOD_LABEL[method], fontsize=10)
        ax.set_xlabel("predicted CIF (bin mean)")
        ax.set_ylabel("true CIF (bin mean)")
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle(f"Reliability diagram (10 bins, pooled across {len(SEEDS)} seeds, "
                 f"both causes & all time-points)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "uq_calibration_reliability.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_width_stratified(results):
    """Width-stratified mean |error| (top row) and conditional coverage (bottom row).

    Two columns: cause 1 and cause 2. Each curve = one method, pooled across seeds.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.cm.tab10.colors
    for j in range(2):
        ax_err = axes[0, j]
        ax_cov = axes[1, j]
        for mi, method in enumerate(METHODS):
            # Pool entries across seeds, then bin
            ws, es, ins = [], [], []
            for seed in SEEDS:
                pe = results[(seed, method)]["per_entry_sgld"]
                ws.append(pe["width"][:, j, :].flatten())
                es.append(pe["abs_err"][:, j, :].flatten())
                ins.append(pe["inside"][:, j, :].flatten().astype(float))
            w = np.concatenate(ws); e = np.concatenate(es); inside = np.concatenate(ins)
            n_bins = 10
            edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
            bin_idx = np.clip(np.digitize(w, edges[1:-1]), 0, n_bins - 1)
            bw, be, bcov = [], [], []
            for b in range(n_bins):
                m = bin_idx == b
                if m.sum() > 0:
                    bw.append(np.mean(w[m]))
                    be.append(np.mean(e[m]))
                    bcov.append(np.mean(inside[m]))
            ax_err.plot(bw, be, marker="o", color=colors[mi], label=METHOD_LABEL[method])
            ax_cov.plot(bw, bcov, marker="s", color=colors[mi], label=METHOD_LABEL[method])
        ax_err.set_xlabel("CI width (bin mean)")
        ax_err.set_ylabel("mean $|$err$|$")
        ax_err.set_title(f"Cause {j+1}: |error| vs width")
        ax_err.legend(fontsize=7); ax_err.grid(ls=":", alpha=0.4)
        ax_cov.set_xlabel("CI width (bin mean)")
        ax_cov.set_ylabel("coverage rate")
        ax_cov.axhline(0.95, color="gray", ls="--", lw=0.7)
        ax_cov.set_title(f"Cause {j+1}: coverage vs width")
        ax_cov.legend(fontsize=7); ax_cov.grid(ls=":", alpha=0.4)
        ax_cov.set_ylim(0, 1.05)
    fig.suptitle(f"Width-stratified UQ diagnostics (pooled across {len(SEEDS)} seeds, 10 bins)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "uq_calibration_width_stratified.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_correlations(correlations):
    """Per method, scatter+box of Spearman rho across seeds for cause 1 and cause 2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cause in zip(axes, [1, 2]):
        data, labels = [], []
        for method in METHODS:
            vals = [correlations[(seed, method)][f"cause{cause}"] for seed in SEEDS]
            data.append(vals)
            labels.append(METHOD_LABEL[method].replace(" -> ", "\n->\n"))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#fd8d3c")
        # Overlay individual seed dots
        for i, vals in enumerate(data):
            ax.scatter(np.full(len(vals), i + 1), vals, color="black", s=12, alpha=0.6)
        ax.axhline(0, color="gray", ls="--", lw=0.7)
        ax.set_title(f"Spearman ρ(width, |error|) — cause {cause}")
        ax.set_ylabel("ρ")
        ax.grid(ls=":", alpha=0.4, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=8)
    fig.suptitle(f"Width-vs-error correlation across {len(SEEDS)} seeds",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "uq_calibration_correlations.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------- Report ----------

def write_report(results, correlations):
    md = []
    md.append("# UQ + calibration probe — multi-seed at one cell")
    md.append("")
    md.append("**Audience:** Jian, Kevin.")
    md.append(f"**Cell:** TEACHER_N={TEACHER_N}, STUDENT_N={STUDENT_N}, "
              f"TEST_N={TEST_N}, teacher_feature_quality=\"{TEACHER_FEATURE_QUALITY}\". "
              f"Seeds = {SEEDS}.")
    _init = "cold-start (random init)" if COLD_START else "warm-started from AdamW"
    md.append(f"**SGLD:** ε₀={EPS0:.0e}→{EPS_T:.0e}, γ={GAMMA}, noise_scale σ={NOISE_SCALE} "
              f"(T≈N·σ²={STUDENT_N*NOISE_SCALE**2:.0f}), "
              f"{SGLD_EPOCHS} ep ({BURNIN_EPOCHS} burnin), "
              f"{N_CHAINS} chains × {SAMPLES_PER_CHAIN} draws, {_init}.")
    md.append("")
    md.append("## 1. Headline metrics per method (median across seeds)")
    md.append("")
    md.append("**SGLD intervals pool all warm-start chains (no test-set chain selection).** "
              "SGLD Ctd is computed on the pooled posterior-median CIF; SGLD Dev is the "
              "posterior-mean deviance over all draws. R-hat / ESS are Gelman-Rubin "
              "diagnostics on the cohort-mean CIF functional (worst cause). Coverage is "
              "vs the closed-form true CIF; nominal target 0.95.")
    md.append("")
    md.append("| Method | Adam Ctd1 | Adam Ctd2 | Adam Dev | SGLD Ctd1 | SGLD Ctd2 | SGLD Dev | SGLD cov | SGLD wid | R-hat | ESS | ρ(c1) | ρ(c2) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for method in METHODS:
        a_c1, a_c2, a_dev = [], [], []
        s_c1, s_c2, s_dev, s_cov, s_wid, s_rhat, s_ess = [], [], [], [], [], [], []
        rho_c1, rho_c2 = [], []
        for seed in SEEDS:
            r = results[(seed, method)]
            a_c1.append(np.median(r["adam_ctd1"]))
            a_c2.append(np.median(r["adam_ctd2"]))
            a_dev.append(np.median(r["adam_dev"]))
            s_c1.append(r["sgld_ctd1"])
            s_c2.append(r["sgld_ctd2"])
            s_dev.append(r["sgld_dev"])
            pe = r["per_entry_sgld"]
            s_cov.append(pe["inside"].mean())
            s_wid.append(pe["width"].mean())
            s_rhat.append(r["rhat"])
            s_ess.append(r["ess"])
            rho_c1.append(correlations[(seed, method)]["cause1"])
            rho_c2.append(correlations[(seed, method)]["cause2"])
        md.append(f"| {METHOD_LABEL[method]} | "
                  f"{np.median(a_c1):.3f} | {np.median(a_c2):.3f} | {np.median(a_dev):.3f} | "
                  f"{np.median(s_c1):.3f} | {np.median(s_c2):.3f} | {np.median(s_dev):.3f} | "
                  f"{np.median(s_cov):.3f} | {np.median(s_wid):.3f} | "
                  f"{np.nanmedian(s_rhat):.2f} | {np.nanmedian(s_ess):.0f} | "
                  f"{np.median(rho_c1):+.3f} | {np.median(rho_c2):+.3f} |")
    md.append("")
    md.append("## 2. Per-seed Spearman ρ(width, |error|) on SGLD posterior")
    md.append("")
    md.append("| Method | Seed | ρ overall | ρ cause 1 | ρ cause 2 |")
    md.append("|---|---|---|---|---|")
    for method in METHODS:
        for seed in SEEDS:
            c = correlations[(seed, method)]
            md.append(f"| {METHOD_LABEL[method]} | {seed} | "
                      f"{c['overall']:+.3f} | {c['cause1']:+.3f} | {c['cause2']:+.3f} |")
    md.append("")
    md.append("## 3. Figures")
    md.append("")
    md.append("### Reliability diagram (10 bins; SGLD posterior median vs closed-form true CIF, pooled across seeds)")
    md.append("![reliability](uq_calibration_reliability.png)")
    md.append("")
    md.append("### Width-stratified diagnostics")
    md.append("![width-stratified](uq_calibration_width_stratified.png)")
    md.append("")
    md.append("Top row: mean |error| as a function of CI-width bin (monotone-increasing curve = informative posterior). "
              "Bottom row: conditional coverage per width bin (curve approaching 0.95 = well-calibrated).")
    md.append("")
    md.append("### Spearman ρ across seeds")
    md.append("![correlations](uq_calibration_correlations.png)")
    md.append("")
    md.append("## 4. Files")
    md.append("")
    md.append("- `uq_calibration_data.npz` — per-(seed, method) per-entry arrays")
    md.append("- `uq_calibration_reliability.{pdf,png}`")
    md.append("- `uq_calibration_width_stratified.{pdf,png}`")
    md.append("- `uq_calibration_correlations.{pdf,png}`")

    out = OUT_DIR / "uq_calibration_report.md"
    out.write_text("\n".join(md) + "\n")


# ---------- Main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print(f"=== UQ + calibration probe ===")
    print(f"Cell: TEACHER_N={TEACHER_N}, STUDENT_N={STUDENT_N}, "
          f"quality={TEACHER_FEATURE_QUALITY}")
    print(f"SGLD: eps0={EPS0:.0e}->{EPS_T:.0e} gamma={GAMMA} noise_scale={NOISE_SCALE} "
          f"init={'COLD' if COLD_START else 'warm'} (T~N*sigma^2={STUDENT_N*NOISE_SCALE**2:.0f})")
    print(f"Seeds: {SEEDS}")
    print(f"Methods: {METHODS}")

    results = {}        # (seed, method) -> rich dict
    correlations = {}   # (seed, method) -> {overall, cause1, cause2}
    grid_cuts = {}      # seed -> right-endpoint cuts (saved for auditability)

    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n=== Seed {seed} ===", flush=True)
        cohorts, time_grid = setup_cohorts(seed)
        teachers = train_teachers(cohorts, time_grid, METHODS)
        true_cif = true_cif_at_grid(cohorts.test, time_grid, num_risks=NUM_RISKS)
        grid_cuts[seed] = np.asarray(time_grid.cuts, dtype=float)
        test_dur_arr = cohorts.test["duration"].to_numpy()
        test_ev_arr = cohorts.test["event"].to_numpy()

        for method in METHODS:
            t_m = time.time()
            adam = run_adamw(method, teachers, cohorts, time_grid, seed)
            chains = run_sgld(method, teachers, cohorts, time_grid, adam["pretrained"], seed)

            # De-leaked reporting: POOL all warm-start chains (no test-set chain
            # selection). All chains start from the same AdamW MAP, so pooling is
            # a legitimate multi-chain posterior estimate; convergence is then
            # disclosed via R-hat / ESS rather than hidden by best-chain picking.
            pooled_cifs = np.concatenate([c["cifs"] for c in chains], axis=0)
            pe_sgld = per_entry(pooled_cifs, true_cif)
            pe_adam = per_entry(adam["cifs"], true_cif)
            per_chain_pe = [per_entry(c["cifs"], true_cif) for c in chains]

            # SGLD point summary on the pooled posterior-median CIF surface.
            sgld_ctd = competing_risk_c_index(pe_sgld["median"], test_dur_arr, test_ev_arr)
            sgld_ctd1, sgld_ctd2 = float(sgld_ctd[0]), float(sgld_ctd[1])
            sgld_dev = float(np.mean([np.mean(c["dev"]) for c in chains]))
            rhat, ess = chain_diagnostics(chains)

            results[(seed, method)] = {
                "adam_ctd1": adam["ctd1"], "adam_ctd2": adam["ctd2"],
                "adam_dev": adam["dev"],
                "sgld_ctd1": sgld_ctd1, "sgld_ctd2": sgld_ctd2, "sgld_dev": sgld_dev,
                "per_entry_sgld": pe_sgld,
                "per_entry_adam": pe_adam,
                "per_chain_pe": per_chain_pe,
                "rhat": rhat, "ess": ess,
                "true_cif": true_cif,
            }
            rho_overall, _ = spearmanr(pe_sgld["width"].flatten(), pe_sgld["abs_err"].flatten())
            rho_c1, _ = spearmanr(pe_sgld["width"][:, 0, :].flatten(),
                                  pe_sgld["abs_err"][:, 0, :].flatten())
            rho_c2, _ = spearmanr(pe_sgld["width"][:, 1, :].flatten(),
                                  pe_sgld["abs_err"][:, 1, :].flatten())
            correlations[(seed, method)] = {
                "overall": float(rho_overall),
                "cause1": float(rho_c1), "cause2": float(rho_c2),
            }
            print(f"  {method:18s}  Adam Ctd1={np.median(adam['ctd1']):.3f} "
                  f"SGLD Ctd1={sgld_ctd1:.3f}(pooled) "
                  f"cov={pe_sgld['inside'].mean():.3f} wid={pe_sgld['width'].mean():.3f} "
                  f"Rhat={rhat:.2f} ESS={ess:.0f} "
                  f"ρ(c1)={rho_c1:+.3f} ρ(c2)={rho_c2:+.3f} "
                  f"({time.time()-t_m:.0f}s)", flush=True)
        print(f"  seed {seed} done in {time.time()-t_seed:.0f}s", flush=True)

    # ---- Save raw ----
    save = {}
    for (seed, method), r in results.items():
        prefix = f"seed{seed}_{method}"
        for k in ["median", "width", "abs_err", "inside", "q025", "q975"]:
            v = r["per_entry_sgld"][k]
            save[f"{prefix}_sgld_{k}"] = v.astype(np.int8) if k == "inside" else v
            v_a = r["per_entry_adam"][k]
            save[f"{prefix}_adam_{k}"] = v_a.astype(np.int8) if k == "inside" else v_a
        # Per-chain per-entry summaries so chain-0 / pooled / validation
        # reanalysis is possible later WITHOUT the raw draws (and without leakage).
        for ci, pe in enumerate(r["per_chain_pe"]):
            for k in ["median", "width", "inside", "q025", "q975"]:
                v = pe[k]
                save[f"{prefix}_chain{ci}_{k}"] = v.astype(np.int8) if k == "inside" else v
        save[f"{prefix}_adam_ctd1"] = r["adam_ctd1"]
        save[f"{prefix}_adam_ctd2"] = r["adam_ctd2"]
        save[f"{prefix}_adam_dev"]  = r["adam_dev"]
        save[f"{prefix}_sgld_ctd1"] = r["sgld_ctd1"]
        save[f"{prefix}_sgld_ctd2"] = r["sgld_ctd2"]
        save[f"{prefix}_sgld_dev"]  = r["sgld_dev"]
        save[f"{prefix}_rhat"] = r["rhat"]
        save[f"{prefix}_ess"]  = r["ess"]
    for seed in SEEDS:
        save[f"seed{seed}_true_cif"] = results[(seed, METHODS[0])]["true_cif"]
        save[f"seed{seed}_cuts"] = grid_cuts[seed]
    # Config metadata for auditability (so coverage can be re-checked offline).
    save["cfg_teacher_n"] = TEACHER_N
    save["cfg_student_n"] = STUDENT_N
    save["cfg_test_n"] = TEST_N
    save["cfg_quality"] = TEACHER_FEATURE_QUALITY
    save["cfg_eps0"] = EPS0
    save["cfg_epsT"] = EPS_T
    save["cfg_gamma"] = GAMMA
    save["cfg_sgld_epochs"] = SGLD_EPOCHS
    save["cfg_n_chains"] = N_CHAINS
    save["cfg_samples_per_chain"] = SAMPLES_PER_CHAIN
    save["cfg_selection"] = "pooled_all_chains"
    np.savez(OUT_DIR / "uq_calibration_data.npz", **save)
    print(f"\nData saved to {OUT_DIR / 'uq_calibration_data.npz'}", flush=True)

    fig_reliability(results)
    fig_width_stratified(results)
    fig_correlations(correlations)
    print("Figures saved.", flush=True)

    write_report(results, correlations)
    print(f"Report saved to {OUT_DIR / 'uq_calibration_report.md'}", flush=True)


if __name__ == "__main__":
    main()
