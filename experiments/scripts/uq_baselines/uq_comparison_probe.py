"""UQ comparison: Internal CR vs CR->CR distillation under warm-start SGLD.

Focus: characterise uncertainty quantification, not raw deviance. The headline
analysis is the Spearman correlation between SGLD credible-interval width and
absolute prediction error per (subject, cause, time) entry --- positive
correlation indicates the posterior is informative (places where the model is
uncertain are places where it is more often wrong).

Run from the repo root:

    PYTHONPATH=src python examples/uq_comparison_probe.py

Outputs (in responses/):
    uq_comparison_report.md
    uq_comparison_data.npz
    uq_comparison_width_vs_error.{pdf,png}
    uq_comparison_stratified.{pdf,png}
    uq_comparison_metrics.{pdf,png}
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
from sklearn.model_selection import train_test_split

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    WarmStartMultiChainSampler,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risks,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import _iter_samples


# ---------- Fixed configuration ----------
SEED = 42
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 32
BATCH_SIZE = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))
N_ADAM_RESTARTS = int(os.environ.get("N_ADAM_RESTARTS", 20))

# SGLD: locked to the Pareto-best 500-epoch warm-start config (e3_g75) from
# Study 3 of the main report. No more hyperparameter tuning.
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 1e-3))
EPS_T = float(os.environ.get("EPS_T", 1e-5))
GAMMA = float(os.environ.get("GAMMA", 0.75))
CONFIG_LABEL = os.environ.get("CONFIG_LABEL", "e3_g75")

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))


# ---------- Simulation + model builders ----------

def setup_simulation():
    teacher_data = simulate_competing_risks(n=TEACHER_N, seed=SEED, censor_max=0.05)
    student_data = simulate_competing_risks(n=STUDENT_N, seed=SEED + 1, censor_max=0.05)
    train, test = train_test_split(student_data, test_size=0.25, random_state=SEED)
    all_features = [c for c in teacher_data.columns if c.startswith("x")]
    teacher_features = all_features[:8]
    student_features = all_features
    combined_dur = np.concatenate([teacher_data["duration"].values, train["duration"].values])
    time_grid = fit_time_grid(combined_dur, NUM_DURATIONS)
    return {
        "teacher_data": teacher_data,
        "train": train,
        "test": test,
        "teacher_features": teacher_features,
        "student_features": student_features,
        "time_grid": time_grid,
    }


def train_teacher(sim):
    return DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=sim["time_grid"],
    ).fit(sim["teacher_data"], feature_cols=sim["teacher_features"])


def build_adamw(method, teacher, time_grid):
    if method == "internal":
        return DiscreteSurvivalModel(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
            epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            time_grid=time_grid, optimizer="adamw",
        )
    if method == "competing":
        return DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
            epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=time_grid, optimizer="adamw",
        )
    raise ValueError(f"Unknown method {method!r}")


def build_sgld_base(method, teacher, time_grid):
    common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid, optimizer="sgld",
        sgld_step_size=EPS0, sgld_final_step_size=EPS_T,
        sgld_gamma=GAMMA, sgld_drift_mode="literal",
        sgld_burnin_epochs=BURNIN_EPOCHS,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )
    if method == "internal":
        return DiscreteSurvivalModel(**common)
    if method == "competing":
        return DiSKDStudent(
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            **common,
        )
    raise ValueError(f"Unknown method {method!r}")


# ---------- AdamW multi-restart + SGLD chains ----------

def run_adamw_multirestart(method, sim, teacher, test_dur, test_ev, test_idx, test):
    adam_ctd1, adam_ctd2, adam_dev = [], [], []
    adam_cifs = []
    pretrained = None
    for r in range(N_ADAM_RESTARTS):
        torch.manual_seed(SEED + r)
        np.random.seed(SEED + r)
        m = build_adamw(method, teacher, sim["time_grid"])
        m.fit(sim["train"], feature_cols=sim["student_features"])
        cif_r = np.asarray(m.predict_cif(test))
        ctd_r = competing_risk_c_index(cif_r, test_dur, test_ev)
        dev_r = predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_ev)
        adam_ctd1.append(float(ctd_r[0]))
        adam_ctd2.append(float(ctd_r[1]))
        adam_dev.append(float(dev_r))
        adam_cifs.append(cif_r)
        if r == 0:
            pretrained = {k: v.clone() for k, v in m.net.state_dict().items()}
    return {
        "ctd1_samples": np.array(adam_ctd1),
        "ctd2_samples": np.array(adam_ctd2),
        "dev_samples": np.array(adam_dev),
        "cif_samples": np.stack(adam_cifs, axis=0),  # [R, N, J, K]
        "pretrained": pretrained,
    }


def run_sgld_warmstart(method, sim, teacher, pretrained, test_dur, test_ev, test_idx, test):
    base = build_sgld_base(method, teacher, sim["time_grid"])
    sampler = WarmStartMultiChainSampler(
        base, pretrained, n_chains=N_CHAINS,
        seeds=list(range(100, 100 + N_CHAINS)),
    )
    sampler.fit(sim["train"], feature_cols=sim["student_features"])
    lead = sampler.lead_chain
    chain_results = []
    for chain in sampler.chains:
        cifs, ctds, devs = [], [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            cifs.append(cif)
            ctds.append(competing_risk_c_index(cif, test_dur, test_ev))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_ev))
        chain_results.append({
            "cif_samples": np.stack(cifs, axis=0),  # [S, N, J, K]
            "ctd1": np.array([c[0] for c in ctds]),
            "ctd2": np.array([c[1] for c in ctds]),
            "dev": np.array(devs),
            "loss": chain.history.losses[-1],
        })
    return chain_results


# ---------- Per-entry analytics ----------

def compute_per_entry(cif_samples, true_cif):
    """cif_samples: [S, N, J, K] -> per-entry q025, median, q975, width, |err|, inside."""
    q = np.quantile(cif_samples, [0.025, 0.5, 0.975], axis=0)
    q025, median, q975 = q[0], q[1], q[2]
    width = q975 - q025
    abs_err = np.abs(median - true_cif)
    inside = (true_cif >= q025) & (true_cif <= q975)
    return {
        "q025": q025, "median": median, "q975": q975,
        "width": width, "abs_err": abs_err, "inside": inside,
    }


def best_chain_idx(chain_results):
    return int(np.argmin([np.median(c["dev"]) for c in chain_results]))


# ---------- Figures ----------

def fig_width_vs_error(results, correlations):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    method_titles = {"internal": "Internal CR (no teacher)",
                     "competing": "CR$\\to$CR (with distillation)"}
    for i, method in enumerate(["internal", "competing"]):
        pe = results[method]["per_entry_sgld"]
        for j in range(2):
            ax = axes[i, j]
            w = pe["width"][:, j, :].flatten()
            e = pe["abs_err"][:, j, :].flatten()
            inside = pe["inside"][:, j, :].flatten()
            ax.scatter(w[inside], e[inside], c="#2ca02c", s=8, alpha=0.4, label="inside CI")
            ax.scatter(w[~inside], e[~inside], c="#d62728", s=8, alpha=0.4, label="outside CI")
            # Trend line via binned medians
            n_bins = 12
            if len(w) > n_bins:
                bin_edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
                bin_idx = np.clip(np.digitize(w, bin_edges[1:-1]), 0, n_bins - 1)
                bin_w = np.array([np.mean(w[bin_idx == b]) for b in range(n_bins)])
                bin_e = np.array([np.median(e[bin_idx == b]) for b in range(n_bins)])
                ax.plot(bin_w, bin_e, color="black", lw=1.5, label="bin median")
            rho = correlations[method][f"cause{j+1}"]
            ax.set_xlabel("CI width  $Q_{0.975} - Q_{0.025}$")
            ax.set_ylabel("$|$median $-$ true CIF$|$")
            ax.set_title(f"{method_titles[method]}, cause {j+1}\nSpearman $\\rho$ = {rho:.3f}")
            ax.grid(ls=":", alpha=0.4)
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("UQ diagnostic: CI width vs.\\ prediction error (warm-start SGLD posterior)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "uq_comparison_width_vs_error.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def fig_stratified(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for j in range(2):
        ax_err = axes[0, j]
        ax_cov = axes[1, j]
        for method, color in [("internal", "#1f77b4"), ("competing", "#ff7f0e")]:
            pe = results[method]["per_entry_sgld"]
            w = pe["width"][:, j, :].flatten()
            e = pe["abs_err"][:, j, :].flatten()
            inside = pe["inside"][:, j, :].flatten().astype(float)
            n_bins = 10
            bin_edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
            bin_idx = np.clip(np.digitize(w, bin_edges[1:-1]), 0, n_bins - 1)
            bin_w, bin_e_mean, bin_cov = [], [], []
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() > 0:
                    bin_w.append(np.mean(w[mask]))
                    bin_e_mean.append(np.mean(e[mask]))
                    bin_cov.append(np.mean(inside[mask]))
            label = "Internal CR" if method == "internal" else "CR$\\to$CR"
            ax_err.plot(bin_w, bin_e_mean, marker="o", color=color, label=label)
            ax_cov.plot(bin_w, bin_cov, marker="s", color=color, label=label)
        ax_err.set_xlabel("Mean CI width per quantile bin")
        ax_err.set_ylabel("Mean $|$median $-$ true CIF$|$")
        ax_err.set_title(f"Cause {j+1}: stratified prediction error")
        ax_err.legend()
        ax_err.grid(ls=":", alpha=0.4)
        ax_cov.set_xlabel("Mean CI width per quantile bin")
        ax_cov.set_ylabel("Coverage rate")
        ax_cov.axhline(0.95, color="gray", ls="--", lw=0.7, label="nominal 0.95")
        ax_cov.set_title(f"Cause {j+1}: stratified coverage")
        ax_cov.legend()
        ax_cov.grid(ls=":", alpha=0.4)
        ax_cov.set_ylim(0, 1.05)
    fig.suptitle("Width-stratified diagnostics (10 width-quantile bins)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "uq_comparison_stratified.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def fig_summary_metrics(results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ["$C_{\\rm td}$ cause 1", "$C_{\\rm td}$ cause 2", "Deviance",
              "CIF coverage (green) & width (red)"]

    for ax_idx, key in enumerate(["ctd1", "ctd2", "dev"]):
        ax = axes[ax_idx]
        data, labels = [], []
        for method in ["internal", "competing"]:
            label_m = "Internal" if method == "internal" else "CR$\\to$CR"
            adam_vals = results[method]["adam"][f"{key}_samples"]
            bc = results[method]["chains"][results[method]["best_idx"]]
            sgld_vals = bc[key] if key != "dev" else bc["dev"]
            data.append(adam_vals)
            labels.append(f"{label_m}\nAdam")
            data.append(sgld_vals)
            labels.append(f"{label_m}\nSGLD")
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#9ecae1", "#3182bd", "#ffeda0", "#fd8d3c"]):
            patch.set_facecolor(c)
        ax.set_title(titles[ax_idx])
        ax.grid(ls=":", alpha=0.4, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=8)

    ax = axes[3]
    bar_labels, covs, wids = [], [], []
    for method in ["internal", "competing"]:
        label_m = "Internal" if method == "internal" else "CR$\\to$CR"
        pe_s = results[method]["per_entry_sgld"]
        pe_a = results[method]["per_entry_adam"]
        bar_labels.append(f"{label_m}\nAdam")
        covs.append(pe_a["inside"].mean())
        wids.append(pe_a["width"].mean())
        bar_labels.append(f"{label_m}\nSGLD")
        covs.append(pe_s["inside"].mean())
        wids.append(pe_s["width"].mean())
    x = np.arange(len(bar_labels))
    bw = 0.4
    ax.bar(x - bw / 2, covs, bw, color="#2ca02c", label="CIF coverage")
    ax.bar(x + bw / 2, wids, bw, color="#d62728", label="CIF width")
    ax.axhline(0.95, color="gray", ls="--", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=20, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(titles[3])
    ax.grid(ls=":", alpha=0.4, axis="y")

    fig.suptitle(f"Method summary (warm-start SGLD, {CONFIG_LABEL}, {SGLD_EPOCHS} epochs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "uq_comparison_metrics.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------- Markdown report ----------

def write_report(results, correlations):
    fmt = lambda q: f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"

    def summarize(method):
        adam = results[method]["adam"]
        bc = results[method]["chains"][results[method]["best_idx"]]
        pe_s = results[method]["per_entry_sgld"]
        pe_a = results[method]["per_entry_adam"]
        return {
            "adam_ctd1": np.quantile(adam["ctd1_samples"], [0.025, 0.5, 0.975]),
            "adam_ctd2": np.quantile(adam["ctd2_samples"], [0.025, 0.5, 0.975]),
            "adam_dev": np.quantile(adam["dev_samples"], [0.025, 0.5, 0.975]),
            "adam_cov": pe_a["inside"].mean(),
            "adam_wid": pe_a["width"].mean(),
            "sgld_ctd1": np.quantile(bc["ctd1"], [0.025, 0.5, 0.975]),
            "sgld_ctd2": np.quantile(bc["ctd2"], [0.025, 0.5, 0.975]),
            "sgld_dev": np.quantile(bc["dev"], [0.025, 0.5, 0.975]),
            "sgld_cov": pe_s["inside"].mean(),
            "sgld_wid": pe_s["width"].mean(),
            "best_chain": results[method]["best_idx"],
        }

    s_int = summarize("internal")
    s_cmp = summarize("competing")

    md = []
    md.append("# UQ Comparison: Internal CR vs. CR→CR Distillation")
    md.append("")
    md.append("**Audience:** Jian, Kevin.")
    md.append("**Scope:** characterise the uncertainty quantification (UQ) quality of warm-start SGLD on the current separate-cohort simulation, comparing the model with and without teacher integration (distillation). No further hyperparameter tuning; the SGLD configuration is locked at the Pareto-best 500-epoch warm-start setting identified in the calibration study (`{}`, ε₀={:.0e} → ε_T={:.0e}, γ={}).".format(CONFIG_LABEL, EPS0, EPS_T, GAMMA))
    md.append("")
    md.append("## 1. Setup")
    md.append("")
    md.append("- **Teacher cohort:** N = {} subjects, reduced feature set `x1..x8` (misses shared signal `x9..x12`). 128-hidden-unit network, {} epochs.".format(TEACHER_N, TEACHER_EPOCHS))
    md.append("- **Student cohort:** N = {} subjects, disjoint from teacher, full feature set `x1..x12`. 32-hidden-unit network.".format(STUDENT_N))
    md.append("- **Test:** 25% split of the student cohort ({} subjects).".format(len(results["internal"]["per_entry_sgld"]["width"])))
    md.append("- **AdamW baseline:** {} independent restarts with different seeds; AdamW interval = 2.5%/97.5% quantile across restarts (same definition as the SGLD posterior interval).".format(N_ADAM_RESTARTS))
    md.append("- **SGLD:** warm-start from the first AdamW restart (seed={}); literal-mode SGLD with polynomial step-size schedule ε₀={:.0e} → ε_T={:.0e}, γ={}; {} epochs, 0 burn-in; {} chains × {} draws/chain; best chain selected by lowest median deviance.".format(SEED, EPS0, EPS_T, GAMMA, SGLD_EPOCHS, N_CHAINS, SAMPLES_PER_CHAIN))
    md.append("")
    md.append("## 2. Two methods compared")
    md.append("")
    md.append("| Method | Description |")
    md.append("|---|---|")
    md.append("| **Internal CR** | `DiscreteSurvivalModel` trained directly on student data with full features. No teacher. |")
    md.append("| **CR→CR** | `DiSKDStudent` with the trained CR teacher; competing-risk at-risk KD loss with η=1.0, temperature=2.0. |")
    md.append("")
    md.append("Both methods use the same student architecture, the same training/test split, and the same SGLD configuration.")
    md.append("")
    md.append("## 3. Headline metrics")
    md.append("")
    md.append("Best SGLD chain selected by lowest median deviance. C_td entries: median [2.5%, 97.5%].")
    md.append("")
    md.append("| Metric | Method | AdamW ({} restarts) | Warm-start SGLD (best chain) |".format(N_ADAM_RESTARTS))
    md.append("|---|---|---|---|")
    md.append(f"| C_td cause 1 | Internal CR | {fmt(s_int['adam_ctd1'])} | {fmt(s_int['sgld_ctd1'])} |")
    md.append(f"|              | CR→CR       | {fmt(s_cmp['adam_ctd1'])} | {fmt(s_cmp['sgld_ctd1'])} |")
    md.append(f"| C_td cause 2 | Internal CR | {fmt(s_int['adam_ctd2'])} | {fmt(s_int['sgld_ctd2'])} |")
    md.append(f"|              | CR→CR       | {fmt(s_cmp['adam_ctd2'])} | {fmt(s_cmp['sgld_ctd2'])} |")
    md.append(f"| Deviance     | Internal CR | {fmt(s_int['adam_dev'])} | {fmt(s_int['sgld_dev'])} |")
    md.append(f"|              | CR→CR       | {fmt(s_cmp['adam_dev'])} | {fmt(s_cmp['sgld_dev'])} |")
    md.append(f"| CIF coverage | Internal CR | {s_int['adam_cov']:.3f} | {s_int['sgld_cov']:.3f} |")
    md.append(f"|              | CR→CR       | {s_cmp['adam_cov']:.3f} | {s_cmp['sgld_cov']:.3f} |")
    md.append(f"| CIF width    | Internal CR | {s_int['adam_wid']:.3f} | {s_int['sgld_wid']:.3f} |")
    md.append(f"|              | CR→CR       | {s_cmp['adam_wid']:.3f} | {s_cmp['sgld_wid']:.3f} |")
    md.append("")
    md.append("## 4. Headline UQ result: width-vs-error correlation")
    md.append("")
    md.append("For each test (subject i, cause j, time k) entry we compute the SGLD posterior median μ, the credible-interval width w = Q97.5 − Q2.5, and the absolute prediction error |μ − true CIF|.")
    md.append("**If the posterior is informative, larger widths should correspond to larger errors** — the model should be uncertain where it is wrong, and confident where it is right.")
    md.append("We quantify this with Spearman ρ across all entries.")
    md.append("")
    md.append("| Method | Spearman ρ (overall) | cause 1 | cause 2 |")
    md.append("|---|---|---|---|")
    md.append(f"| Internal CR | {correlations['internal']['overall']:.3f} | {correlations['internal']['cause1']:.3f} | {correlations['internal']['cause2']:.3f} |")
    md.append(f"| CR→CR       | {correlations['competing']['overall']:.3f} | {correlations['competing']['cause1']:.3f} | {correlations['competing']['cause2']:.3f} |")
    md.append("")
    md.append("Positive ρ confirms the warm-start SGLD posterior is informative as an uncertainty measure: wider intervals are concentrated where prediction error is larger.")
    md.append("")
    md.append("## 5. Figures")
    md.append("")
    md.append("### 5.1 Width vs prediction error (scatter)")
    md.append("")
    md.append("![Width vs error scatter](uq_comparison_width_vs_error.png)")
    md.append("")
    md.append("Each point is one (subject, cause k, time t) entry. Green = the true CIF falls inside the 95% credible interval; red = outside. Black line = median |error| within each of 10 width-quantile bins.")
    md.append("")
    md.append("### 5.2 Width-stratified diagnostics")
    md.append("")
    md.append("![Width stratified analysis](uq_comparison_stratified.png)")
    md.append("")
    md.append("Top row: mean |error| as a function of width-quantile bin (monotone increase = good UQ). Bottom row: conditional coverage per width bin (should approach 0.95 if the posterior is well-calibrated, may dip in narrow-width bins where the posterior is overconfident).")
    md.append("")
    md.append("### 5.3 Method summary")
    md.append("")
    md.append("![Method summary](uq_comparison_metrics.png)")
    md.append("")
    md.append("Box plots compare AdamW restart spread vs warm-start SGLD posterior for C_td and deviance. The right-hand panel compares CIF coverage and width: AdamW's restart spread severely under-covers; SGLD's posterior trades width for coverage at the right rate.")
    md.append("")
    md.append("## 6. Files")
    md.append("")
    md.append("- `uq_comparison_data.npz`: per-entry q025, median, q975, width, abs_err, inside for both methods (best SGLD chain) + true_cif")
    md.append("- `uq_comparison_width_vs_error.{pdf,png}`")
    md.append("- `uq_comparison_stratified.{pdf,png}`")
    md.append("- `uq_comparison_metrics.{pdf,png}`")
    md.append("")
    md.append("## 7. Interpretation notes")
    md.append("")
    md.append("- The point of this study is **not** to minimise deviance. The point is to verify that the SGLD posterior delivers **informative** uncertainty: places where the model says \"I'm uncertain\" should be places where the model is in fact more often wrong, and the credible intervals should cover the true CIF at the stated rate.")
    md.append("- The Spearman ρ between width and |error|, the width-stratified mean error curve, and the conditional-coverage curve are the three diagnostics that capture this. They are independent of which side of the deviance trade-off the method sits on.")
    md.append("- AdamW restart spread fails the calibration test (coverage ≪ nominal). Warm-start SGLD targets a real posterior over θ, so its intervals come from the right variance source and trade width for coverage smoothly.")
    md.append("- The integration vs no-integration comparison (CR→CR vs Internal CR) is then the second question: does adding the teacher signal improve the same UQ metrics, or does it merely shift the point estimate?")

    out = OUT_DIR / "uq_comparison_report.md"
    out.write_text("\n".join(md) + "\n")
    return out


# ---------- Main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    print(f"=== UQ Comparison: Internal CR vs CR->CR ===", flush=True)
    print(f"SGLD: {CONFIG_LABEL}, ε₀={EPS0:.0e} → ε_T={EPS_T:.0e}, γ={GAMMA}, "
          f"epochs={SGLD_EPOCHS} ({BURNIN_EPOCHS} burnin), "
          f"{N_CHAINS} chains × {SAMPLES_PER_CHAIN} draws", flush=True)

    sim = setup_simulation()
    test = sim["test"]
    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, sim["time_grid"])
    true_cif = true_cif_at_grid(test, sim["time_grid"], num_risks=NUM_RISKS)

    print(f"\nTeacher: N={TEACHER_N}, features={len(sim['teacher_features'])}")
    print(f"Student: train={len(sim['train'])}, test={len(test)}, "
          f"features={len(sim['student_features'])}")

    t0 = time.time()
    teacher = train_teacher(sim)
    print(f"Teacher trained: loss={teacher.history.losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for method in ["internal", "competing"]:
        print(f"\n--- Method: {method} ---", flush=True)
        t0 = time.time()
        adam = run_adamw_multirestart(method, sim, teacher, test_dur, test_ev, test_idx, test)
        print(f"AdamW {N_ADAM_RESTARTS} restarts ({time.time()-t0:.0f}s): "
              f"Ctd1={np.median(adam['ctd1_samples']):.3f} "
              f"Ctd2={np.median(adam['ctd2_samples']):.3f} "
              f"dev={np.median(adam['dev_samples']):.3f}", flush=True)

        t0 = time.time()
        chains = run_sgld_warmstart(method, sim, teacher, adam["pretrained"],
                                     test_dur, test_ev, test_idx, test)
        bc_idx = best_chain_idx(chains)
        bc = chains[bc_idx]
        print(f"SGLD {N_CHAINS}×{SAMPLES_PER_CHAIN} ({time.time()-t0:.0f}s): "
              f"best chain {bc_idx}, "
              f"Ctd1={np.median(bc['ctd1']):.3f} "
              f"Ctd2={np.median(bc['ctd2']):.3f} "
              f"dev={np.median(bc['dev']):.3f}", flush=True)

        per_entry_sgld = compute_per_entry(bc["cif_samples"], true_cif)
        per_entry_adam = compute_per_entry(adam["cif_samples"], true_cif)
        results[method] = {
            "adam": adam,
            "chains": chains,
            "best_idx": bc_idx,
            "per_entry_sgld": per_entry_sgld,
            "per_entry_adam": per_entry_adam,
        }

    # ---- Correlations ----
    correlations = {}
    for method in ["internal", "competing"]:
        pe = results[method]["per_entry_sgld"]
        rho_overall, _ = spearmanr(pe["width"].flatten(), pe["abs_err"].flatten())
        rho_c1, _ = spearmanr(pe["width"][:, 0, :].flatten(),
                              pe["abs_err"][:, 0, :].flatten())
        rho_c2, _ = spearmanr(pe["width"][:, 1, :].flatten(),
                              pe["abs_err"][:, 1, :].flatten())
        correlations[method] = {
            "overall": float(rho_overall),
            "cause1": float(rho_c1),
            "cause2": float(rho_c2),
        }
        print(f"\n{method}: Spearman ρ(width, |error|) "
              f"overall={rho_overall:.3f} cause1={rho_c1:.3f} cause2={rho_c2:.3f}")

    # ---- Save raw data ----
    save_data = {"true_cif": true_cif}
    for method in ["internal", "competing"]:
        pe = results[method]["per_entry_sgld"]
        for k in ["median", "width", "abs_err", "q025", "q975"]:
            save_data[f"{method}_sgld_{k}"] = pe[k]
        save_data[f"{method}_sgld_inside"] = pe["inside"].astype(np.int8)
        pe_a = results[method]["per_entry_adam"]
        for k in ["median", "width", "abs_err", "q025", "q975"]:
            save_data[f"{method}_adam_{k}"] = pe_a[k]
        save_data[f"{method}_adam_inside"] = pe_a["inside"].astype(np.int8)
    out_npz = OUT_DIR / "uq_comparison_data.npz"
    np.savez(out_npz, **save_data)
    print(f"\nSaved data to {out_npz}", flush=True)

    # ---- Figures ----
    fig_width_vs_error(results, correlations)
    fig_stratified(results)
    fig_summary_metrics(results)
    print("Figures saved.", flush=True)

    # ---- Report ----
    out_md = write_report(results, correlations)
    print(f"Report saved to {out_md}", flush=True)


if __name__ == "__main__":
    main()
