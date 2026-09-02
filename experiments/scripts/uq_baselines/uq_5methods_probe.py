"""UQ comparison across all 5 distillation schemes.

Simulation design (matches the UPDATED original/DiscreteSurvKD-main):
  - Uses `simulate_competing_risk_cohorts(teacher_feature_quality="reduced")`.
  - Three independent cohorts: teacher, student, test (all generated from the
    same DGP but with independent seeds).
  - Teacher: REDUCED feature subset `x1..x8` (misses shared signal block).
  - Student & test: FULL feature set `x1..x12`.
  - Overall teacher: trained on a binary any-event target on the teacher cohort.
  - Binary teachers: logistic regression on the teacher cohort with cause-specific
    indicator at horizon t_k.

Methods:
  1. Internal CR         — no teacher (baseline)
  2. CR -> CR            — DiSKD-C, teacher_type="competing"
  3. Overall -> CR       — DiSKD-O, overall (single-risk) teacher with event_any
  4. Binary-1 -> CR      — logistic teacher predicting P(event=1 by t_k)
  5. Binary-2 -> CR      — logistic teacher predicting P(event=2 by t_k)

For each method, we run:
  - N_ADAM_RESTARTS independent AdamW fits (for the Adam interval baseline)
  - 5 warm-start SGLD chains x 500 draws (locked e3_g75 config from Study 3)

Outputs:
  uq_5methods_report.md
  uq_5methods_data.npz
  uq_5methods_metrics.{pdf,png}        # boxplots: Ctd1, Ctd2, deviance per method
  uq_5methods_uq.{pdf,png}             # CIF coverage and width per method
  uq_5methods_width_vs_error.{pdf,png} # 5x2 scatter (method x cause)
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
    WarmStartMultiChainSampler,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import _iter_samples


# ---------- Configuration (locked; no hyperparameter sweep) ----------
SEED = 42
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
N_ADAM_RESTARTS = int(os.environ.get("N_ADAM_RESTARTS", 20))
HORIZON_INDEX = int(os.environ.get("HORIZON_INDEX", 7))

# SGLD: locked to e3_g75 (500 epochs, no burn-in) from the calibration study.
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 1e-3))
EPS_T = float(os.environ.get("EPS_T", 1e-5))
GAMMA = float(os.environ.get("GAMMA", 0.75))
CONFIG_LABEL = os.environ.get("CONFIG_LABEL", "e3_g75")

# Comma-separated list of methods to run (default: all 5).
METHOD_KEYS = [
    "internal", "cr_to_cr", "overall_to_cr", "binary1_to_cr", "binary2_to_cr",
]
METHODS = [m.strip() for m in os.environ.get(
    "METHODS", ",".join(METHOD_KEYS)).split(",") if m.strip() in METHOD_KEYS]
METHOD_LABEL = {
    "internal":       "Internal CR",
    "cr_to_cr":       "CR -> CR",
    "overall_to_cr":  "Overall -> CR",
    "binary1_to_cr":  "Binary-1 -> CR",
    "binary2_to_cr":  "Binary-2 -> CR",
}

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))


# ---------- Simulation ----------

def setup_simulation():
    """Use the updated original-repo design via `simulate_competing_risk_cohorts`.

    Three independent cohorts (teacher, student, test). Teacher gets the reduced
    feature subset by default; student and test get the full feature set.
    """
    cohorts = simulate_competing_risk_cohorts(
        n_teacher=TEACHER_N,
        n_student=STUDENT_N,
        n_test=TEST_N,
        seed=SEED,
        teacher_feature_quality=TEACHER_FEATURE_QUALITY,
    )
    combined_dur = np.concatenate([cohorts.teacher["duration"].values,
                                    cohorts.student["duration"].values])
    time_grid = fit_time_grid(combined_dur, NUM_DURATIONS)
    return {
        "teacher_data": cohorts.teacher,
        "train": cohorts.student,
        "test": cohorts.test,
        "teacher_features": cohorts.teacher_features,
        "student_features": cohorts.student_features,
        "time_grid": time_grid,
        "teacher_feature_quality": cohorts.teacher_feature_quality,
    }


def train_cr_teacher(sim):
    """Competing-risk teacher on the teacher cohort using the reduced feature subset."""
    return DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=sim["time_grid"],
    ).fit(sim["teacher_data"], feature_cols=sim["teacher_features"])


def train_overall_teacher(sim):
    """Single-risk (overall any-event) teacher on the teacher cohort.

    Matches the updated tutorial: teacher_train has `event_any = (event > 0)` and
    the teacher is a `DiscreteSurvivalModel(num_risks=1)` fit with `event_col="event_any"`.
    """
    td = sim["teacher_data"].copy()
    td["event_any"] = (td["event"] > 0).astype(int)
    return DiscreteSurvivalModel(
        num_risks=1, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=sim["time_grid"],
    ).fit(td, feature_cols=sim["teacher_features"], event_col="event_any")


def train_binary_teacher(sim, risk_label):
    """Logistic-regression binary teacher: P(event=risk_label by horizon_idx) trained on the teacher cohort.

    Uses only the teacher feature subset, but the returned predict() callable
    must accept dataframes with the full student feature set and select the
    reduced columns from them.
    """
    horizon_time = float(sim["time_grid"].cuts[HORIZON_INDEX])
    td = sim["teacher_data"]
    y = ((td["event"] == risk_label) & (td["duration"] <= horizon_time)).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    teacher_features = list(sim["teacher_features"])
    clf.fit(td[teacher_features], y)

    def predict(frame):
        return clf.predict_proba(frame[teacher_features])[:, 1]

    return predict


# ---------- Model builders ----------

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
        sgld_burnin_epochs=BURNIN_EPOCHS,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )
    return _model_from(method, teachers, common)


# ---------- Runners ----------

def run_adamw(method, teachers, sim, test_dur, test_ev, test_idx, test):
    ctd1, ctd2, dev = [], [], []
    cifs = []
    pretrained = None
    for r in range(N_ADAM_RESTARTS):
        torch.manual_seed(SEED + r)
        np.random.seed(SEED + r)
        m = build_adamw(method, teachers, sim["time_grid"])
        m.fit(sim["train"], feature_cols=sim["student_features"])
        cif_r = np.asarray(m.predict_cif(test))
        ctd_r = competing_risk_c_index(cif_r, test_dur, test_ev)
        dev_r = predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_ev)
        ctd1.append(float(ctd_r[0]))
        ctd2.append(float(ctd_r[1]))
        dev.append(float(dev_r))
        cifs.append(cif_r)
        if r == 0:
            pretrained = {k: v.clone() for k, v in m.net.state_dict().items()}
    return {
        "ctd1": np.array(ctd1),
        "ctd2": np.array(ctd2),
        "dev": np.array(dev),
        "cifs": np.stack(cifs, axis=0),
        "pretrained": pretrained,
    }


def run_sgld(method, teachers, sim, pretrained, test_dur, test_ev, test_idx, test):
    base = build_sgld(method, teachers, sim["time_grid"])
    sampler = WarmStartMultiChainSampler(
        base, pretrained, n_chains=N_CHAINS,
        seeds=list(range(100, 100 + N_CHAINS)),
    )
    sampler.fit(sim["train"], feature_cols=sim["student_features"])
    lead = sampler.lead_chain
    chains = []
    for chain in sampler.chains:
        cifs, ctd1s, ctd2s, devs = [], [], [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            cifs.append(cif)
            ctd = competing_risk_c_index(cif, test_dur, test_ev)
            ctd1s.append(float(ctd[0]))
            ctd2s.append(float(ctd[1]))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_ev))
        chains.append({
            "cifs": np.stack(cifs, axis=0),
            "ctd1": np.array(ctd1s),
            "ctd2": np.array(ctd2s),
            "dev": np.array(devs),
            "loss": chain.history.losses[-1],
        })
    return chains


def per_entry_metrics(cif_samples, true_cif):
    q = np.quantile(cif_samples, [0.025, 0.5, 0.975], axis=0)
    median = q[1]
    width = q[2] - q[0]
    abs_err = np.abs(median - true_cif)
    inside = (true_cif >= q[0]) & (true_cif <= q[2])
    return {"median": median, "width": width, "abs_err": abs_err,
            "inside": inside, "q025": q[0], "q975": q[2]}


# ---------- Figures ----------

def fmt(q):
    return f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"


def fig_metrics(results):
    """Boxplots of Ctd1, Ctd2, deviance for AdamW vs SGLD across all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    method_labels_short = [METHOD_LABEL[m].replace(" -> ", "\n->\n") for m in METHODS]
    titles = ["$C_{\\rm td}$ cause 1", "$C_{\\rm td}$ cause 2", "Deviance"]
    keys = ["ctd1", "ctd2", "dev"]
    for ax_idx, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[ax_idx]
        positions, values, colors, labels = [], [], [], []
        for mi, method in enumerate(METHODS):
            adam_vals = results[method]["adam"][key]
            bc = results[method]["chains"][results[method]["best_idx"]]
            sgld_vals = bc[key]
            positions.append(mi - 0.18)
            values.append(adam_vals)
            colors.append("#9ecae1")
            positions.append(mi + 0.18)
            values.append(sgld_vals)
            colors.append("#fd8d3c")
        bp = ax.boxplot(values, positions=positions, widths=0.3, patch_artist=True,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_xticklabels(method_labels_short, fontsize=8)
        ax.set_title(title)
        ax.grid(ls=":", alpha=0.4, axis="y")
        # Reference line: AdamW Internal CR median (baseline to read distillation lift)
        if "internal" in [m for m in METHODS]:
            internal_idx = METHODS.index("internal")
            ref = np.median(results["internal"]["adam"][key])
            ax.axhline(ref, color="gray", ls="--", lw=0.7,
                       label=f"Internal Adam median ({ref:.3f})")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"AdamW (light blue, 20 restarts) vs warm-start SGLD (orange, best chain) — "
                 f"{SGLD_EPOCHS}ep, {CONFIG_LABEL}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "uq_5methods_metrics.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_uq(results):
    """Coverage and width per method: Adam vs SGLD."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    method_labels_short = [METHOD_LABEL[m].replace(" -> ", "\n->\n") for m in METHODS]
    cov_a, cov_s, wid_a, wid_s = [], [], [], []
    for method in METHODS:
        pe_a = results[method]["per_entry_adam"]
        pe_s = results[method]["per_entry_sgld"]
        cov_a.append(pe_a["inside"].mean())
        cov_s.append(pe_s["inside"].mean())
        wid_a.append(pe_a["width"].mean())
        wid_s.append(pe_s["width"].mean())
    x = np.arange(len(METHODS))
    bw = 0.35

    ax = axes[0]
    ax.bar(x - bw / 2, cov_a, bw, color="#9ecae1", label="Adam (20 restarts)")
    ax.bar(x + bw / 2, cov_s, bw, color="#fd8d3c", label="SGLD (best chain)")
    ax.axhline(0.95, color="gray", ls="--", lw=0.7, label="nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels_short, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CIF coverage")
    ax.set_title("CIF coverage per method")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.4, axis="y")

    ax = axes[1]
    ax.bar(x - bw / 2, wid_a, bw, color="#9ecae1", label="Adam (20 restarts)")
    ax.bar(x + bw / 2, wid_s, bw, color="#fd8d3c", label="SGLD (best chain)")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels_short, fontsize=8)
    ax.set_ylabel("Mean CI width  $Q_{0.975} - Q_{0.025}$")
    ax.set_title("CIF interval width per method")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.4, axis="y")

    fig.suptitle("Uncertainty quantification: Adam restart spread vs warm-start SGLD posterior",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "uq_5methods_uq.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_width_vs_error(results, correlations):
    """5x2 grid: rows = methods, cols = cause 1/2."""
    fig, axes = plt.subplots(len(METHODS), 2, figsize=(10, 2.4 * len(METHODS)))
    if len(METHODS) == 1:
        axes = axes[np.newaxis, :]
    for mi, method in enumerate(METHODS):
        pe = results[method]["per_entry_sgld"]
        for j in range(2):
            ax = axes[mi, j]
            w = pe["width"][:, j, :].flatten()
            e = pe["abs_err"][:, j, :].flatten()
            inside = pe["inside"][:, j, :].flatten()
            ax.scatter(w[inside], e[inside], c="#2ca02c", s=4, alpha=0.3)
            ax.scatter(w[~inside], e[~inside], c="#d62728", s=4, alpha=0.3)
            n_bins = 10
            if len(w) > n_bins:
                edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
                bin_idx = np.clip(np.digitize(w, edges[1:-1]), 0, n_bins - 1)
                bw = np.array([np.mean(w[bin_idx == b]) for b in range(n_bins)])
                be = np.array([np.median(e[bin_idx == b]) for b in range(n_bins)])
                ax.plot(bw, be, color="black", lw=1.2)
            rho = correlations[method][f"cause{j+1}"]
            ax.set_title(f"{METHOD_LABEL[method]}, cause {j+1}, $\\rho$={rho:.3f}",
                         fontsize=9)
            ax.grid(ls=":", alpha=0.4)
            if mi == len(METHODS) - 1:
                ax.set_xlabel("CI width")
            if j == 0:
                ax.set_ylabel("$|$err$|$")
    fig.suptitle("Width vs. prediction error across methods (warm-start SGLD posterior)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "uq_5methods_width_vs_error.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------- Report ----------

def write_report(results, correlations, teacher_summaries):
    md = []
    md.append("# UQ Comparison — All 5 Distillation Schemes")
    md.append("")
    md.append("**Audience:** Jian, Kevin.")
    md.append("**Scope:** characterise uncertainty quantification (UQ) and distillation benefit of all 5 DiSKD schemes under the original-paper simulation design. Locked SGLD config: `{}` (ε₀={:.0e} → ε_T={:.0e}, γ={}), 500 epochs warm-start from AdamW, no burn-in.".format(CONFIG_LABEL, EPS0, EPS_T, GAMMA))
    md.append("")
    md.append("## 1. Simulation (updated original-repo helper)")
    md.append("")
    md.append("Generated via `simulate_competing_risk_cohorts(...)` from the updated `original/DiscreteSurvKD-main`.")
    md.append("")
    md.append("- **Teacher cohort:** N = {} subjects, feature subset = `{}`, teacher_feature_quality = `{}`.".format(TEACHER_N, ", ".join(results[METHODS[0]].get("teacher_features_repr", ["x1..x8"])) if False else "x1..x8", TEACHER_FEATURE_QUALITY))
    md.append("- **Student cohort:** N = {} subjects (disjoint), full feature set `x1..x12`.".format(STUDENT_N))
    md.append("- **Test cohort:** N = {} subjects (independent of student; not a split).".format(TEST_N))
    md.append("- The teacher misses the shared-signal block `x9..x12`. The expected role of distillation here is **regularisation**: even a feature-incomplete teacher can supply prior structure that stabilises a small-N student.")
    md.append("")
    md.append("## 2. Methods")
    md.append("")
    md.append("| Method | Description | Teacher | KD loss |")
    md.append("|---|---|---|---|")
    md.append("| Internal CR     | No teacher (baseline) | — | — |")
    md.append("| CR → CR         | DiSKD-C, at-risk KL | CR teacher (reduced features) | competing |")
    md.append("| Overall → CR    | DiSKD-O, overall-hazard BCE | single-risk teacher on `event_any` | overall |")
    md.append("| Binary-1 → CR   | Fixed-horizon binary CIF BCE on cause 1 | Logistic teacher at t_k | binary_horizon |")
    md.append("| Binary-2 → CR   | Fixed-horizon binary CIF BCE on cause 2 | Logistic teacher at t_k | binary_horizon |")
    md.append("")
    md.append(f"Horizon index k = {HORIZON_INDEX} (time t_k = {teacher_summaries['horizon_time']:.4f}).")
    md.append("")
    md.append("## 3. Teacher quality")
    md.append("")
    md.append(f"CR teacher final loss on teacher cohort = {teacher_summaries['cr_loss']:.4f}.")
    md.append(f"Overall teacher (single-risk on `event_any`) final loss = {teacher_summaries['overall_loss']:.4f}.")
    md.append(f"Binary-1 teacher AUC ≈ {teacher_summaries['bin1_auc']:.3f} on teacher cohort.")
    md.append(f"Binary-2 teacher AUC ≈ {teacher_summaries['bin2_auc']:.3f} on teacher cohort.")
    md.append("")
    md.append("## 4. Headline metrics — distillation benefit check")
    md.append("")
    md.append("Best SGLD chain selected by lowest median deviance. Entries: median [2.5%, 97.5%].")
    md.append("")
    md.append("### AdamW (20 restarts):")
    md.append("")
    md.append("| Method | C_td cause 1 | C_td cause 2 | Deviance | CIF cov | CIF wid |")
    md.append("|---|---|---|---|---|---|")
    for method in METHODS:
        adam = results[method]["adam"]
        pe_a = results[method]["per_entry_adam"]
        md.append("| {} | {} | {} | {} | {:.3f} | {:.3f} |".format(
            METHOD_LABEL[method],
            fmt(np.quantile(adam["ctd1"], [0.025, 0.5, 0.975])),
            fmt(np.quantile(adam["ctd2"], [0.025, 0.5, 0.975])),
            fmt(np.quantile(adam["dev"], [0.025, 0.5, 0.975])),
            pe_a["inside"].mean(),
            pe_a["width"].mean(),
        ))
    md.append("")
    md.append("### Warm-start SGLD (best chain):")
    md.append("")
    md.append("| Method | C_td cause 1 | C_td cause 2 | Deviance | CIF cov | CIF wid |")
    md.append("|---|---|---|---|---|---|")
    for method in METHODS:
        bc = results[method]["chains"][results[method]["best_idx"]]
        pe_s = results[method]["per_entry_sgld"]
        md.append("| {} | {} | {} | {} | {:.3f} | {:.3f} |".format(
            METHOD_LABEL[method],
            fmt(np.quantile(bc["ctd1"], [0.025, 0.5, 0.975])),
            fmt(np.quantile(bc["ctd2"], [0.025, 0.5, 0.975])),
            fmt(np.quantile(bc["dev"], [0.025, 0.5, 0.975])),
            pe_s["inside"].mean(),
            pe_s["width"].mean(),
        ))
    md.append("")
    md.append("## 5. Spearman ρ(width, |error|) per method")
    md.append("")
    md.append("Positive ρ ⇒ posterior is informative as an uncertainty measure (wider intervals concentrate where prediction error is larger).")
    md.append("")
    md.append("| Method | ρ overall | ρ cause 1 | ρ cause 2 |")
    md.append("|---|---|---|---|")
    for method in METHODS:
        c = correlations[method]
        md.append(f"| {METHOD_LABEL[method]} | {c['overall']:.3f} | {c['cause1']:.3f} | {c['cause2']:.3f} |")
    md.append("")
    md.append("## 6. Figures")
    md.append("")
    md.append("### 6.1 Point-estimate metrics per method")
    md.append("![Method metrics](uq_5methods_metrics.png)")
    md.append("")
    md.append("Dashed gray reference line = Internal CR (no teacher) AdamW median; distillation methods that beat this line show a real distillation benefit.")
    md.append("")
    md.append("### 6.2 Uncertainty quantification per method")
    md.append("![UQ per method](uq_5methods_uq.png)")
    md.append("")
    md.append("Adam (light blue) shows restart spread; SGLD (orange) shows within-chain posterior. The two are evaluated using the same 2.5/97.5 quantile-interval definition.")
    md.append("")
    md.append("### 6.3 Width vs prediction error per method")
    md.append("![Width vs error](uq_5methods_width_vs_error.png)")
    md.append("")
    md.append("Each row = one method; columns = cause 1 and cause 2. Green dots = inside 95% CI, red = outside. Black line = bin-median |error|.")
    md.append("")
    md.append("## 7. Files")
    md.append("")
    md.append("- `uq_5methods_data.npz` — per-entry width/error/inside arrays for all methods")
    md.append("- `uq_5methods_metrics.{pdf,png}`")
    md.append("- `uq_5methods_uq.{pdf,png}`")
    md.append("- `uq_5methods_width_vs_error.{pdf,png}`")
    md.append("")
    md.append("## 8. Interpretation notes")
    md.append("")
    md.append("- **Distillation benefit:** with the original-paper design (teacher full features, student reduced features), distillation should lift C_td relative to Internal CR. Compare the Ctd boxplots vs the gray reference line.")
    md.append("- **UQ quality:** what matters most is (a) the Spearman ρ between credible width and prediction error (informativeness) and (b) the empirical CIF coverage at the stated 95% level (calibration). Both should be higher for a well-tuned posterior.")
    md.append("- **Adam vs SGLD UQ:** Adam's 20-restart spread captures init noise only; SGLD's within-chain spread targets the actual posterior over θ. For a fair UQ comparison the SGLD interval should be at least as wide as Adam's at matched coverage; the question is whether SGLD makes that trade-off in a way that supports calibrated downstream decisions.")

    out = OUT_DIR / "uq_5methods_report.md"
    out.write_text("\n".join(md) + "\n")
    return out


# ---------- Main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    print(f"=== UQ comparison: 5 methods, updated simulate_competing_risk_cohorts ===")
    print(f"Methods to run: {METHODS}")
    print(f"SGLD: {CONFIG_LABEL}, ε₀={EPS0:.0e} → ε_T={EPS_T:.0e}, γ={GAMMA}, "
          f"{SGLD_EPOCHS} ep ({BURNIN_EPOCHS} burnin), "
          f"{N_CHAINS} chains × {SAMPLES_PER_CHAIN} draws")

    sim = setup_simulation()
    test = sim["test"]
    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, sim["time_grid"])
    true_cif = true_cif_at_grid(test, sim["time_grid"], num_risks=NUM_RISKS)
    horizon_time = float(sim["time_grid"].cuts[HORIZON_INDEX])

    print(f"\nTeacher: N={TEACHER_N}, features={len(sim['teacher_features'])} "
          f"({sim['teacher_features'][0]}..{sim['teacher_features'][-1]}), "
          f"quality={sim['teacher_feature_quality']}")
    print(f"Student: train={len(sim['train'])}, "
          f"features={len(sim['student_features'])} "
          f"({sim['student_features'][0]}..{sim['student_features'][-1]})")
    print(f"Test:    N={len(test)} (independent cohort)")
    print(f"Horizon index k = {HORIZON_INDEX} (t_k = {horizon_time:.4f})")

    # ---- Train teachers ----
    teachers = {}
    teacher_summaries = {"horizon_time": horizon_time}
    if "cr_to_cr" in METHODS:
        t0 = time.time()
        teachers["cr"] = train_cr_teacher(sim)
        cr_loss = teachers["cr"].history.losses[-1]
        teacher_summaries["cr_loss"] = cr_loss
        print(f"\nCR teacher trained: loss={cr_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if "overall_to_cr" in METHODS:
        t0 = time.time()
        teachers["overall"] = train_overall_teacher(sim)
        ov_loss = teachers["overall"].history.losses[-1]
        teacher_summaries["overall_loss"] = ov_loss
        print(f"Overall teacher trained: loss={ov_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if "binary1_to_cr" in METHODS:
        t0 = time.time()
        teachers["bin1"] = train_binary_teacher(sim, 1)
        # AUC on teacher cohort as a sanity check
        from sklearn.metrics import roc_auc_score
        td = sim["teacher_data"]
        y = ((td["event"] == 1) & (td["duration"] <= horizon_time)).astype(int)
        teacher_summaries["bin1_auc"] = float(roc_auc_score(y, teachers["bin1"](td)))
        print(f"Binary-1 teacher trained: AUC={teacher_summaries['bin1_auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if "binary2_to_cr" in METHODS:
        t0 = time.time()
        teachers["bin2"] = train_binary_teacher(sim, 2)
        from sklearn.metrics import roc_auc_score
        td = sim["teacher_data"]
        y = ((td["event"] == 2) & (td["duration"] <= horizon_time)).astype(int)
        teacher_summaries["bin2_auc"] = float(roc_auc_score(y, teachers["bin2"](td)))
        print(f"Binary-2 teacher trained: AUC={teacher_summaries['bin2_auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    teacher_summaries.setdefault("cr_loss", float("nan"))
    teacher_summaries.setdefault("overall_loss", float("nan"))
    teacher_summaries.setdefault("bin1_auc", float("nan"))
    teacher_summaries.setdefault("bin2_auc", float("nan"))

    # ---- Run each method ----
    results = {}
    correlations = {}
    for method in METHODS:
        print(f"\n--- {METHOD_LABEL[method]} ---", flush=True)
        t0 = time.time()
        adam = run_adamw(method, teachers, sim, test_dur, test_ev, test_idx, test)
        print(f"AdamW {N_ADAM_RESTARTS} restarts ({time.time()-t0:.0f}s): "
              f"Ctd1={np.median(adam['ctd1']):.3f} "
              f"Ctd2={np.median(adam['ctd2']):.3f} "
              f"dev={np.median(adam['dev']):.3f}", flush=True)

        t0 = time.time()
        chains = run_sgld(method, teachers, sim, adam["pretrained"],
                          test_dur, test_ev, test_idx, test)
        bc_idx = int(np.argmin([np.median(c["dev"]) for c in chains]))
        bc = chains[bc_idx]
        print(f"SGLD {N_CHAINS}×{SAMPLES_PER_CHAIN} ({time.time()-t0:.0f}s): "
              f"best chain {bc_idx}, "
              f"Ctd1={np.median(bc['ctd1']):.3f} "
              f"Ctd2={np.median(bc['ctd2']):.3f} "
              f"dev={np.median(bc['dev']):.3f}", flush=True)

        pe_sgld = per_entry_metrics(bc["cifs"], true_cif)
        pe_adam = per_entry_metrics(adam["cifs"], true_cif)
        results[method] = {
            "adam": adam, "chains": chains, "best_idx": bc_idx,
            "per_entry_sgld": pe_sgld, "per_entry_adam": pe_adam,
        }
        rho_overall, _ = spearmanr(pe_sgld["width"].flatten(), pe_sgld["abs_err"].flatten())
        rho_c1, _ = spearmanr(pe_sgld["width"][:, 0, :].flatten(),
                              pe_sgld["abs_err"][:, 0, :].flatten())
        rho_c2, _ = spearmanr(pe_sgld["width"][:, 1, :].flatten(),
                              pe_sgld["abs_err"][:, 1, :].flatten())
        correlations[method] = {
            "overall": float(rho_overall), "cause1": float(rho_c1), "cause2": float(rho_c2),
        }
        print(f"  Spearman ρ(width, |err|) overall={rho_overall:.3f} "
              f"cause1={rho_c1:.3f} cause2={rho_c2:.3f}", flush=True)

    # ---- Save raw data ----
    save = {"true_cif": true_cif}
    for method in METHODS:
        pe_s = results[method]["per_entry_sgld"]
        pe_a = results[method]["per_entry_adam"]
        for k in ["median", "width", "abs_err", "q025", "q975"]:
            save[f"{method}_sgld_{k}"] = pe_s[k]
            save[f"{method}_adam_{k}"] = pe_a[k]
        save[f"{method}_sgld_inside"] = pe_s["inside"].astype(np.int8)
        save[f"{method}_adam_inside"] = pe_a["inside"].astype(np.int8)
        save[f"{method}_adam_ctd1"] = results[method]["adam"]["ctd1"]
        save[f"{method}_adam_ctd2"] = results[method]["adam"]["ctd2"]
        save[f"{method}_adam_dev"] = results[method]["adam"]["dev"]
    np.savez(OUT_DIR / "uq_5methods_data.npz", **save)
    print(f"\nData saved to {OUT_DIR / 'uq_5methods_data.npz'}", flush=True)

    # ---- Figures ----
    fig_metrics(results)
    fig_uq(results)
    fig_width_vs_error(results, correlations)
    print("Figures saved.", flush=True)

    # ---- Report ----
    out = write_report(results, correlations, teacher_summaries)
    print(f"Report saved to {out}", flush=True)


if __name__ == "__main__":
    main()
