#!/usr/bin/env python
"""Eta-selection probe for Bayesian DiSKD.

Settings (per user decision, 2026-08-10):
  * ORIGINAL DiSKD DGP (squared rates), identical to writeup Section 1.3.
  * Fixed-epsilon SGLD, warm-started from the AdamW MAP (the eta-selection
    phase does NOT chase R-hat convergence; it conditions on the warm-start
    posterior, so a fixed step + shared MAP start is the intended setting).

Implements the fixed-eta selection criteria, tried one by one:
  METHOD 1  CV-A  : original-DiSKD-style 5-fold CV (point estimate). For each
                    eta, train the student on 4 folds and score held-out
                    predictive deviance; eta_hat = argmin summed deviance.
                    ---> IMPLEMENTED HERE.
  METHOD 2  LPML/CPO, METHOD 3 DIC : reuse one warm-start fixed-eps posterior
                    per eta.  ---> scaffolded (SELECT=lpml_dic), added next.
  METHOD 4  GBIC, random-eta path sampling : later tiers.

Env knobs: DGP=squared D=12 CENSOR_MAX=0.05 BETA_R1/2/SHARED, NUM_DURATIONS=20,
TEACHER_N=5000 STUDENT_N=500 TEST_N=1000, ETAS grid, CV_FOLDS=5, SEEDS,
SELECT={cv|lpml_dic|all}. Sampler knobs mirror converged_probe for the
Bayesian tiers (EPS fixed, warm start).
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diskd import DiscreteSurvivalModel, DiSKDStudent
from diskd.metrics import predictive_deviance, competing_risk_c_index
from diskd.preprocessing import transform_durations

# ----------------------------- config ---------------------------------------
DEVICE = os.environ.get("DEVICE", "cpu")
D = int(os.environ.get("D", 12)); J = 2
K = int(os.environ.get("NUM_DURATIONS", 20))
TEACHER_N = int(os.environ.get("TEACHER_N", 5000)); STUDENT_N = int(os.environ.get("STUDENT_N", 500))
TEST_N = int(os.environ.get("TEST_N", 1000)); HID = int(os.environ.get("HIDDEN", 32)); BATCH = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100)); ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 60))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 2.0))
CV_FOLDS = int(os.environ.get("CV_FOLDS", 5))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.1,0.25,0.5,1,2,4").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
SELECT = os.environ.get("SELECT", "cv")  # cv | lpml_dic | all

# --- ORIGINAL DiSKD squared-rate DGP (simulation.py:45) ---
CENSOR_MAX = float(os.environ.get("CENSOR_MAX", 0.05))
BETA_R1 = float(os.environ.get("BETA_R1", 2.0)); BETA_R2 = float(os.environ.get("BETA_R2", 2.0))
BETA_SHARED = float(os.environ.get("BETA_SHARED", 8.0))

_DEFAULT = Path(__file__).resolve().parent.parent / "responses_eta_selection"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def rates(x):
    """Original DiSKD squared rates; covariate groups fixed as in simulation.py."""
    z1 = x[:, 0:4].sum(1); z2 = x[:, 4:8].sum(1); z3 = x[:, 8:12].sum(1)
    r1 = np.clip((BETA_R1 * z1) ** 2 + (BETA_SHARED * z3) ** 2, 1e-3, None)
    r2 = np.clip((BETA_R2 * z2) ** 2 + (BETA_SHARED * z3) ** 2, 1e-3, None)
    return np.stack([r1, r2], axis=1)


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D)); r = rates(x)
    t = rng.exponential(scale=1.0 / r); cause = t.argmin(1); et = t.min(1)
    cens = rng.uniform(0.0, CENSOR_MAX, size=n)  # original DiSKD uniform censoring
    censored = cens < et
    dur = np.where(censored, cens, et); event = np.where(censored, 0, cause + 1).astype(np.int64)
    df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(D)])
    df["duration"] = dur.astype(float); df["event"] = event
    return df


def train_teacher(df, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=HID, epochs=TEACHER_EPOCHS,
                              batch_size=BATCH, device=DEVICE, optimizer="adamw")
    m.fit(df)
    return m


def train_student(df, teacher, eta, seed, valid=None):
    """Point-estimate (AdamW) student at a fixed eta, sharing the teacher grid."""
    torch.manual_seed(seed); np.random.seed(seed)
    common = dict(num_risks=J, num_durations=K, hidden_dim=HID, epochs=ADAMW_EPOCHS, batch_size=BATCH,
                  device=DEVICE, optimizer="adamw", time_grid=teacher.time_grid)
    if eta == 0:
        m = DiscreteSurvivalModel(**common)
    else:
        m = DiSKDStudent(teacher_model=teacher, teacher_type="competing", eta=eta,
                         temperature=TEMPERATURE, **common)
    m.fit(df, valid_data=valid)
    return m


def heldout_deviance(model, holdout, tg):
    """Summed held-out predictive deviance = -2 * sum_i log L_i (student likelihood)."""
    probs = model.predict_interval_probs(holdout).detach().cpu().numpy()  # [N, J+1, K]
    idx = transform_durations(holdout["duration"].to_numpy(), tg)
    ev = holdout["event"].to_numpy(np.int64)
    return predictive_deviance(probs, idx, ev, reduction="sum")


def cv_a(student_df, teacher, seed):
    """METHOD 1 (CV-A): original-DiSKD-style 5-fold CV over the eta grid.

    Returns {eta: summed held-out deviance over folds} plus the full refit test scores.
    """
    n = len(student_df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, CV_FOLDS)
    cv_dev = {e: 0.0 for e in ETAS}
    for e in ETAS:
        for v in range(CV_FOLDS):
            va = folds[v]; tr = np.concatenate([folds[j] for j in range(CV_FOLDS) if j != v])
            tr_df = student_df.iloc[tr].reset_index(drop=True)
            va_df = student_df.iloc[va].reset_index(drop=True)
            m = train_student(tr_df, teacher, e, seed + v)
            cv_dev[e] += heldout_deviance(m, va_df, teacher.time_grid)
    return cv_dev


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    print(f"=== Eta-selection probe (ORIGINAL DiSKD DGP, K={K}) === SELECT={SELECT} "
          f"etas={ETAS} folds={CV_FOLDS} seeds={SEEDS} teacher/student/test={TEACHER_N}/{STUDENT_N}/{TEST_N}",
          flush=True)
    rows = []
    for seed in SEEDS:
        teacher_df = simulate(TEACHER_N, seed)
        student_df = simulate(STUDENT_N, seed + 1000)
        test_df = simulate(TEST_N, seed + 2000)
        teacher = train_teacher(teacher_df, seed)
        tg = teacher.time_grid
        idx_te = transform_durations(test_df["duration"].to_numpy(), tg)
        ev_te = test_df["event"].to_numpy(np.int64)

        cv_dev = cv_a(student_df, teacher, seed) if SELECT in ("cv", "all") else {}

        # Full refit at each eta on all student data + test-set sensitivity scores.
        for e in ETAS:
            m = train_student(student_df, teacher, e, seed)
            probs = m.predict_interval_probs(test_df).detach().cpu().numpy()
            test_dev = predictive_deviance(probs, idx_te, ev_te, reduction="mean")
            cif = m.predict_cif(test_df)  # [N, J, K]
            cidx = competing_risk_c_index(cif, idx_te, ev_te)
            row = dict(seed=seed, eta=e, cv_dev=cv_dev.get(e, np.nan),
                       test_dev=test_dev, cidx1=float(cidx[0]), cidx2=float(cidx[1]))
            rows.append(row)
            print(f"  seed{seed} eta={e:<4g} | CV dev(sum)={row['cv_dev']:.1f} | "
                  f"test dev={test_dev:.4f} | C-idx=({cidx[0]:.3f},{cidx[1]:.3f})", flush=True)
        if cv_dev:
            eta_hat = min(cv_dev, key=cv_dev.get)
            print(f"  seed{seed}: CV-A eta_hat = {eta_hat:g}  (min summed held-out deviance)", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "eta_selection.csv", index=False)

    # -------- markdown summary --------
    md = ["# Eta selection -- Method 1 (CV-A), original DiSKD DGP", ""]
    md.append(f"Squared-rate DGP (beta={BETA_R1}/{BETA_R2}/{BETA_SHARED}), K={K}, uniform censoring "
              f"[0,{CENSOR_MAX}]. Teacher/student/test = {TEACHER_N}/{STUDENT_N}/{TEST_N}. "
              f"{CV_FOLDS}-fold CV, temperature={TEMPERATURE}. Seeds {SEEDS}.")
    md.append("")
    md.append("CV-A selects eta = argmin (summed held-out student predictive deviance); "
              "lower is better. test dev / C-index are the sensitivity scores at each eta.")
    md.append("")
    md.append("| eta | CV dev (mean over seeds) | test dev | C-idx cause1 | C-idx cause2 |")
    md.append("|---:|---:|---:|---:|---:|")
    for e in ETAS:
        s = df[df.eta == e]
        md.append(f"| {e:g} | {s['cv_dev'].mean():.1f} | {s['test_dev'].mean():.4f} | "
                  f"{s['cidx1'].mean():.3f} | {s['cidx2'].mean():.3f} |")
    md.append("")
    if SELECT in ("cv", "all"):
        # per-seed eta_hat, then modal / mean
        hats = []
        for seed in SEEDS:
            s = df[(df.seed == seed)].dropna(subset=["cv_dev"])
            if len(s):
                hats.append(float(s.loc[s["cv_dev"].idxmin(), "eta"]))
        if hats:
            md.append(f"**CV-A eta_hat per seed:** {hats}  (mean {np.mean(hats):.3f}).")
            md.append("")
    md.append("Per-seed rows: `eta_selection.csv`. Report sensitivity across neighboring eta "
              "(power-prior guidance: the selected value is a guide, not a deterministic choice).")
    (OUT_DIR / "eta_selection.md").write_text("\n".join(md))
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
