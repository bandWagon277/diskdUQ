"""Last-layer Godambe / sandwich variance correction for Bayesian DiSKD.

Implements Experiments 1 and 2 of the variance-correction plan.

Motivation. The DiSKD objective is a generalized (not true-likelihood) loss

    m_i(theta) = r_i(theta) + eta * q_i(theta),

with r_i the internal discrete-time competing-risk NLL and q_i the teacher KL.
Crucially, q_i is evaluated on the *same local subjects and at-risk intervals* as
r_i, so the two scores are correlated and the correct repeated-sampling target is
the Godambe (sandwich) covariance

    V^G_eta = (1/n) H_eta^{-1} [ J_R + eta (J_RQ + J_QR) + eta^2 J_Q ] H_eta^{-1},

not the naive inverse curvature (1/(n*omega)) H_eta^{-1} that the raw generalized
posterior implies. Undercoverage of the raw posterior is consistent with
J_eta > omega^{-1} H_eta.

Scope. We correct the **last layer** (`head`, a Linear(hidden_dim, num_risks)),
freezing the backbone. For hidden=32, J=2 that is only 66 parameters, so H, J and
the sandwich are computed *exactly* (no diagonal/low-rank approximation), and the
CIF Jacobian is obtained by forward-mode autodiff (66 JVPs).

Outputs (OUT_DIR, default responses_sandwich/):
  sandwich_correction_report.md
  sandwich_correction.csv
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import jacfwd, hessian

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    competing_risk_c_index,
    fit_time_grid,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import true_cif_at_grid
from diskd.losses import CompetingRiskNLLLoss
from diskd.networks import sinusoidal_time_embedding
from diskd.utils import (
    competing_cif,
    competing_interval_probs,
    full_probs_from_event_probs,
    make_at_risk_mask,
    temperature_scale_probs,
)

# ---------- config ----------
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
TEST_N = int(os.environ.get("TEST_N", 500))
QUALITY = os.environ.get("TEACHER_FEATURE_QUALITY", "full")
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = int(os.environ.get("STUDENT_HIDDEN", 32))
BATCH_SIZE = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 2.0))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.25,0.5,1,2,4").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",")]
Z = 1.959963985  # 95% normal quantile
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_sandwich"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


# ---------- setup ----------

def setup(seed):
    co = simulate_competing_risk_cohorts(
        n_teacher=TEACHER_N, n_student=STUDENT_N, n_test=TEST_N,
        seed=seed, teacher_feature_quality=QUALITY)
    dur = np.concatenate([co.teacher["duration"].values, co.student["duration"].values])
    tg = fit_time_grid(dur, NUM_DURATIONS)
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=tg,
    ).fit(co.teacher, feature_cols=co.teacher_features)
    return co, tg, teacher


def fit_student(co, tg, teacher, eta, seed):
    torch.manual_seed(1000 * seed); np.random.seed(1000 * seed)
    m = DiSKDStudent(
        teacher_model=teacher, teacher_type="competing", eta=eta, temperature=TEMPERATURE,
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=tg, optimizer="adamw",
    )
    m.fit(co.student, feature_cols=co.student_features)
    return m


# ---------- last-layer plumbing ----------

def backbone_features(net, x):
    """Input to `head`: [N, K, hidden] (dropout disabled)."""
    net.eval()
    with torch.no_grad():
        h = net.feature_projection(x).unsqueeze(1)
        emb = sinusoidal_time_embedding(net.num_durations, h.shape[-1],
                                        device=x.device, dtype=x.dtype)
        h = h + emb.unsqueeze(0)
        h = net.blocks(h)
    return h


def flat_theta(net):
    return torch.cat([net.head.weight.detach().reshape(-1),
                      net.head.bias.detach().reshape(-1)]).double()


def logits_of(theta, feat, J, Hd):
    W = theta[: J * Hd].view(J, Hd)
    b = theta[J * Hd:]
    return torch.einsum("nkd,jd->njk", feat, W) + b.view(1, J, 1)


def make_loss_fns(feat, idx, events, teacher_full, J, Hd, K):
    """Return per-subject r_i(theta) and q_i(theta) as pure functions of theta."""
    nll = CompetingRiskNLLLoss()
    mask = make_at_risk_mask(idx, K).to(feat.dtype)
    t_t = temperature_scale_probs(teacher_full, TEMPERATURE)
    log_t = t_t.clamp_min(1e-12).log()

    def r_vec(theta):
        lg = logits_of(theta, feat, J, Hd)
        return nll(lg, idx, events, reduction="none")

    def q_vec(theta):
        lg = logits_of(theta, feat, J, Hd)
        s_t = temperature_scale_probs(competing_interval_probs(lg), TEMPERATURE)
        kl = t_t * (log_t - s_t.clamp_min(1e-12).log())
        kl_i = (kl.sum(dim=1) * mask).sum(dim=1)
        return (TEMPERATURE ** 2) * kl_i

    return r_vec, q_vec


def cif_flat_fn(feat, J, Hd):
    def f(theta):
        lg = logits_of(theta, feat, J, Hd)
        return competing_cif(competing_interval_probs(lg)).reshape(-1)
    return f


RCOND = float(os.environ.get("RCOND", 1e-6))


def _psd_pinv(M, rcond=RCOND):
    """Truncated PSD pseudo-inverse.

    The last-layer curvature is rank-deficient: the backbone features span a
    subspace of dimension < hidden (a 12-covariate projection through ReLU), so
    directions of W orthogonal to that span have exactly zero curvature. Those
    same directions leave the logits -- and hence the CIF -- unchanged, so the
    delta-method variance has no contribution there. Dropping them is therefore
    correct, and necessary: a naive inverse blows up in the null space.
    Returns (pinv, numerical_rank).
    """
    evals, evecs = torch.linalg.eigh(M)
    top = evals.max().clamp_min(0.0)
    thresh = rcond * top
    keep = evals > thresh
    inv_evals = torch.where(keep, 1.0 / evals.clamp_min(thresh), torch.zeros_like(evals))
    return (evecs * inv_evals) @ evecs.T, int(keep.sum())


def refine_last_layer(theta0, r_vec, q_vec, eta, steps=200):
    """Re-optimize the last layer to stationarity given the frozen backbone.

    The sandwich is an M-estimator result and assumes theta solves the estimating
    equation (mean score = 0). AdamW on the *full* network leaves a small residual
    last-layer gradient, so we polish the p<=66 last-layer parameters with LBFGS.
    Returns (theta_hat, grad_norm_before, grad_norm_after).
    """
    def obj(t):
        return (r_vec(t) + eta * q_vec(t)).mean()

    th = theta0.clone().requires_grad_(True)
    g0 = torch.autograd.grad(obj(th), th)[0].abs().max().item()
    opt = torch.optim.LBFGS([th], max_iter=steps, tolerance_grad=1e-12,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = obj(th)
        loss.backward()
        return loss

    opt.step(closure)
    th = th.detach()
    th_req = th.clone().requires_grad_(True)
    g1 = torch.autograd.grad(obj(th_req), th_req)[0].abs().max().item()
    return th, g0, g1


# ---------- core: sandwich at one (seed, eta) ----------

def sandwich_at(co, tg, teacher, eta, seed, true_cif):
    student = fit_student(co, tg, teacher, eta, seed)
    net = student.net
    J, Hd, K = NUM_RISKS, STUDENT_HIDDEN, NUM_DURATIONS

    # --- training-side tensors (the objective that was actually minimized) ---
    x_tr = student.preprocessor.transform(co.student)
    x_tr = torch.as_tensor(x_tr, dtype=torch.float32, device=DEVICE)
    idx_tr = torch.as_tensor(transform_durations(co.student["duration"].values, tg),
                             dtype=torch.long, device=DEVICE)
    ev_tr = torch.as_tensor(co.student["event"].values, dtype=torch.long, device=DEVICE)
    tprobs = teacher.predict_interval_probs(co.student)[:, :NUM_RISKS, :]
    teacher_full = full_probs_from_event_probs(
        torch.as_tensor(tprobs, dtype=torch.float64, device=DEVICE))

    feat_tr = backbone_features(net, x_tr).double()
    theta = flat_theta(net).to(DEVICE)
    n = feat_tr.shape[0]

    r_vec, q_vec = make_loss_fns(feat_tr, idx_tr, ev_tr, teacher_full, J, Hd, K)

    # The sandwich assumes theta solves the estimating equation. AdamW on the full
    # network leaves a small residual last-layer gradient, so polish it first.
    theta, grad0, grad1 = refine_last_layer(theta, r_vec, q_vec, eta)

    # --- per-subject scores (forward-mode: only p=66 JVPs) ---
    S_R = jacfwd(r_vec)(theta)                      # [n, p]
    S_Q = jacfwd(q_vec)(theta)                      # [n, p]

    def centered(S):
        return S - S.mean(0, keepdim=True)

    cR, cQ = centered(S_R), centered(S_Q)
    J_R = cR.T @ cR / n
    J_Q = cQ.T @ cQ / n
    J_RQ = cR.T @ cQ / n
    J_eta = J_R + eta * (J_RQ + J_RQ.T) + (eta ** 2) * J_Q

    # --- curvature of the mean combined loss ---
    def m_mean(th):
        return (r_vec(th) + eta * q_vec(th)).mean()

    H_eta = hessian(m_mean)(theta)                   # [p, p]
    H_eta = 0.5 * (H_eta + H_eta.T)
    H_inv, rank = _psd_pinv(H_eta)

    V_naive = H_inv / n                              # raw generalized posterior, omega=1
    V_sand = H_inv @ J_eta @ H_inv / n               # Godambe
    tr_naive = float(torch.diagonal(H_inv).sum())
    tr_sand = float(torch.diagonal(H_inv @ J_eta @ H_inv).sum())
    omega_star = tr_naive / tr_sand if tr_sand > 0 else float("nan")
    V_omega = H_inv / (n * omega_star) if np.isfinite(omega_star) else V_naive
    # information-equality diagnostic restricted to the identified subspace
    HiJ = H_inv @ J_eta
    ratio_JH = float(torch.diagonal(HiJ).sum()) / max(rank, 1)

    # --- function-space delta method on the TEST set ---
    x_te = torch.as_tensor(student.preprocessor.transform(co.test),
                           dtype=torch.float32, device=DEVICE)
    feat_te = backbone_features(net, x_te).double()
    f_cif = cif_flat_fn(feat_te, J, Hd)
    cif_point = f_cif(theta).reshape(TEST_N, J, K)
    D = jacfwd(f_cif)(theta)                         # [N*J*K, p]

    truth = torch.as_tensor(true_cif, dtype=torch.float64, device=DEVICE)
    # Is undercoverage a variance problem or a bias/misspecification problem?
    # Compare the actual prediction error against the model-implied standard error.
    abs_err = (cif_point - truth).abs()
    rmse = float((abs_err ** 2).mean().sqrt())
    mae = float(abs_err.mean())
    out = {}
    se_mean = {}
    for name, V in (("naive", V_naive), ("sandwich", V_sand), ("omega_matched", V_omega)):
        var = ((D @ V) * D).sum(-1).clamp_min(0.0).reshape(TEST_N, J, K)
        se_mean[name] = float(var.sqrt().mean())
        half = Z * var.sqrt()
        lo = (cif_point - half).clamp(0.0, 1.0)
        hi = (cif_point + half).clamp(0.0, 1.0)
        inside = (truth >= lo) & (truth <= hi)
        out[name] = {
            "coverage": float(inside.double().mean()),
            "width": float((hi - lo).mean()),
            # simultaneous: all J*K entries of a subject inside
            "simult": float(inside.reshape(TEST_N, -1).all(dim=1).double().mean()),
        }

    ctd = competing_risk_c_index(cif_point.detach().cpu().numpy(),
                                 co.test["duration"].to_numpy(),
                                 co.test["event"].to_numpy())
    return {
        "eta": eta, "seed": seed, "n": n, "p": int(theta.numel()),
        "rank_H": rank, "grad_before": grad0, "grad_after": grad1,
        "tr_HinvJ_over_rank": ratio_JH,
        "rmse": rmse, "mae": mae,
        "se_naive": se_mean["naive"], "se_sandwich": se_mean["sandwich"],
        "err_over_se_naive": mae / se_mean["naive"] if se_mean["naive"] > 0 else float("nan"),
        "tr_JR": float(torch.diagonal(J_R).sum()),
        "tr_JQ": float(torch.diagonal(J_Q).sum()),
        "tr_Jcross": float(torch.diagonal(J_RQ + J_RQ.T).sum()),
        "tr_Jeta": float(torch.diagonal(J_eta).sum()),
        "tr_H": float(torch.diagonal(H_eta).sum()),
        "omega_star": omega_star,
        "ctd1": float(ctd[0]), "ctd2": float(ctd[1]),
        **{f"{k}_{m}": v[m] for k, v in out.items() for m in ("coverage", "width", "simult")},
    }


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print("=== Last-layer Godambe/sandwich correction ===")
    print(f"device={DEVICE}; seeds={SEEDS}; etas={ETAS}; hidden={STUDENT_HIDDEN}")

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n--- seed {seed} ---", flush=True)
        co, tg, teacher = setup(seed)
        true_cif = true_cif_at_grid(co.test, tg, num_risks=NUM_RISKS)
        for eta in ETAS:
            r = sandwich_at(co, tg, teacher, eta, seed, true_cif)
            rows.append(r)
            print(f"  eta={eta:<5} p={r['p']} rank={r['rank_H']} "
                  f"|grad| {r['grad_before']:.1e}->{r['grad_after']:.1e} "
                  f"omega*={r['omega_star']:.3f} | "
                  f"cov naive={r['naive_coverage']:.3f} sand={r['sandwich_coverage']:.3f} "
                  f"om={r['omega_matched_coverage']:.3f} | "
                  f"wid naive={r['naive_width']:.3f} sand={r['sandwich_width']:.3f} | "
                  f"trJR={r['tr_JR']:.3g} trJQ={r['tr_JQ']:.3g} trJx={r['tr_Jcross']:.3g}",
                  flush=True)
        print(f"  seed {seed} done in {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sandwich_correction.csv", index=False)

    agg = df.groupby("eta").mean(numeric_only=True).reset_index()
    md = ["# Last-Layer Godambe / Sandwich Variance Correction", ""]
    md.append(f"Cell: teacher N={TEACHER_N}, student N={STUDENT_N}, test N={TEST_N}, "
              f"quality={QUALITY}, J={NUM_RISKS}, K={NUM_DURATIONS}, temperature={TEMPERATURE}.")
    md.append(f"Last layer = `head` Linear({STUDENT_HIDDEN}, {NUM_RISKS}); "
              f"p={int(df['p'].iloc[0])} parameters, so H, J and the sandwich are EXACT "
              f"(no diagonal/low-rank approximation). Seeds {SEEDS}, averaged.")
    md.append("")
    md.append("V_naive = H^-1/n (raw generalized posterior, omega=1); "
              "V_sandwich = H^-1 J H^-1 /n (Godambe); "
              "V_omega = H^-1/(n*omega*) with omega* = tr(H^-1)/tr(H^-1 J H^-1).")
    md.append("Intervals are function-space (delta method) on the test-set CIF, "
              "compared against the closed-form true CIF. Nominal 0.95.")
    md.append("")
    md.append("## Coverage / width by eta")
    md.append("")
    md.append("| eta | omega* | cov naive | cov sandwich | cov omega* | wid naive | wid sandwich | "
              "simult naive | simult sandwich | Ctd1 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        md.append(f"| {r['eta']:g} | {r['omega_star']:.3f} | {r['naive_coverage']:.3f} | "
                  f"**{r['sandwich_coverage']:.3f}** | {r['omega_matched_coverage']:.3f} | "
                  f"{r['naive_width']:.3f} | {r['sandwich_width']:.3f} | "
                  f"{r['naive_simult']:.3f} | {r['sandwich_simult']:.3f} | {r['ctd1']:.3f} |")
    md.append("")
    md.append("## Score-variance decomposition by eta")
    md.append("")
    md.append("J_eta = J_R + eta (J_RQ + J_QR) + eta^2 J_Q. The cross term is the quantity the "
              "Cox-style correction omits; a non-negligible `tr(J_RQ+J_QR)` is direct evidence "
              "that the teacher-KL score is correlated with the internal likelihood score.")
    md.append("")
    md.append("| eta | tr(J_R) | tr(J_Q) | tr(J_RQ+J_QR) | tr(J_eta) | tr(H_eta) | cross share |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        share = (r["eta"] * r["tr_Jcross"] / r["tr_Jeta"]) if r["tr_Jeta"] else float("nan")
        md.append(f"| {r['eta']:g} | {r['tr_JR']:.4g} | {r['tr_JQ']:.4g} | {r['tr_Jcross']:.4g} | "
                  f"{r['tr_Jeta']:.4g} | {r['tr_H']:.4g} | {share:+.3f} |")
    md.append("")
    md.append("## Is undercoverage a variance problem or a bias problem?")
    md.append("")
    md.append("`MAE` is the actual mean |CIF error| against the closed-form truth; `SE` is the "
              "model-implied standard error. If MAE >> SE, the intervals are mis-centred "
              "(bias / misspecification), and NO variance correction can reach nominal coverage.")
    md.append("")
    md.append("| eta | MAE | RMSE | SE naive | SE sandwich | MAE / SE naive |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        md.append(f"| {r['eta']:g} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['se_naive']:.4f} | "
                  f"{r['se_sandwich']:.4f} | **{r['err_over_se_naive']:.2f}** |")
    md.append("")
    md.append("Raw per-seed rows: `sandwich_correction.csv`.")
    (OUT_DIR / "sandwich_correction_report.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
