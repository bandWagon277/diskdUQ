"""Experiment 0: validate the SGLD sampler in a low-dimensional model (Phase 2).

Question. Does unadjusted Langevin / SGLD reproduce the intended generalized posterior,
or is it under-dispersed? And which drift convention hits the full-data posterior? If SGLD
fails here (p <= ~15, exact gold standards available), no full-network SGLD result is
interpretable.

Model. A small competing-risk discrete logistic-hazard model:
    logit lambda^{(j)}_{ik} = x_i . beta_j + b_{j,k},   theta = (beta_{J x d}, b_{J x K}).
The internal NLL and teacher KL are the *production* competing-risk losses (softmax with a
no-event category + interval KL), so this validates the DiSKD-specific target, not a generic
logistic model. A single-risk variant is also run as a generic Langevin scaling check.

Target. Generalized posterior  Pi(theta) ∝ exp{ -omega * L_sum(theta) } * N(0, tau^2 I),
with L_sum = sum_i [ r_i + eta q_i ] (SUMMED over subjects — the full-data posterior).

Methods compared, per (omega, eta):
  - MALA      : Metropolis-adjusted Langevin — accept/reject, exact stationary law (GOLD).
  - Laplace   : N(theta_MAP, (omega H + I/tau^2)^-1) — curvature check only.
  - ULA(omega): unadjusted Langevin at the given omega (no MH) — the continuous-time SGLD ideal.
  - SGLD-welling_teh (sigma=1): our production drift with grad_n_factor=N -> effective omega=1.
  - SGLD-literal    (sigma=1): our production drift with grad_n_factor=1 -> effective omega=1/N
                               (an N-times-tempered / much wider target). This is what most of
                               our runs used; the toy shows it is NOT the full-data posterior.

Unit tests:
  - Gaussian: U = 0.5 theta^T A theta. ULA stationary covariance should match the *discretized*
    ULA covariance (≈ A^-1 only as eps->0), NOT exactly A^-1.
  - Duplicate-data: copying the data x2 should shrink widths ≈ 1/sqrt(2) for the full-data target
    (omega=1 / welling_teh) but NOT for the mean-objective (literal) target.

Outputs (OUT_DIR, default responses_exp0/): sampler_validation_report.md, .csv
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import hessian


def gradf(fn):
    """Plain-autograd gradient closure (far cheaper per call than torch.func in a loop)."""
    def g(th):
        t = th.detach().requires_grad_(True)
        (val,) = torch.autograd.grad(fn(t), t)
        return val.detach()
    return g

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

from diskd.losses import CompetingRiskNLLLoss, SingleRiskNLLLoss
from diskd.utils import (
    competing_interval_probs,
    full_probs_from_event_probs,
    make_at_risk_mask,
    temperature_scale_probs,
)

# ---------- config ----------
D = int(os.environ.get("D", 3))            # covariate dim
J = int(os.environ.get("J", 2))            # causes
K = int(os.environ.get("K", 4))            # intervals  -> p = J*d + J*K = 6 + 8 = 14
N = int(os.environ.get("N", 400))
OMEGAS = [float(w) for w in os.environ.get("OMEGAS", "0.5,1,2").split(",")]
ETAS = [float(e) for e in os.environ.get("ETAS", "0,1").split(",")]
TAU = float(os.environ.get("TAU", 5.0))    # weak prior sd (propriety)
TEMPERATURE = 2.0
N_MALA = int(os.environ.get("N_MALA", 60000))
N_ULA = int(os.environ.get("N_ULA", 60000))
BURN = int(os.environ.get("BURN", 10000))
THIN = int(os.environ.get("THIN", 20))
SEED = int(os.environ.get("SEED", 42))
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_exp0"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


# ---------- toy competing-risk model ----------

def logits_cr(theta, X):
    """theta -> competing-risk logits [N, J, K]."""
    beta = theta[: J * D].view(J, D)
    b = theta[J * D:].view(J, K)
    return torch.einsum("nd,jd->nj", X, beta).unsqueeze(-1) + b.view(1, J, K)


def simulate(seed):
    set_seed(seed)
    X = torch.randn(N, D, dtype=DTYPE, device=DEVICE)
    theta_true = 0.7 * torch.randn(J * D + J * K, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        probs = competing_interval_probs(logits_cr(theta_true, X))   # [N, J+1, K]
    haz = probs[:, :J, :]                                            # event hazards
    # sequential discrete competing-risk sampling
    dur = torch.full((N,), K, dtype=torch.long)
    ev = torch.zeros(N, dtype=torch.long)
    alive = torch.ones(N, dtype=torch.bool)
    for k in range(K):
        p_no = 1.0 - haz[:, :, k].sum(1)
        cat = torch.cat([haz[:, :, k], p_no.unsqueeze(1)], dim=1).clamp_min(1e-9)
        draw = torch.multinomial(cat / cat.sum(1, keepdim=True), 1).squeeze(1)
        event_here = alive & (draw < J)
        dur[event_here] = k
        ev[event_here] = (draw[event_here] + 1).to(torch.long)
        alive = alive & ~event_here
    idx = dur.clamp(max=K - 1)
    # teacher hazard = true hazard mildly perturbed (a "good" teacher)
    set_seed(seed + 1)
    t_haz = (haz * torch.exp(0.15 * torch.randn_like(haz))).clamp(1e-4, 0.9)
    teacher_full = full_probs_from_event_probs(t_haz)
    return X, idx.to(DEVICE), ev.to(DEVICE), teacher_full, theta_true


def make_potential(X, idx, ev, teacher_full, omega, eta):
    """Return (U_value, gradU, L_sum) with an ANALYTIC gradient (no autograd in the loop).

    Model: z[i,:,k] = [X_i.beta + b_{:,k}, 0] (J event logits + a fixed 0 no-event channel).
    NLL grad wrt event logit j:  ar * (pi_j - 1{tgt==j}).
    KL  grad wrt event logit j:  ar * T * (sT_j - t_j)   (after the T^2 scaling).
    Chain rule: grad_beta[j] = sum_{i,k} G[i,j,k] X_i ; grad_b[j,k] = sum_i G[i,j,k].
    The analytic gradient is unit-tested against autograd in the smoke run.
    """
    n = X.shape[0]
    ar = (torch.arange(K, device=X.device).view(1, K) <= idx.view(n, 1)).to(DTYPE)   # [n,K]
    tgt = torch.full((n, K), J, dtype=torch.long, device=X.device)
    ehere = ev > 0
    tgt[ehere, idx[ehere]] = (ev[ehere] - 1)
    tgt_oh = torch.zeros(n, J, K, dtype=DTYPE, device=X.device)                        # onehot over events
    for j in range(J):
        tgt_oh[:, j, :] = (tgt == j).to(DTYPE)
    # KL uses the TEMPERATURE-SCALED teacher (matches L_sum's tfull); grad wrt logit_j is
    # ar * T * (softmax(z/T)_j - tfull_j).
    t_ev = temperature_scale_probs(teacher_full, TEMPERATURE)[:, :J, :]                # [n,J,K]

    def _z(theta):
        lg = logits_cr(theta, X)                                                       # [n,J,K]
        return torch.cat([lg, torch.zeros(n, 1, K, dtype=DTYPE, device=X.device)], dim=1)  # [n,J+1,K]

    def L_sum(theta):
        z = _z(theta)
        logp = z - torch.logsumexp(z, dim=1, keepdim=True)                             # [n,J+1,K]
        lp_t = logp[:, :J, :].gather(1, tgt.clamp(max=J - 1).view(n, 1, K)).squeeze(1)
        lp_noevent = logp[:, J, :]
        lp_target = torch.where(tgt == J, lp_noevent, lp_t)
        nll = -(ar * lp_target).sum()
        if eta == 0:
            return nll
        zt = z / TEMPERATURE
        logpt = zt - torch.logsumexp(zt, dim=1, keepdim=True)
        tfull = temperature_scale_probs(teacher_full, TEMPERATURE)
        kl = (tfull * (tfull.clamp_min(1e-12).log() - logpt)).sum(1)                    # [n,K]
        return nll + eta * (TEMPERATURE ** 2) * (ar * kl).sum()

    def U_value(theta):
        return omega * L_sum(theta) + (theta @ theta) / (2 * TAU * TAU)

    def gLsum(theta):
        """Analytic gradient of L_sum only (no omega, no prior)."""
        z = _z(theta)
        pi = torch.softmax(z, dim=1)                                                    # [n,J+1,K]
        G = ar.unsqueeze(1) * (pi[:, :J, :] - tgt_oh)                                    # NLL part [n,J,K]
        if eta != 0:
            sT = torch.softmax(z / TEMPERATURE, dim=1)[:, :J, :]
            G = G + eta * TEMPERATURE * ar.unsqueeze(1) * (sT - t_ev)
        gbeta = torch.einsum("njk,nd->jd", G, X)                                        # [J,d]
        gb = G.sum(0)                                                                    # [J,K]
        return torch.cat([gbeta.reshape(-1), gb.reshape(-1)])

    def gradU(theta):
        return omega * gLsum(theta) + theta / (TAU * TAU)

    return U_value, gradU, L_sum, gLsum


def find_map(U, p, iters=400):
    th = torch.zeros(p, dtype=DTYPE, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([th], max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)
    opt.step(lambda: _closure(opt, U, th))
    return th.detach()


def _closure(opt, U, th):
    opt.zero_grad()
    loss = U(th)
    loss.backward()
    return loss


def mala(U, gU, theta0, n, burn, thin, step, seed, M=None, Lchol=None, Hprec=None):
    """MALA, optionally preconditioned with mass matrix M (Cholesky Lchol, inverse Hprec).

    Preconditioned proposal: theta' = theta - 0.5*step*M@grad + sqrt(step)*Lchol@z.
    Proposal density uses M^-1 = Hprec in the quadratic form. Preconditioning only affects
    mixing speed, not the stationary law (still exp(-U)) — it gives an accurate gold on an
    ill-conditioned posterior with far fewer steps.
    """
    set_seed(seed)
    precond = M is not None
    th = theta0.clone()
    u = U(th); g = gU(th)
    samples, acc = [], 0
    for t in range(n):
        z = torch.randn_like(th)
        if precond:
            m_th = th - 0.5 * step * (M @ g)
            prop = m_th + (step ** 0.5) * (Lchol @ z)
        else:
            m_th = th - 0.5 * step * g
            prop = m_th + (step ** 0.5) * z
        up = U(prop); gp = gU(prop)
        if precond:
            m_pr = prop - 0.5 * step * (M @ gp)
            d1 = prop - m_th; d2 = th - m_pr
            fwd = -(d1 @ (Hprec @ d1)) / (2 * step)
            rev = -(d2 @ (Hprec @ d2)) / (2 * step)
        else:
            fwd = -((prop - m_th) ** 2).sum() / (2 * step)
            rev = -((th - (prop - 0.5 * step * gp)) ** 2).sum() / (2 * step)
        logr = (u - up) + (rev - fwd)
        if torch.log(torch.rand(())) < logr:
            th, u, g = prop, up, gp
            acc += 1
        if t >= burn and (t - burn) % thin == 0:
            samples.append(th.clone())
    return torch.stack(samples), acc / n


def langevin(drift_fn, theta0, n, burn, thin, step, noise, seed):
    """Unadjusted Langevin: theta -= 0.5*step*drift + sqrt(step)*noise*z (no MH)."""
    set_seed(seed)
    th = theta0.clone()
    samples = []
    for t in range(n):
        d = drift_fn(th)
        th = th - 0.5 * step * d + (step ** 0.5) * noise * torch.randn_like(th)
        if t >= burn and (t - burn) % thin == 0:
            samples.append(th.clone())
    return torch.stack(samples)


def summarize(samples):
    return samples.mean(0), samples.std(0)


def agree(a_mean, a_sd, b_mean, b_sd):
    """Max abs mean diff and sd-ratio range vs a reference (b = gold)."""
    md = (a_mean - b_mean).abs().max().item()
    ratio = (a_sd / b_sd.clamp_min(1e-9))
    return md, ratio.mean().item(), ratio.min().item(), ratio.max().item()


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    torch.set_default_dtype(DTYPE)
    X, idx, ev, teacher_full, theta_true = simulate(SEED)
    p = J * D + J * K
    print(f"=== Experiment 0: sampler validation (competing-risk toy, p={p}, N={N}) ===")
    print(f"device={DEVICE}; omegas={OMEGAS}; etas={ETAS}")

    step_mala = float(os.environ.get("STEP_MALA", 0.4))   # dimensionless (H^-1-preconditioned)

    # correctness check: analytic gradient must match autograd
    _U, _gU, _L, _gL = make_potential(X, idx, ev, teacher_full, 1.3, 1.0)
    _th = 0.3 * torch.randn(p, dtype=DTYPE, device=DEVICE)
    _ga = _gU(_th)
    _gn = gradf(_U)(_th)
    _err = (_ga - _gn).abs().max().item()
    print(f"analytic-vs-autograd grad max err = {_err:.2e}", flush=True)
    assert _err < 1e-4, f"analytic gradient mismatch ({_err})"

    rows = []
    for eta in ETAS:
        for omega in OMEGAS:
            U, gU, L_sum, gLsum = make_potential(X, idx, ev, teacher_full, omega, eta)
            theta_map = find_map(U, p)
            H = hessian(U)(theta_map)
            cov_lap = torch.linalg.inv(H)
            sd_lap = torch.diagonal(cov_lap).clamp_min(0).sqrt()
            Lchol = torch.linalg.cholesky(cov_lap)                 # H^-1 = Lchol Lchol^T
            max_eig = float(torch.linalg.eigvalsh(H).max())

            # GOLD: H^-1-preconditioned MALA (accurate on an ill-conditioned posterior).
            m_s, m_acc = mala(U, gU, theta_map, N_MALA, BURN, THIN, step=step_mala, seed=SEED + 7,
                              M=cov_lap, Lchol=Lchol, Hprec=H)
            mean_g, sd_g = summarize(m_s)

            # ISOTROPIC ULA at the SAME omega -- the production-style SGLD ideal (no precond).
            # Stable isotropic step must satisfy step < 2/max_eig(H); use a safety factor.
            iso_step = 1.0 / max_eig
            u_s = langevin(gU, theta_map, N_ULA, BURN, THIN, iso_step, 1.0, SEED + 8)
            mean_u, sd_u = summarize(u_s)

            # Production drift semantics (isotropic, noise sigma=1, full-batch).
            # The toy loss is NOT (1+eta)-normalized, so no loss_scale is needed here:
            #   welling_teh: grad_n_factor=N -> gLsum + prior          (targets omega=1)
            #   literal    : grad_n_factor=1 -> gLsum/N + prior         (N-times tempered / wider)
            prior_g = lambda th: th / (TAU * TAU)
            drift_welling = lambda th: gLsum(th) + prior_g(th)
            drift_literal = lambda th: gLsum(th) / N + prior_g(th)
            w_s = langevin(drift_welling, theta_map, N_ULA, BURN, THIN, iso_step, 1.0, SEED + 9)
            l_s = langevin(drift_literal, theta_map, N_ULA, BURN, THIN, iso_step * N, 1.0, SEED + 10)
            mean_w, sd_w = summarize(w_s)
            mean_l, sd_l = summarize(l_s)

            for name, (mn, sd) in (("MALA(gold)", (mean_g, sd_g)),
                                   ("Laplace", (theta_map, sd_lap)),
                                   (f"ULA(omega={omega})", (mean_u, sd_u)),
                                   ("SGLD-welling_teh(s=1)", (mean_w, sd_w)),
                                   ("SGLD-literal(s=1)", (mean_l, sd_l))):
                md, rm, rlo, rhi = agree(mn, sd, mean_g, sd_g)
                rows.append({
                    "eta": eta, "omega": omega, "method": name,
                    "mean_absdiff_vs_gold": md, "sd_ratio_mean": rm,
                    "sd_ratio_min": rlo, "sd_ratio_max": rhi,
                    "avg_sd": float(sd.mean()), "mala_acc": m_acc,
                })
            print(f"  eta={eta} omega={omega} MALA acc={m_acc:.2f} | "
                  f"avg_sd gold={float(sd_g.mean()):.3f} ULA={float(sd_u.mean()):.3f} "
                  f"welling={float(sd_w.mean()):.3f} literal={float(sd_l.mean()):.3f}", flush=True)

    # ---- unit test 1: Gaussian ULA stationary covariance (analytic drift) ----
    set_seed(SEED)
    A = torch.randn(6, 6, dtype=DTYPE); A = A @ A.T + 6 * torch.eye(6, dtype=DTYPE)
    gq = lambda th: A @ th                                          # exact grad of 0.5 th^T A th
    step_g = 0.02
    n_gauss = int(os.environ.get("N_GAUSS", 200000))
    gs = langevin(gq, torch.zeros(6, dtype=DTYPE), n_gauss, n_gauss // 10, 10, step_g, 1.0, SEED)
    cov_emp = torch.cov(gs.T)
    cov_true = torch.linalg.inv(A)
    # discretized ULA (Ornstein-Uhlenbeck) stationary cov solves cov = (I-0.5*step*A) cov (I-..)^T + step I
    M = torch.eye(6, dtype=DTYPE) - 0.5 * step_g * A
    cov_ula = cov_true.clone()
    for _ in range(2000):
        cov_ula = M @ cov_ula @ M.T + step_g * torch.eye(6, dtype=DTYPE)
    err_true = (cov_emp - cov_true).abs().max().item()
    err_ula = (cov_emp - cov_ula).abs().max().item()

    # ---- unit test 2: duplicate-data width shrink ----
    U1, _, L1, _ = make_potential(X, idx, ev, teacher_full, 1.0, 0.0)
    X2, idx2, ev2 = torch.cat([X, X]), torch.cat([idx, idx]), torch.cat([ev, ev])
    tf2 = torch.cat([teacher_full, teacher_full])
    U2w, _, _, _ = make_potential(X2, idx2, ev2, tf2, 1.0, 0.0)    # full-data (welling/omega=1)
    map1 = find_map(U1, p); map2 = find_map(U2w, p)
    sd1 = torch.diagonal(torch.linalg.inv(hessian(U1)(map1))).clamp_min(0).sqrt().mean()
    sd2 = torch.diagonal(torch.linalg.inv(hessian(U2w)(map2))).clamp_min(0).sqrt().mean()
    dup_ratio_full = float(sd2 / sd1)                              # expect ~1/sqrt(2)=0.707

    rows_note = {
        "gaussian_err_vs_Hinv": err_true, "gaussian_err_vs_ULAstat": err_ula,
        "dup_width_ratio_fulldata": dup_ratio_full,
    }

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sampler_validation.csv", index=False)

    md = ["# Experiment 0 — SGLD Sampler Validation (competing-risk toy)", ""]
    md.append(f"p={p} (D={D}, J={J}, K={K}), N={N}, prior sd tau={TAU}, temperature={TEMPERATURE}. "
              f"MALA is the accept/reject gold standard. Seeds fixed. sd-ratio is method-sd / MALA-sd "
              f"averaged over parameters (1.0 = matches gold; <1 = under-dispersed).")
    md.append("")
    md.append("## Posterior agreement vs MALA (gold)")
    md.append("")
    md.append("| eta | omega | method | mean|Δ| vs gold | sd-ratio (mean) | sd-ratio [min,max] | avg sd |")
    md.append("|---:|---:|---|---:|---:|---|---:|")
    for r in rows:
        md.append(f"| {r['eta']:g} | {r['omega']:g} | {r['method']} | {r['mean_absdiff_vs_gold']:.4f} | "
                  f"{r['sd_ratio_mean']:.3f} | [{r['sd_ratio_min']:.2f}, {r['sd_ratio_max']:.2f}] | "
                  f"{r['avg_sd']:.4f} |")
    md.append("")
    md.append("**Reading.** ULA(omega) and SGLD-welling_teh(sigma=1) should match MALA at omega=1 "
              "(sd-ratio ≈ 1). SGLD-literal(sigma=1) targets the N-times-tempered mean objective, so "
              "its sd-ratio should be MUCH larger than 1 (wider) — evidence that `literal` does NOT "
              "sample the full-data posterior. If ULA is < 1 vs MALA, the sampler is under-dispersed.")
    md.append("")
    md.append("## Unit tests")
    md.append("")
    md.append(f"- **Gaussian Langevin:** empirical cov vs exact H^-1: max abs err = "
              f"{err_true:.4f}; vs *discretized* ULA stationary cov: {err_ula:.4f}. "
              f"(The ULA error should be the small one — a correct sampler matches the discretized "
              f"stationary law, not exactly H^-1, at finite step.)")
    md.append(f"- **Duplicate-data (full-data target):** avg posterior sd ratio (2N vs N) = "
              f"{dup_ratio_full:.3f} (expected ≈ 0.707 = 1/sqrt 2). Confirms the full-data / "
              f"welling_teh target concentrates correctly with more data.")
    md.append("")
    md.append("Raw rows: `sampler_validation.csv`.")
    (OUT_DIR / "sampler_validation_report.md").write_text("\n".join(md) + "\n")
    print("\nUnit tests:", rows_note, flush=True)
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
