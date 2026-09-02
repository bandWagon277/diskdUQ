#!/usr/bin/env python
"""Eta selection by formal, interpretable criteria (not coverage/width).

Criteria (from eta_selection_plan.tex), each evaluated over the eta grid:
  * LPML / CPO  : leave-one-subject-out predictive; log CPO_i via the DiSKD
                  importance-weight identity.  Select argmax.
  * WAIC        : widely-applicable information criterion (LOO-consistent).  argmin.
  * DIC         : deviance information criterion (student deviance).          argmin.
  * GBIC        : generalized BIC with last-layer effective df.               argmin.

Setting (per user, 2026-08-10): ORIGINAL DiSKD DGP (squared rates, Section 1.3),
fixed-epsilon SGLD, 500 iterations, 5 experiments (seeds).  We use a LAST-LAYER
Bayesian posterior: freeze the AdamW-MAP backbone and sample the 66-dim output
head from the EXACT generalized posterior
    Pi_eta(w) ∝ pi0(w) * L(w) * exp{-eta * Q(w)},   Q = T^2 * sum_i KL_i,
whose potential is U(w) = NLL_sum(w) + eta * Q_sum(w) (NO 1/(1+eta) tempering),
so the CPO / DIC identities hold exactly.  GBIC uses the 66x66 head Hessians.

Env: DGP=squared D=12 CENSOR_MAX=0.05 BETA_R1/2/SHARED, NUM_DURATIONS=20,
TEACHER_N=5000 STUDENT_N=500, ETAS (10-pt grid in [0,5]), SEEDS (5),
SGLD_ITERS=500 EPS (fixed) N_CHAINS SAMPLES_PER_CHAIN BURNIN PRIOR_PREC TEMP.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diskd import DiscreteSurvivalModel, DiSKDStudent, fit_time_grid, transform_durations
from diskd.networks import sinusoidal_time_embedding
from diskd.losses import CompetingRiskNLLLoss
from diskd.utils import (competing_interval_probs, full_probs_from_event_probs,
                         make_at_risk_mask, temperature_scale_probs)

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
D = int(os.environ.get("D", 12)); J = 2
K = int(os.environ.get("NUM_DURATIONS", 20))
TEACHER_N = int(os.environ.get("TEACHER_N", 5000)); STUDENT_N = int(os.environ.get("STUDENT_N", 500))
HID = int(os.environ.get("HIDDEN", 32)); BATCH = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100)); ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 60))
TEMP = float(os.environ.get("TEMP", 2.0))
# 10-point eta grid in [0,5]
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.5,1,1.5,2,2.5,3,3.5,4,5").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
SGLD_ITERS = int(os.environ.get("SGLD_ITERS", 500)); EPS = float(os.environ.get("EPS", 2e-5))
N_CHAINS = int(os.environ.get("N_CHAINS", 4)); SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 200))
BURNIN = int(os.environ.get("BURNIN", 100)); PRIOR_PREC = float(os.environ.get("PRIOR_PREC", 1e-2))
# posterior for the criteria: "sgld" (fixed-eps), "laplace" (analytic, mixing-free), or "both"
POSTERIOR = os.environ.get("POSTERIOR", "both"); LAPLACE_M = int(os.environ.get("LAPLACE_M", 800))
# original DiSKD squared-rate DGP
CENSOR_MAX = float(os.environ.get("CENSOR_MAX", 0.05))
BETA_R1 = float(os.environ.get("BETA_R1", 2.0)); BETA_R2 = float(os.environ.get("BETA_R2", 2.0))
BETA_SHARED = float(os.environ.get("BETA_SHARED", 8.0))
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_eta_criteria"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def rates(x):
    z1 = x[:, 0:4].sum(1); z2 = x[:, 4:8].sum(1); z3 = x[:, 8:12].sum(1)
    r1 = np.clip((BETA_R1*z1)**2 + (BETA_SHARED*z3)**2, 1e-3, None)
    r2 = np.clip((BETA_R2*z2)**2 + (BETA_SHARED*z3)**2, 1e-3, None)
    return np.stack([r1, r2], axis=1)

def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); r = rates(x)
    t = rng.exponential(scale=1.0/r); cause = t.argmin(1); et = t.min(1)
    cens = rng.uniform(0.0, CENSOR_MAX, size=n); censored = cens < et
    dur = np.where(censored, cens, et); event = np.where(censored, 0, cause+1).astype(np.int64)
    df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(D)]); df["duration"] = dur.astype(float); df["event"] = event
    return df


def backbone_feats(net, x):
    net.eval()
    with torch.no_grad():
        h = net.feature_projection(x).unsqueeze(1)
        emb = sinusoidal_time_embedding(net.num_durations, h.shape[-1], device=x.device, dtype=x.dtype)
        h = net.blocks(h + emb.unsqueeze(0))
    return h.double()                                           # [N,K,HID]


def make_ops(feat, idx, ev, tfull):
    """Return head->logits, exact potential builder, and per-subject NLL_i / Q_i."""
    Hd = HID; nll = CompetingRiskNLLLoss(); mask = make_at_risk_mask(idx, K).double()
    def logits_of(th):
        W = th[:J*Hd].view(J, Hd); bb = th[J*Hd:]
        return torch.einsum("nkd,jd->njk", feat, W) + bb.view(1, J, 1)
    def nll_sum(th):
        return nll(logits_of(th), idx, ev, reduction="sum")
    def q_sum(th):
        lg = logits_of(th); tt = temperature_scale_probs(tfull, TEMP)
        st = temperature_scale_probs(competing_interval_probs(lg), TEMP)
        kl = (tt*(tt.clamp_min(1e-12).log() - st.clamp_min(1e-12).log())).sum(1)   # [N,K]
        return (TEMP*TEMP)*(kl*mask).sum()
    def potential(th, eta):                                     # U = NLL_sum + eta*Q_sum + prior
        u = nll_sum(th) + 0.5*PRIOR_PREC*(th*th).sum()
        return u if eta == 0 else u + eta*q_sum(th)
    def per_subject(th):                                        # (NLL_i [N], Q_i [N])
        lg = logits_of(th)
        nll_i = nll(lg, idx, ev, reduction="none")
        tt = temperature_scale_probs(tfull, TEMP)
        st = temperature_scale_probs(competing_interval_probs(lg), TEMP)
        kl = (tt*(tt.clamp_min(1e-12).log() - st.clamp_min(1e-12).log())).sum(1)
        q_i = (TEMP*TEMP)*(kl*mask).sum(1)
        return nll_i, q_i
    return logits_of, nll_sum, q_sum, potential, per_subject


def sgld_fixed(potential, eta, theta0, seed):
    """Fixed-epsilon SGLD on the head; warm-started from theta0 (the MAP head)."""
    torch.manual_seed(seed); th = theta0.clone(); draws = []
    thin = max(1, (SGLD_ITERS - BURNIN) // SAMPLES_PER_CHAIN)
    for t in range(SGLD_ITERS):
        thr = th.clone().requires_grad_(True)
        g = torch.autograd.grad(potential(thr, eta), thr)[0]
        th = (th - 0.5*EPS*g + (EPS**0.5)*torch.randn_like(th)).detach()
        if t >= BURNIN and (t-BURNIN) % thin == 0 and len(draws) < SAMPLES_PER_CHAIN:
            draws.append(th.clone())
    return draws


def logmeanexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def criteria(nll_arr, q_arr, thbar, per_subject, nll_sum, eta, n):
    """nll_arr,q_arr: [M,N]. Returns dict of criterion values."""
    M, N = nll_arr.shape
    # --- LPML / CPO: log CPO_i = LSE_m(eta Q_i) - LSE_m(eta Q_i + NLL_i)  (M cancels) ---
    a = eta*q_arr                                              # [M,N]
    num = logmeanexp(a, axis=0)                                # [N]
    den = logmeanexp(a + nll_arr, axis=0)                      # [N]  (1/L_i = e^{nll_i})
    log_cpo = num - den
    lpml = float(np.sum(log_cpo))
    # importance-weight ESS (denominator weights) as a stability check
    w = np.exp((a + nll_arr) - np.max(a + nll_arr, axis=0, keepdims=True))
    ess_w = float(np.mean((w.sum(0)**2) / np.sum(w**2, axis=0)))
    # --- WAIC ---
    lppd = float(np.sum(logmeanexp(-nll_arr, axis=0)))
    p_waic = float(np.sum(np.var(nll_arr, axis=0, ddof=1)))
    waic = -2.0*(lppd - p_waic)
    # --- DIC (classical, last-layer theta-bar) ---
    Dbar = 2.0*float(np.mean(nll_arr.sum(axis=1)))
    with torch.no_grad():
        Dhat = 2.0*float(nll_sum(thbar).item())
    p_dic = Dbar - Dhat; dic = Dbar + p_dic
    return dict(lpml=lpml, waic=waic, dic=dic, p_waic=p_waic, p_dic=p_dic, ess_w=ess_w)


def hessians(nll_sum, q_sum, th_map, eta):
    """Head Hessians at the MAP: H_R = d^2 NLL, H_Q = d^2 Q (None if eta==0)."""
    from torch.autograd.functional import hessian
    H_R = hessian(nll_sum, th_map).detach()
    H_Q = None if eta == 0 else hessian(q_sum, th_map).detach()
    return H_R, H_Q


def gbic(nll_sum, th_map, H_R, H_Q, eta, n):
    """GBIC = 2*NLL(th_map) + df_eff*log n, df_eff = tr(H_R (H_R+eta H_Q)^-1)."""
    p = th_map.numel(); ridge = 1e-6*float(torch.diagonal(H_R).abs().mean()) + 1e-8
    if eta == 0 or H_Q is None:
        df_eff = float(p)
    else:
        A = H_R + eta*H_Q + ridge*torch.eye(p, dtype=H_R.dtype, device=H_R.device)
        df_eff = float(torch.diagonal(torch.linalg.solve(A, H_R)).sum())
    dev = 2.0*float(nll_sum(th_map).item())
    return dev + df_eff*np.log(n), df_eff


def laplace_draws(th_map, H_R, H_Q, eta, seed, M):
    """Analytic last-layer posterior N(th_map, (H_R + eta H_Q + prior)^-1) draws."""
    p = th_map.numel(); dev = H_R.device
    prec = H_R + PRIOR_PREC*torch.eye(p, dtype=H_R.dtype, device=dev)
    if eta != 0 and H_Q is not None:
        prec = prec + eta*H_Q
    ridge = 1e-6*float(torch.diagonal(prec).abs().mean()) + 1e-9
    cov = torch.linalg.inv(prec + ridge*torch.eye(p, dtype=prec.dtype, device=dev))
    cov = 0.5*(cov + cov.T)                                    # symmetrize
    L = torch.linalg.cholesky(cov)
    g = torch.Generator(device=dev if dev.type == "cpu" else "cpu").manual_seed(seed)
    z = torch.randn(M, p, generator=g, dtype=th_map.dtype).to(dev)
    return [th_map + L @ z[m] for m in range(M)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    print(f"=== Eta selection by criteria (DGP=squared, K={K}) === LPML/WAIC/DIC/GBIC | "
          f"etas={ETAS} seeds={SEEDS} | last-layer SGLD {SGLD_ITERS} iters, FIXED eps={EPS:g}, "
          f"{N_CHAINS}x{SAMPLES_PER_CHAIN} draws", flush=True)
    rows = []
    for seed in SEEDS:
        teacher_df = simulate(TEACHER_N, 10*seed+1); student_df = simulate(STUDENT_N, 10*seed+2)
        tg = fit_time_grid(np.concatenate([teacher_df["duration"].values, student_df["duration"].values]), K)
        feats = [f"x{i+1}" for i in range(D)]
        teacher = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=HID, epochs=TEACHER_EPOCHS,
                                        batch_size=BATCH, device=DEVICE, time_grid=tg).fit(teacher_df, feature_cols=feats)
        tp = teacher.predict_interval_probs(student_df)[:, :J, :]
        for eta in ETAS:
            torch.manual_seed(1000*seed); np.random.seed(1000*seed)
            common = dict(num_risks=J, num_durations=K, hidden_dim=HID, epochs=ADAMW_EPOCHS, batch_size=BATCH,
                          device=DEVICE, optimizer="adamw", time_grid=tg)
            adam = (DiscreteSurvivalModel(**common) if eta == 0 else
                    DiSKDStudent(teacher_model=teacher, teacher_type="competing", eta=eta, temperature=TEMP, **common))
            adam.fit(student_df, feature_cols=feats)
            net = adam.net
            xtr = torch.as_tensor(adam.preprocessor.transform(student_df), dtype=torch.float32, device=DEVICE)
            idx = torch.as_tensor(transform_durations(student_df["duration"].values, tg), dtype=torch.long, device=DEVICE)
            ev = torch.as_tensor(student_df["event"].values.copy(), dtype=torch.long, device=DEVICE)
            tfull = full_probs_from_event_probs(torch.as_tensor(tp, dtype=torch.float64, device=DEVICE))
            feat = backbone_feats(net, xtr)
            logits_of, nll_sum, q_sum, potential, per_subject = make_ops(feat, idx, ev, tfull)
            th_map = torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
            H_R, H_Q = hessians(nll_sum, q_sum, th_map, eta)
            gb, df_eff = gbic(nll_sum, th_map, H_R, H_Q, eta, STUDENT_N)

            modes = ["sgld", "laplace"] if POSTERIOR == "both" else [POSTERIOR]
            for mode in modes:
                if mode == "sgld":
                    draws = []
                    for c in range(N_CHAINS):
                        draws += sgld_fixed(potential, eta, th_map + 1e-3*torch.randn_like(th_map), 100*seed+c)
                else:
                    draws = laplace_draws(th_map, H_R, H_Q, eta, 100*seed, LAPLACE_M)
                with torch.no_grad():
                    nll_arr = np.stack([per_subject(th)[0].cpu().numpy() for th in draws])   # [M,N]
                    q_arr = np.stack([per_subject(th)[1].cpu().numpy() for th in draws])      # [M,N]
                    thbar = torch.stack(draws).mean(0)
                crit = criteria(nll_arr, q_arr, thbar, per_subject, nll_sum, eta, STUDENT_N)
                rec = dict(seed=seed, eta=eta, post=mode, **crit, gbic=gb, df_eff=df_eff)
                rows.append(rec)
                print(f"  seed{seed} eta={eta:<4g} [{mode:7s}] | LPML={crit['lpml']:8.1f} WAIC={crit['waic']:8.1f} "
                      f"DIC={crit['dic']:8.1f} GBIC={gb:8.1f} | pWAIC={crit['p_waic']:5.1f} dfeff={df_eff:5.1f} "
                      f"wESS={crit['ess_w']:.0f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR/"eta_criteria.csv", index=False)
    _report(df)
    print(f"\nSaved to {OUT_DIR}", flush=True)


def _report(df):
    md = ["# Eta selection by formal criteria (original DiSKD DGP)", ""]
    md.append(f"Squared-rate DGP (beta={BETA_R1}/{BETA_R2}/{BETA_SHARED}), K={K}, teacher/student={TEACHER_N}/{STUDENT_N}. "
              f"Last-layer EXACT generalized posterior Pi_eta ∝ pi0*L*exp(-eta*Q), TEMP={TEMP}, seeds {SEEDS}. "
              f"Posteriors: SGLD ({SGLD_ITERS} fixed-eps={EPS:g} iters, {N_CHAINS}x{SAMPLES_PER_CHAIN} draws) "
              f"and analytic Laplace ({LAPLACE_M} draws). Rule: LPML argmax; WAIC/DIC/GBIC argmin. Means over seeds.")
    md.append("")
    posts = list(df["post"].unique()) if "post" in df.columns else [None]
    for post in posts:
        d = df if post is None else df[df.post == post]
        md.append(f"## Posterior: {post if post else 'default'}")
        note = ("  *(under-dispersed: p_WAIC << df_eff, so WAIC/DIC collapse to deviance)*"
                if post == "sgld" else
                "  *(trustworthy: p_WAIC tracks df_eff)*" if post == "laplace" else "")
        md.append(note); md.append("")
        md.append("| eta | LPML (max) | WAIC (min) | DIC (min) | GBIC (min) | p_WAIC | df_eff | wESS |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for e in ETAS:
            s = d[d.eta == e]
            md.append(f"| {e:g} | {s['lpml'].mean():.1f} | {s['waic'].mean():.1f} | {s['dic'].mean():.1f} | "
                      f"{s['gbic'].mean():.1f} | {s['p_waic'].mean():.1f} | {s['df_eff'].mean():.1f} | {s['ess_w'].mean():.0f} |")
        md.append("")
        def sel(col, how):
            hats = []
            for seed in sorted(d.seed.unique()):
                s = d[d.seed == seed]
                if len(s): hats.append(float(s.loc[(s[col].idxmax() if how == "max" else s[col].idxmin()), "eta"]))
            return hats
        for col, how in [("lpml", "max"), ("waic", "min"), ("dic", "min"), ("gbic", "min")]:
            h = sel(col, how)
            md.append(f"- **{col.upper()} eta_hat per seed:** {h}  (mean {np.mean(h):.2f})" if h else f"- {col}: n/a")
        md.append("")
    md.append("wESS = mean CPO importance-weight ESS (stability). p_WAIC vs df_eff = posterior-dispersion "
              "check: they should agree; if p_WAIC << df_eff the chain is under-dispersed and WAIC/DIC are unreliable.")
    (OUT_DIR/"eta_criteria.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
