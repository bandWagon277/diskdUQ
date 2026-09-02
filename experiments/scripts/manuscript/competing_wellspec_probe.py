#!/usr/bin/env python
"""Competing-risk well-specified control with teacher types (overall->CR, CR->CR).

DGP  : discrete competing hazards, logit_{j,k}(x) = alpha_{j,k} + x.beta_j (J=2 causes, 3 betas each),
       interval probs = softmax([logit_1, logit_2, 0]); intrinsic intervals, admin censoring.
Model: the SAME form (alpha_{j,k} baselines + beta_j), sampled by MALA from random init.
Teacher (eta=1): oracle predictions from the true model.
  * CR->CR   : KL over the full (J+1)-way interval distribution.
  * overall  : KL on the OVERALL event hazard only (sum of cause hazards vs teacher's overall).
Reports, for eta=0 AND eta=1, both CIF and lambda: coverage vs N (Question A) and per-horizon
credible-interval width (Question B).  omega=1 fixed.  Env: TEACHER={none|cr_cr|overall},
ETA, N, NUM_DURATIONS(K), SEEDS, FIXED_TEST_SEED, MALA_ITERS/BURNIN/THIN/N_CHAINS/STEP, PRIOR_SIGMA,
COV_SCALE (heterogeneity knob), NBETA_FIT (misspecification: fit fewer betas).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch

from diskd.losses import CompetingRiskNLLLoss
from diskd.utils import competing_interval_probs, competing_cif, make_at_risk_mask, temperature_scale_probs
from diskd.uncertainty import gelman_rubin_rhat, effective_sample_size

DEVICE = "cpu"; J = 2
D_COV = int(os.environ.get("D_COV", 3)); NBETA_TRUE = 3
NBETA_FIT = int(os.environ.get("NBETA_FIT", NBETA_TRUE))          # < NBETA_TRUE -> misspecification
K = int(os.environ.get("NUM_DURATIONS", 10)); N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 400))
COV_SCALE = float(os.environ.get("COV_SCALE", 1.0))              # heterogeneity: covariate spread
ALPHA_BASE = float(os.environ.get("ALPHA_BASE", -2.2))          # per-cause baseline logit (~ hazard 0.1 each)
BETA_TRUE = np.array([[0.8, -0.6, 0.5], [-0.5, 0.7, 0.4]])[:, :NBETA_TRUE]   # [J, NBETA_TRUE]
ETA = float(os.environ.get("ETA", 0.0)); OMEGA = float(os.environ.get("OMEGA", 1.0))
TEACHER = os.environ.get("TEACHER", "none")                     # none | cr_cr | overall
MALA_ITERS = int(os.environ.get("MALA_ITERS", 8000)); BURNIN = int(os.environ.get("BURNIN", 4000))
THIN = int(os.environ.get("THIN", 4)); N_CHAINS = int(os.environ.get("N_CHAINS", 4))
STEP0 = float(os.environ.get("STEP", 1e-3))
_PS = os.environ.get("PRIOR_SIGMA", "10"); PRIOR_PREC = 0.0 if _PS.lower() in ("flat","inf") else 1.0/float(_PS)**2
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", -1))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_competing")))


def alpha_true(): return ALPHA_BASE + np.zeros((J, K))
def true_logits_np(x):                                          # [N,J,K]
    lp = x[:, :NBETA_TRUE] @ BETA_TRUE.T                        # [N,J]
    return alpha_true()[None, :, :] + lp[:, :, None]
def true_probs_np(x):                                          # [N,J+1,K] softmax with appended zero no-event
    lg = true_logits_np(x); z = np.concatenate([lg, np.zeros((lg.shape[0], 1, K))], axis=1)
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)

def simulate(n, seed):
    rng = np.random.default_rng(seed); x = COV_SCALE*rng.normal(size=(n, D_COV)); p = true_probs_np(x)  # [n,J+1,K]
    d = np.full(n, K-1, dtype=np.int64); ev = np.zeros(n, dtype=np.int64)
    U = rng.random((n, K)); cum = np.cumsum(p, axis=1)         # [n,J+1,K] cumulative over channels
    done = np.zeros(n, dtype=bool)
    for k in range(K):
        out = (U[:, k][:, None] < cum[:, :, k]).argmax(1)       # 0..J-1 = cause j+1, J = no-event
        fire = (~done) & (out < J)
        d[fire] = k; ev[fire] = out[fire] + 1; done |= fire
    return x, d, ev


def logits_of(theta, x):                                       # theta = [alpha(J*K), beta(J*NBETA_FIT)]
    a = theta[:J*K].view(J, K); b = theta[J*K:].view(J, NBETA_FIT)
    return a[None, :, :] + torch.einsum("nd,jd->nj", x[:, :NBETA_FIT], b)[:, :, None]   # [N,J,K]

_nll = CompetingRiskNLLLoss()
def nll_sum(theta, x, d, e):
    return _nll(logits_of(theta, x), d, e, reduction="sum")

def bern_kl(a, b):
    a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6)
    return a*torch.log(a/b) + (1-a)*torch.log((1-a)/(1-b))

def q_sum(theta, x, d, e, teacher_full):                       # teacher_full: [N,J+1,K]
    student = competing_interval_probs(logits_of(theta, x))    # [N,J+1,K]
    mask = make_at_risk_mask(d, K).double()
    if TEACHER == "cr_cr":
        tt = temperature_scale_probs(teacher_full, 1.0); st = temperature_scale_probs(student, 1.0)
        kl = (tt*(tt.clamp_min(1e-9).log() - st.clamp_min(1e-9).log())).sum(1)          # [N,K]
        return (kl*mask).sum()
    # overall->CR: match overall event hazard (1 - no-event prob)
    t_ov = 1.0 - teacher_full[:, J, :]; s_ov = 1.0 - student[:, J, :]
    return (bern_kl(t_ov, s_ov)*mask).sum()

def U_fn(theta, x, d, e, teacher_full):
    base = nll_sum(theta, x, d, e)
    if ETA > 0 and teacher_full is not None:
        base = base + ETA*q_sum(theta, x, d, e, teacher_full)
    return OMEGA*base + 0.5*PRIOR_PREC*(theta @ theta)


def mala(x, d, e, theta0, seed, teacher_full):
    torch.manual_seed(seed); th = theta0.clone().requires_grad_(True)
    Uv = U_fn(th, x, d, e, teacher_full); g = torch.autograd.grad(Uv, th)[0].detach(); Uv = float(Uv)
    step = STEP0; draws = []; acc = 0; win = 0
    for t in range(MALA_ITERS):
        prop = (th.detach() - 0.5*step*g + step**0.5*torch.randn_like(th)).requires_grad_(True)
        Up = U_fn(prop, x, d, e, teacher_full); gp = torch.autograd.grad(Up, prop)[0].detach(); Up = float(Up)
        lqf = -((prop.detach()-th.detach()+0.5*step*g)**2).sum()/(2*step)
        lqb = -((th.detach()-prop.detach()+0.5*step*gp)**2).sum()/(2*step)
        if torch.log(torch.rand(())) < (-Up+Uv)+(lqb-lqf):
            th = prop.detach().requires_grad_(True); Uv = Up; g = gp; acc += 1; win += 1
        else:
            th = th.detach().requires_grad_(True)
        if t < BURNIN and (t+1) % 50 == 0:
            r = win/50.0; win = 0
            step *= 2.0 if r > 0.8 else 1.3 if r > 0.6 else 0.5 if r < 0.3 else 0.8 if r < 0.5 else 1.0
        if t >= BURNIN and (t-BURNIN) % THIN == 0: draws.append(th.detach().clone())
    return draws, acc/MALA_ITERS


def rhat_ess(A):
    Af = A.reshape(A.shape[0], A.shape[1], -1); rr, ee = [], []
    for j in range(Af.shape[2]):
        try: rr.append(float(gelman_rubin_rhat(Af[:, :, j]))); ee.append(float(effective_sample_size(Af[:, :, j])))
        except Exception: pass
    return (float(np.nanmax(rr)) if rr else float("nan")), (float(np.nanmin(ee)) if ee else float("nan"))

def per_h(draws, truth):                                        # draws [M,N,J,K], truth [N,J,K]; reduce over N,J -> K
    pm = draws.mean(0); L = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    red = (0, 1)
    cov = ((truth >= L) & (truth <= Uq)).mean(red)
    return dict(cov=cov, width=(Uq-L).mean(red), level=truth.mean(red),
                r2=1 - ((pm-truth)**2).sum(red)/np.maximum(((truth-truth.mean(red,keepdims=True))**2).sum(red),1e-12))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    P = J*K + J*NBETA_FIT
    print(f"=== Competing-risk well-spec === teacher={TEACHER} eta={ETA} omega={OMEGA} | K={K} N={N} "
          f"cov_scale={COV_SCALE} fit_betas={NBETA_FIT}/{NBETA_TRUE} | {N_CHAINS}x{MALA_ITERS} MALA random-init | seeds={SEEDS}", flush=True)
    A = {k: [] for k in ["cif_cov","cif_w","haz_cov","haz_w","cif_r2","haz_r2","cif_rhat","haz_rhat"]}; hz = []
    for seed in SEEDS:
        x, d, e = simulate(N, 10*seed+2)
        xt, _, _ = simulate(TEST_N, FIXED_TEST_SEED if FIXED_TEST_SEED >= 0 else 10*seed+3)
        xtt = torch.as_tensor(xt, dtype=torch.float64)
        tprob = torch.as_tensor(true_probs_np(xt), dtype=torch.float64)          # [Nt,J+1,K]
        tcif = competing_cif(tprob).numpy(); tlam = tprob[:, :J, :].numpy()       # [Nt,J,K]
        xtr = torch.as_tensor(x, dtype=torch.float64); dt = torch.as_tensor(d); et = torch.as_tensor(e)
        teacher_full = torch.as_tensor(true_probs_np(x), dtype=torch.float64) if TEACHER != "none" else None
        cif_ps, haz_ps, cif_cm, haz_cm, accs = [], [], [], [], []
        for c in range(N_CHAINS):
            torch.manual_seed(1000*seed+c); theta0 = 0.3*torch.randn(P, dtype=torch.float64)
            draws, acc = mala(xtr, dt, et, theta0, 7*seed+c, teacher_full); accs.append(acc)
            probs = [competing_interval_probs(logits_of(th, xtt)) for th in draws]
            cif = np.stack([competing_cif(pr).numpy() for pr in probs])           # [nd,Nt,J,K]
            lam = np.stack([pr[:, :J, :].numpy() for pr in probs])
            cif_ps.append(cif); haz_ps.append(lam); cif_cm.append(cif.mean((1,2))); haz_cm.append(lam.mean((1,2)))
        pc = np.concatenate(cif_ps, 0); ph = np.concatenate(haz_ps, 0)
        rc, _ = rhat_ess(np.stack(cif_cm)); rh, _ = rhat_ess(np.stack(haz_cm))
        hc = per_h(pc, tcif); hh = per_h(ph, tlam); hz.append(dict(cif=hc, haz=hh))
        A["cif_cov"].append(hc["cov"].mean()); A["cif_w"].append(hc["width"].mean()); A["cif_r2"].append(hc["r2"].mean()); A["cif_rhat"].append(rc)
        A["haz_cov"].append(hh["cov"].mean()); A["haz_w"].append(hh["width"].mean()); A["haz_r2"].append(hh["r2"].mean()); A["haz_rhat"].append(rh)
        print(f"  seed{seed}: acc={np.mean(accs):.2f} | CIF R-hat={rc:.3f} R2={hc['r2'].mean():.3f} cov={hc['cov'].mean():.3f} w={hc['width'].mean():.3f} | "
              f"lam R-hat={rh:.3f} R2={hh['r2'].mean():.3f} cov={hh['cov'].mean():.3f}", flush=True)
    def stk(fn, key): return np.stack([h[fn][key] for h in hz]).mean(0)
    print(f"=== teacher={TEACHER} eta={ETA} K={K} N={N} cov_scale={COV_SCALE} fit={NBETA_FIT}: "
          f"CIF R-hat {np.mean(A['cif_rhat']):.3f} R2 {np.mean(A['cif_r2']):.3f} cov {np.mean(A['cif_cov']):.3f} w {np.mean(A['cif_w']):.3f} | "
          f"lam R2 {np.mean(A['haz_r2']):.3f} cov {np.mean(A['haz_cov']):.3f} w {np.mean(A['haz_w']):.3f} ===", flush=True)
    npz = {"cuts": np.arange(1, K+1)}
    for fn in ["cif", "haz"]:
        for key in ["cov", "width", "level", "r2"]: npz[f"{fn}_{key}"] = stk(fn, key)
    np.savez(OUT_DIR/f"cr_T{TEACHER}_eta{ETA:g}_K{K}_N{N}_cov{COV_SCALE:g}_fit{NBETA_FIT}.npz", **npz)
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
