#!/usr/bin/env python
"""omega (scalar spread calibration) vs last-layer Godambe/sandwich covariance correction.

Simplest well-specified single-risk logistic-hazard case, oracle teacher, at eta>0 (where the raw
generalized posterior OVER-covers). Compare three functional-space uncertainties:

  raw     : theta ~ N(theta_map, Sigma_raw),  Sigma_raw = H_eta^{-1}
  omega   : theta ~ N(theta_map, (1/w) Sigma_raw)   (w matched to sandwich mean functional variance)
  sandwich: theta ~ N(theta_map, Sigma_sand),  Sigma_sand = H_eta^{-1} V^{-1} H_eta^{-1}

with H_eta = V^{-1} + eta A + prior,  V^{-1} = Hessian(NLL) (internal info),  A = Hessian(Q_KL).
Report per-horizon 95% CIF/lambda coverage for each -> which controls the over-coverage symptom, and
does the scalar omega match the direction-aware sandwich (only if J ∝ H).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch

K = int(os.environ.get("NUM_DURATIONS", 10)); NBETA = 3; D_COV = int(os.environ.get("D_COV", 3))
NBETA_FIT = int(os.environ.get("NBETA_FIT", NBETA))          # < NBETA -> misspecification (drop covariates)
J_TYPE = os.environ.get("J_TYPE", "info")                    # info (V^{-1}=H_R) or empirical (score covariance)
N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 2000))
ETA = float(os.environ.get("ETA", 1.0)); PRIOR_PREC = float(os.environ.get("PRIOR_PREC", 1e-2))
ADAM_STEPS = int(os.environ.get("ADAM_STEPS", 2000)); ADAM_LR = float(os.environ.get("ADAM_LR", 0.05))
M = int(os.environ.get("M", 4000)); SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", 999))
BETA_TRUE = np.array([0.8, -0.6, 0.5])[:NBETA]; ALPHA_BASE = float(os.environ.get("ALPHA_BASE", -1.7346))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_omega_sandwich")))


def sig(z): return 1.0/(1.0+np.exp(-z))
def true_lambda(x): return sig(ALPHA_BASE + x[:, :NBETA] @ BETA_TRUE)[:, None] * np.ones(K)[None, :]
def true_cif(x): lam = true_lambda(x); return 1.0 - np.cumprod(1.0 - lam, axis=1)
def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D_COV)); lam = true_lambda(x)
    U = rng.random((n, K)); fires = U < lam; has = fires.any(1); first = fires.argmax(1)
    d = np.where(has, first, K-1).astype(np.int64); e = np.where(has, 1, 0).astype(np.int64)
    return x, d, e

def logits_of(theta, x):
    return theta[:K][None, :] + (x[:, :NBETA_FIT] @ theta[K:])[:, None]
def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_persubj(theta, x, d, e):
    lg = logits_of(theta, x); tgt = torch.zeros_like(lg); tgt[torch.arange(len(d)), d] = e.double()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none")
    return (bce*_mask(d)).sum(1)
def nll_sum(theta, x, d, e): return nll_persubj(theta, x, d, e).sum()
def bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def q_sum(theta, x, d, e, tlam):
    return (bkl(tlam, torch.sigmoid(logits_of(theta, x)))*_mask(d)).sum()

def cif_draws(thetas, xt):                                  # [M,P] -> [M,Nt,K]
    with torch.no_grad():
        lam = np.stack([torch.sigmoid(logits_of(th, xt)).numpy() for th in thetas])
    return 1.0 - np.cumprod(1.0 - lam, axis=2), lam

def cover(draws, truth):                                    # draws [M,Nt,K], truth [Nt,K] -> per-horizon cov [K]
    L = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    return ((truth >= L) & (truth <= Uq)).mean(0)


def run_seed(seed):
    x, d, e = simulate(N, seed); xt, _, _ = simulate(TEST_N, FIXED_TEST_SEED)
    xtr = torch.as_tensor(x, dtype=torch.float64); dt = torch.as_tensor(d); et = torch.as_tensor(e)
    xtt = torch.as_tensor(xt, dtype=torch.float64)
    tcif = true_cif(xt); tlam = true_lambda(xt); tlam_tr = torch.as_tensor(true_lambda(x), dtype=torch.float64)
    # MAP of the generalized posterior
    torch.manual_seed(seed); th = (0.3*torch.randn(K+NBETA_FIT, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.AdamW([th], lr=ADAM_LR)
    for _ in range(ADAM_STEPS):
        opt.zero_grad(); U = nll_sum(th, xtr, dt, et) + 0.5*PRIOR_PREC*(th@th)
        if ETA > 0: U = U + ETA*q_sum(th, xtr, dt, et, tlam_tr)
        U.backward(); opt.step()
    thm = th.detach()
    from torch.autograd.functional import hessian, jacobian
    H_R = hessian(lambda t: nll_sum(t, xtr, dt, et), thm).detach()                 # V^{-1} (info)
    H_Q = hessian(lambda t: q_sum(t, xtr, dt, et, tlam_tr), thm).detach()          # A (unscaled)
    p = thm.numel(); I = torch.eye(p, dtype=torch.float64)
    H_eta = H_R + ETA*H_Q + PRIOR_PREC*I                                           # posterior precision
    if J_TYPE == "empirical":                                                      # Godambe meat = score covariance
        G = jacobian(lambda t: nll_persubj(t, xtr, dt, et), thm).detach()          # [N,p]
        J = G.T @ G
    else:
        J = H_R                                                                    # well-specified: J = V^{-1}
    S_raw = torch.linalg.inv(H_eta)
    S_sand = S_raw @ J @ S_raw                                                     # H^{-1} J H^{-1}
    S_raw = 0.5*(S_raw+S_raw.T); S_sand = 0.5*(S_sand+S_sand.T)
    # shared standard normals for a paired comparison
    g = torch.Generator().manual_seed(1000+seed); z = torch.randn(M, p, generator=g, dtype=torch.float64)
    Lr = torch.linalg.cholesky(S_raw + 1e-10*I); Ls = torch.linalg.cholesky(S_sand + 1e-10*I)
    raw = [thm + Lr @ z[m] for m in range(M)]; sand = [thm + Ls @ z[m] for m in range(M)]
    cr, lr_ = cif_draws(raw, xtt); cs, ls_ = cif_draws(sand, xtt)
    # omega matched to sandwich MEAN functional variance (fairest scalar): w = mean v_post / mean v_sand
    w = float(cr.var(0).mean() / max(cs.var(0).mean(), 1e-12))
    om = [thm + (1.0/np.sqrt(w)) * (Lr @ z[m]) for m in range(M)]
    co, lo_ = cif_draws(om, xtt)
    pm = cr.mean(0)                                          # posterior-mean CIF (same center for raw/omega/sand)
    return dict(w=w, tcif_lo=float(tcif.min()), tcif_hi=float(tcif.max()),
                bias=float((pm - tcif).mean()), abias=float(np.abs(pm - tcif).mean()),
                psd=dict(raw=float(cr.std(0).mean()), omega=float(co.std(0).mean()), sand=float(cs.std(0).mean())),
                cif=dict(raw=cover(cr, tcif), omega=cover(co, tcif), sand=cover(cs, tcif)),
                lam=dict(raw=cover(lr_, tlam), omega=cover(lo_, tlam), sand=cover(ls_, tlam)))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== omega vs sandwich === eta={ETA} K={K} N={N} | linear well-spec, oracle teacher | seeds={SEEDS}", flush=True)
    res = [run_seed(s) for s in SEEDS]
    def agg(fn, key): return np.stack([r[fn][key] for r in res]).mean(0)
    w = np.mean([r["w"] for r in res])
    print(f"matched omega (mean over seeds) = {w:.2f}  [omega>1 sharpens raw to match sandwich mean variance]", flush=True)
    tlo = np.mean([r["tcif_lo"] for r in res]); thi = np.mean([r["tcif_hi"] for r in res])
    bias = np.mean([r["bias"] for r in res]); abias = np.mean([r["abias"] for r in res])
    print(f"true CIF range = [{tlo:.3f}, {thi:.3f}] | CIF bias = {bias:+.4f} (|bias| {abias:.4f}) | "
          f"PostSD: raw {np.mean([r['psd']['raw'] for r in res]):.4f} omega {np.mean([r['psd']['omega'] for r in res]):.4f} "
          f"sand {np.mean([r['psd']['sand'] for r in res]):.4f}", flush=True)
    for fn in ["cif", "lam"]:
        print(f"\n--- {fn.upper()} 95% coverage per horizon (target 0.95) ---", flush=True)
        for key in ["raw", "omega", "sand"]:
            c = agg(fn, key)
            print(f"  {key:8}: {np.round(c,3)}  | mean {c.mean():.3f}  |dev|_from_.95 {np.abs(c-0.95).mean():.3f}", flush=True)
    np.savez(OUT_DIR/f"omega_sandwich_eta{ETA:g}.npz",
             **{f"{fn}_{k}": agg(fn, k) for fn in ["cif", "lam"] for k in ["raw", "omega", "sand"]}, w=w)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
