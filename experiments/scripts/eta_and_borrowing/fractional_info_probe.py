#!/usr/bin/env python
"""E0 clean decomposition: is the eta-borrowing over-coverage the FRACTIONAL-INFORMATION geometry
(H_eta enters as eta, J_eta as eta^2) or omitted teacher uncertainty?

PERFECT teacher: teacher hazard = true hazard lambda_0  => teacher uncertainty = 0 AND teacher bias = 0.
Sweep eta = 0 -> 1. Per eta report (over R fresh training datasets, FIXED test set):
  - raw CIF coverage (posterior Sigma_raw = H_eta^{-1})   [over-covers if geometry effect]
  - sandwich CIF coverage (Sigma_sand = H_eta^{-1} J_eta H_eta^{-1})  [should be ~nominal]
  - PostSD (within-seed posterior SD of CIF) vs EmpSD (across-seed SD of the posterior mean) and the ratio
  - analytic V_post/V_samp
  - information identities tr(H_R)/tr(J_R) (~1, Bartlett for the likelihood) and tr(H_Q)/tr(J_Q)
    (the teacher factor is NOT a true likelihood -> generally != 1 -> over-coverage need not vanish at eta=1)
If over-coverage persists with the perfect teacher, it is the pseudo-likelihood geometry, not teacher UQ.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch

K = int(os.environ.get("NUM_DURATIONS", 10)); NBETA = 3; D_COV = 3
N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 2000))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.25,0.5,0.75,1").split(",")]
PRIOR_PREC = float(os.environ.get("PRIOR_PREC", 1e-2)); ADAM_STEPS = int(os.environ.get("ADAM_STEPS", 2000))
ADAM_LR = float(os.environ.get("ADAM_LR", 0.05)); M = int(os.environ.get("M", 3000))
R = int(os.environ.get("R", 20)); SEED0 = int(os.environ.get("SEED0", 42)); FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", 999))
TEACHER = os.environ.get("TEACHER", "oracle")               # oracle | fitted (E6A: retrained per replicate)
TEACHER_N = int(os.environ.get("TEACHER_N", 2000)); KAPPA = int(os.environ.get("KAPPA", 0)); KBOOT = int(os.environ.get("KBOOT", 12))
PERHZ = int(os.environ.get("PERHZ", 0))                     # E3: save per-horizon raw/sandwich decomposition
BETA_TRUE = np.array([0.8, -0.6, 0.5]); ALPHA_BASE = -1.7346
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_fractional")))

def sig(z): return 1.0/(1.0+np.exp(-z))
def true_lambda(x): return sig(ALPHA_BASE + x @ BETA_TRUE)[:, None]*np.ones(K)[None, :]
def true_cif(x): return 1.0 - np.cumprod(1.0 - true_lambda(x), 1)
def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D_COV)); lam = true_lambda(x)
    U = rng.random((n, K)); fires = U < lam; has = fires.any(1)
    d = np.where(has, fires.argmax(1), K-1).astype(np.int64); e = np.where(has, 1, 0).astype(np.int64)
    return x, d, e

def logits_of(th, x): return th[:K][None, :] + (x @ th[K:])[:, None]
def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_ps(th, x, d, e):
    lg = logits_of(th, x); tgt = torch.zeros_like(lg); tgt[torch.arange(len(d)), d] = e.double()
    return (torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none")*_mask(d)).sum(1)
def bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def q_ps(th, x, d, e, tl, kap=None):
    w = bkl(tl, torch.sigmoid(logits_of(th, x)))*_mask(d)
    if kap is not None: w = w*kap
    return w.sum(1)

def fit_teacher(seed):
    xe, de, ee = simulate(TEACHER_N, seed); xt = torch.as_tensor(xe, dtype=torch.float64); dt = torch.as_tensor(de); et = torch.as_tensor(ee)
    torch.manual_seed(seed); th = (0.3*torch.randn(K+NBETA, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.AdamW([th], lr=ADAM_LR)
    for _ in range(ADAM_STEPS): opt.zero_grad(); (nll_ps(th, xt, dt, et).sum()+0.5*PRIOR_PREC*(th@th)).backward(); opt.step()
    thm = th.detach()
    return lambda x: torch.sigmoid(logits_of(thm, torch.as_tensor(x, dtype=torch.float64))).detach().numpy()

def build_teacher(seed, x):                                  # returns (teacher_lam [N,K], kappa [N,K] or None)
    if TEACHER == "oracle": return true_lambda(x), None
    tl = fit_teacher(600000+seed)(x)
    if not KAPPA: return tl, None
    preds = np.stack([fit_teacher(600000+seed + 7919*(b+1))(x) for b in range(KBOOT)])   # bootstrap teacher variance
    v = preds.var(0) + 1e-6; q = np.clip(tl, 1e-4, 1-1e-4); kap = np.clip(q*(1-q)/v - 1.0, 0.1, 1e4)
    return tl, kap/kap.mean()                                # normalized so eta interpretation stays stable

def cif_of(thetas, xt):
    with torch.no_grad():
        lam = np.stack([torch.sigmoid(logits_of(th, xt)).numpy() for th in thetas])
    return 1.0 - np.cumprod(1.0 - lam, axis=2)
def cover(dr, tr): L = np.quantile(dr, .025, 0); Uq = np.quantile(dr, .975, 0); return float(((tr >= L) & (tr <= Uq)).mean())
def cover_h(dr, tr): L = np.quantile(dr, .025, 0); Uq = np.quantile(dr, .975, 0); return ((tr >= L) & (tr <= Uq)).mean(0)   # [K]
def width_h(dr): return (np.quantile(dr, .975, 0)-np.quantile(dr, .025, 0)).mean(0)

def run_eta(eta, xtt, tcif):
    from torch.autograd.functional import hessian, jacobian
    pms, psds, covr, covs, vpv, hrjr, hqjq = [], [], [], [], [], [], []
    covrh, covsh, wrh, wsh, hq_l, jq_l = [], [], [], [], [], []
    for r in range(R):
        x, d, e = simulate(N, SEED0 + r); xtr = torch.as_tensor(x, dtype=torch.float64)
        dt = torch.as_tensor(d); et = torch.as_tensor(e)
        tl_np, kap_np = build_teacher(SEED0+r, x)
        tl = torch.as_tensor(tl_np, dtype=torch.float64); kap = torch.as_tensor(kap_np, dtype=torch.float64) if kap_np is not None else None
        torch.manual_seed(SEED0+r); th = (0.3*torch.randn(K+NBETA, dtype=torch.float64)).requires_grad_(True)
        opt = torch.optim.AdamW([th], lr=ADAM_LR)
        for _ in range(ADAM_STEPS):
            opt.zero_grad(); Uv = nll_ps(th, xtr, dt, et).sum() + 0.5*PRIOR_PREC*(th@th)
            if eta > 0: Uv = Uv + eta*q_ps(th, xtr, dt, et, tl, kap).sum()
            Uv.backward(); opt.step()
        thm = th.detach(); p = thm.numel(); I = torch.eye(p, dtype=torch.float64)
        H_R = hessian(lambda t: nll_ps(t, xtr, dt, et).sum(), thm).detach()
        H_Q = hessian(lambda t: q_ps(t, xtr, dt, et, tl, kap).sum(), thm).detach()
        H_eta = H_R + eta*H_Q + PRIOR_PREC*I
        Gn = jacobian(lambda t: nll_ps(t, xtr, dt, et), thm).detach()          # [N,p]
        Gq = jacobian(lambda t: q_ps(t, xtr, dt, et, tl, kap), thm).detach()
        J_R = Gn.T@Gn; J_Q = Gq.T@Gq; Ge = Gn + eta*Gq; J_eta = Ge.T@Ge         # full meat
        S_raw = torch.linalg.inv(H_eta); S_sand = S_raw@J_eta@S_raw
        S_raw = 0.5*(S_raw+S_raw.T); S_sand = 0.5*(S_sand+S_sand.T)
        g = torch.Generator().manual_seed(7000+r); z = torch.randn(M, p, generator=g, dtype=torch.float64)
        Lr = torch.linalg.cholesky(S_raw+1e-10*I); Ls = torch.linalg.cholesky(S_sand+1e-10*I)
        dr = cif_of([thm+Lr@z[m] for m in range(M)], xtt); ds = cif_of([thm+Ls@z[m] for m in range(M)], xtt)
        pms.append(dr.mean(0)); psds.append(dr.std(0).mean()); covr.append(cover(dr, tcif)); covs.append(cover(ds, tcif))
        vpv.append(dr.var(0).mean()/max(ds.var(0).mean(), 1e-12))
        hrjr.append(float(torch.trace(H_R)/torch.trace(J_R)))
        hqjq.append(float(torch.trace(H_Q)/max(float(torch.trace(J_Q)), 1e-9)))
        hq_l.append(float(torch.trace(H_Q))); jq_l.append(float(torch.trace(J_Q)))
        if PERHZ:
            covrh.append(cover_h(dr, tcif)); covsh.append(cover_h(ds, tcif)); wrh.append(width_h(dr)); wsh.append(width_h(ds))
    empsd = np.stack(pms).std(0).mean()
    out = dict(cov_raw=np.mean(covr), cov_sand=np.mean(covs), postsd=np.mean(psds), empsd=empsd,
               ratio=np.mean(psds)/max(empsd, 1e-12), vpv=np.mean(vpv), hrjr=np.mean(hrjr), hqjq=np.mean(hqjq),
               hq=np.mean(hq_l), jq=np.mean(jq_l))
    if PERHZ:
        pm_arr = np.stack(pms)                                # [R,Nt,K]
        out["perhz"] = dict(cov_raw=np.stack(covrh).mean(0), cov_sand=np.stack(covsh).mean(0),
                            width_raw=np.stack(wrh).mean(0), width_sand=np.stack(wsh).mean(0),
                            empsd=pm_arr.std(0).mean(0), bias=(pm_arr.mean(0)-tcif).mean(0))
    return out

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    xt, _, _ = simulate(TEST_N, FIXED_TEST_SEED); xtt = torch.as_tensor(xt, dtype=torch.float64); tcif = true_cif(xt)
    tag = f"teacher={TEACHER}" + (f" N_teacher={TEACHER_N}" if TEACHER != "oracle" else "") + (" KAPPA" if KAPPA else "")
    print(f"=== fractional-information ({tag}) === K={K} N={N} R={R} etas={ETAS}", flush=True)
    print(f"{'eta':>5} {'cov_raw':>8} {'cov_sand':>9} {'PostSD':>7} {'EmpSD':>7} {'Post/Emp':>8} {'Vpost/Vsamp':>11} {'trHR/trJR':>10} {'trHQ/trJQ':>10} {'trH_Q':>8} {'trJ_Q':>8}", flush=True)
    rows = []
    for eta in ETAS:
        m = run_eta(eta, xtt, tcif); rows.append(dict(eta=eta, **{k: v for k, v in m.items() if k != "perhz"}))
        print(f"{eta:>5g} {m['cov_raw']:>8.3f} {m['cov_sand']:>9.3f} {m['postsd']:>7.4f} {m['empsd']:>7.4f} "
              f"{m['ratio']:>8.2f} {m['vpv']:>11.2f} {m['hrjr']:>10.2f} {m['hqjq']:>10.2f} {m['hq']:>8.2f} {m['jq']:>8.4f}", flush=True)
        if PERHZ and "perhz" in m:
            np.savez(OUT_DIR/f"perhz_eta{eta:g}.npz", **m["perhz"])
    import pandas as pd; pd.DataFrame(rows).to_csv(OUT_DIR/f"fractional_{TEACHER}{'_kappa' if KAPPA else ''}.csv", index=False)
    print(f"\nSaved to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
