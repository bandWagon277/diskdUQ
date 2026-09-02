#!/usr/bin/env python
"""E1: joint (eta, sigma) sensitivity in the simplest correctly-specified setting.

Well-specified single-risk logistic discrete hazard, K=10, standardized covariates, oracle teacher.
Grid eta in {0,0.5,1} x sigma in {1,5,10,30}. 5 MALA chains per fit; R independent datasets (fixed test)
for EmpSD / repeated-sampling coverage. Reports the full metric suite PER TIME INTERVAL and overall:
Bias, RMSE, MeanPostSD, EmpSD, MeanPostSD/EmpSD, 95% coverage, width, interval score (IS_0.05) for CIF
and lambda; plus C^td, IBS, predictive deviance, CIF MAE.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from diskd.metrics import predictive_deviance, concordance_index, integrated_brier_score

K = int(os.environ.get("NUM_DURATIONS", 10)); NBETA = 3; D_COV = 3
NBETA_FIT = int(os.environ.get("NBETA_FIT", NBETA))          # < NBETA -> misspecification (E4: omitted covariate)
CONVEX = int(os.environ.get("CONVEX", 0))                    # E5: convex target pi0 L^{1-a} e^{-a Q}, a=eta/(1+eta)
TEACHER_BIAS = float(os.environ.get("TEACHER_BIAS", 0.0))    # biased oracle teacher (logit shift)
TEACHER = os.environ.get("TEACHER", "oracle")                # oracle | fitted (Experiment A: same-pop GLM teacher on D_ext)
TEACHER_N = int(os.environ.get("TEACHER_N", 10000))          # external teacher-training size N_ext
TEACHER_ADAM = int(os.environ.get("TEACHER_ADAM", 2000)); TEACHER_LR = float(os.environ.get("TEACHER_LR", 0.05))
TEACHER_PREC = float(os.environ.get("TEACHER_PREC", 1e-2)); TEACHER_SEED0 = int(os.environ.get("TEACHER_SEED0", 600000))
N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 1000)); R = int(os.environ.get("R", 20))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.5,1").split(",")]
SIGMAS = [float(s) for s in os.environ.get("SIGMAS", "1,5,10,30").split(",")]
N_CHAINS = int(os.environ.get("N_CHAINS", 5)); MALA_ITERS = int(os.environ.get("MALA_ITERS", 4000))
BURNIN = int(os.environ.get("BURNIN", 2000)); THIN = int(os.environ.get("THIN", 4)); STEP0 = float(os.environ.get("STEP", 1e-3))
SEED0 = int(os.environ.get("SEED0", 42)); FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", 999))
BETA_TRUE = np.array([0.8, -0.6, 0.5]); ALPHA_BASE = -1.7346
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_exp1")))

def sig(z): return 1.0/(1.0+np.exp(-z))
def true_lambda(x): return sig(ALPHA_BASE + x @ BETA_TRUE)[:, None]*np.ones(K)[None, :]
def true_cif(x): return 1.0 - np.cumprod(1.0 - true_lambda(x), 1)
def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D_COV)); lam = true_lambda(x)
    U = rng.random((n, K)); fires = U < lam; has = fires.any(1)
    d = np.where(has, fires.argmax(1), K-1).astype(np.int64); e = np.where(has, 1, 0).astype(np.int64)
    return x, d, e

def teacher_lam_of(x):                                       # oracle teacher (optionally biased in logit space)
    tl = true_lambda(x)
    if TEACHER_BIAS != 0.0:
        lt = np.log(tl/(1-tl)) + TEACHER_BIAS; tl = 1.0/(1.0+np.exp(-lt))
    return tl

# --- fitted same-class GLM teacher (Experiment A): trained by MLE on a fresh D_ext each replication ---
_TEACHER_CACHE = {}
def _teacher_theta(seed):                                    # well-specified 13-param GLM MAP on external data
    xe, de, ee = simulate(TEACHER_N, seed)
    xt = torch.as_tensor(xe, dtype=torch.float64); dt = torch.as_tensor(de); et = torch.as_tensor(ee)
    torch.manual_seed(seed); th = (0.3*torch.randn(K+NBETA, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.AdamW([th], lr=TEACHER_LR); m = _mask(dt)
    for _ in range(TEACHER_ADAM):
        opt.zero_grad(); lg = th[:K][None, :] + (xt[:, :NBETA] @ th[K:])[:, None]
        tgt = torch.zeros_like(lg); tgt[torch.arange(len(dt)), dt] = et.double()
        nll = (torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none")*m).sum()
        (nll + 0.5*TEACHER_PREC*(th@th)).backward(); opt.step()
    return th.detach()

def _teacher_theta_r(r):
    if r not in _TEACHER_CACHE: _TEACHER_CACHE[r] = _teacher_theta(TEACHER_SEED0 + SEED0 + r)
    return _TEACHER_CACHE[r]

def teacher_haz_for_rep(r, x):                               # teacher hazard on covariates x for replicate r
    if TEACHER != "fitted": return teacher_lam_of(x)
    th = _teacher_theta_r(r)
    with torch.no_grad():
        lg = th[:K][None, :] + (torch.as_tensor(x[:, :NBETA], dtype=torch.float64) @ th[K:])[:, None]
        return torch.sigmoid(lg).numpy()
def logits_of(th, x): return th[:K][None, :] + (x[:, :NBETA_FIT] @ th[K:])[:, None]
def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_sum(th, x, d, e):
    lg = logits_of(th, x); tgt = torch.zeros_like(lg); tgt[torch.arange(len(d)), d] = e.double()
    return (torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none")*_mask(d)).sum()
def bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def q_sum(th, x, d, e, tl): return (bkl(tl, torch.sigmoid(logits_of(th, x)))*_mask(d)).sum()

def U_fn(th, x, d, e, tl, eta, prec):
    data = nll_sum(th, x, d, e) + (eta*q_sum(th, x, d, e, tl) if eta > 0 else 0.0)
    if CONVEX: data = data / (1.0 + eta)                     # convex: (1/(1+eta))(NLL + eta Q) + prior
    return data + 0.5*prec*(th@th)

def mala(x, d, e, tl, eta, prec, seed):
    torch.manual_seed(seed); th = (0.3*torch.randn(K+NBETA_FIT, dtype=torch.float64)).requires_grad_(True)
    Uv = U_fn(th, x, d, e, tl, eta, prec); g = torch.autograd.grad(Uv, th)[0].detach(); Uv = float(Uv)
    step = STEP0; draws = []; win = 0
    for t in range(MALA_ITERS):
        prop = (th.detach()-0.5*step*g+step**0.5*torch.randn_like(th)).requires_grad_(True)
        Up = U_fn(prop, x, d, e, tl, eta, prec); gp = torch.autograd.grad(Up, prop)[0].detach(); Up = float(Up)
        lqf = -((prop.detach()-th.detach()+0.5*step*g)**2).sum()/(2*step); lqb = -((th.detach()-prop.detach()+0.5*step*gp)**2).sum()/(2*step)
        if torch.log(torch.rand(())) < (-Up+Uv)+(lqb-lqf): th = prop.detach().requires_grad_(True); Uv = Up; g = gp; win += 1
        else: th = th.detach().requires_grad_(True)
        if t < BURNIN and (t+1) % 50 == 0:
            r = win/50.0; win = 0; step *= 2.0 if r > 0.8 else 1.3 if r > 0.6 else 0.5 if r < 0.3 else 0.8 if r < 0.5 else 1.0
        if t >= BURNIN and (t-BURNIN) % THIN == 0: draws.append(th.detach().clone())
    return draws

def cif_lam(thetas, xt):
    with torch.no_grad(): lam = np.stack([torch.sigmoid(logits_of(th, xt)).numpy() for th in thetas])  # [M,Nt,K]
    return 1.0 - np.cumprod(1.0 - lam, axis=2), lam

def per_h(draws, truth):                                    # draws [M,Nt,K], truth [Nt,K]
    pm = draws.mean(0); L = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    cov = ((truth >= L) & (truth <= Uq)).mean(0)
    width = (Uq-L).mean(0); psd = draws.std(0).mean(0)
    isc = ((Uq-L) + (2/0.05)*(L-truth)*(truth < L) + (2/0.05)*(truth-Uq)*(truth > Uq)).mean(0)
    return dict(pm=pm, cov=cov, width=width, psd=psd, bias=(pm-truth).mean(0),
                rmse=np.sqrt(((pm-truth)**2).mean(0)), isc=isc)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    xt, dte, ete = simulate(TEST_N, FIXED_TEST_SEED); xtt = torch.as_tensor(xt, dtype=torch.float64)
    tcif = true_cif(xt); tlam = true_lambda(xt)
    print(f"=== E1 (eta,sigma) grid === K={K} N={N} R={R} chains={N_CHAINS} | etas={ETAS} sigmas={SIGMAS} teacher={TEACHER}"
          + (f" N_teacher={TEACHER_N}" if TEACHER == "fitted" else ""), flush=True)
    if TEACHER == "fitted":                                   # teacher quality on the fixed test set (avg over R retrained teachers)
        tdev, tibs, tcx, trmse, tr2c, tr2h = [], [], [], [], [], []
        def _r2(pred, truth): return 1.0 - ((pred-truth)**2).sum()/max(((truth-truth.mean())**2).sum(), 1e-12)
        for r in range(R):
            th = _teacher_theta_r(r)
            with torch.no_grad():
                lg = th[:K][None, :] + (torch.as_tensor(xt[:, :NBETA], dtype=torch.float64) @ th[K:])[:, None]
                tlam_te = torch.sigmoid(lg).numpy()
            tcif_te = 1.0-np.cumprod(1.0-tlam_te, 1); surv = np.cumprod(1.0-tlam_te, 1); iv = np.stack([tlam_te, 1-tlam_te], 1)
            tdev.append(predictive_deviance(iv, dte, ete, reduction="mean")); tibs.append(float(integrated_brier_score(surv, dte, ete)))
            tcx.append(float(concordance_index(dte, ete, tcif_te[:, -1]))); trmse.append(float(np.sqrt(((tcif_te-tcif)**2).mean())))
            tr2c.append(float(_r2(tcif_te, tcif))); tr2h.append(float(_r2(tlam_te, tlam)))
        print(f"    TEACHER quality on test (avg over {R} retrained teachers): R2_CIF {np.mean(tr2c):.4f} R2_haz {np.mean(tr2h):.4f} "
              f"CIF-RMSE {np.mean(trmse):.4f} dev {np.mean(tdev):.3f} IBS {np.mean(tibs):.3f} C {np.mean(tcx):.3f}", flush=True)
    rows = []
    for eta in ETAS:
        for sg in SIGMAS:
            prec = 1.0/sg**2
            pm_c, pm_l = [], []; acc = {k: [] for k in ["cov_c","w_c","psd_c","bias_c","rmse_c","is_c","cov_l","w_l","psd_l","is_l","cidx","ibs","dev","mae"]}
            hz_c, hz_l = [], []
            for r in range(R):
                x, d, e = simulate(N, SEED0+r); xtr = torch.as_tensor(x, dtype=torch.float64)
                dt = torch.as_tensor(d); et = torch.as_tensor(e); tl = torch.as_tensor(teacher_haz_for_rep(r, x), dtype=torch.float64)
                pooled = []
                for c in range(N_CHAINS): pooled += mala(xtr, dt, et, tl, eta, prec, 1000*(SEED0+r)+c)
                cif, lam = cif_lam(pooled, xtt)
                hc = per_h(cif, tcif); hl = per_h(lam, tlam); hz_c.append(hc); hz_l.append(hl)
                pm_c.append(hc["pm"]); pm_l.append(hl["pm"])
                acc["cov_c"].append(hc["cov"].mean()); acc["w_c"].append(hc["width"].mean()); acc["psd_c"].append(hc["psd"].mean())
                acc["bias_c"].append(np.abs(hc["bias"]).mean()); acc["rmse_c"].append(hc["rmse"].mean()); acc["is_c"].append(hc["isc"].mean())
                acc["cov_l"].append(hl["cov"].mean()); acc["w_l"].append(hl["width"].mean()); acc["psd_l"].append(hl["psd"].mean()); acc["is_l"].append(hl["isc"].mean())
                mlam = lam.mean(0); interval = np.stack([mlam, 1-mlam], 1); surv = np.cumprod(1-mlam, 1)
                acc["dev"].append(predictive_deviance(interval, dte, ete, reduction="mean"))
                acc["cidx"].append(float(concordance_index(dte, ete, hc["pm"][:, -1])))
                acc["ibs"].append(float(integrated_brier_score(surv, dte, ete)))
                acc["mae"].append(float(np.abs(hc["pm"]-tcif).mean()))
            empsd_c = np.stack(pm_c).std(0).mean(0); empsd_l = np.stack(pm_l).std(0).mean(0)   # per-horizon EmpSD
            mpsd_c = np.stack([h["psd"] for h in hz_c]).mean(0); mpsd_l = np.stack([h["psd"] for h in hz_l]).mean(0)
            ratio_c = mpsd_c/np.maximum(empsd_c, 1e-9)
            row = dict(eta=eta, sigma=sg, **{k: np.mean(v) for k, v in acc.items()},
                       empsd_c=empsd_c.mean(), ratio_c=ratio_c.mean(), empsd_l=empsd_l.mean())
            rows.append(row)
            # per-horizon npz (means over R)
            npz = {f"cif_{k}": np.stack([h[k] for h in hz_c]).mean(0) for k in ["cov","width","psd","bias","rmse","isc"]}
            npz.update({f"lam_{k}": np.stack([h[k] for h in hz_l]).mean(0) for k in ["cov","width","psd","isc"]})
            npz["cif_empsd"] = empsd_c; npz["cif_ratio"] = ratio_c; npz["lam_empsd"] = empsd_l
            np.savez(OUT_DIR/f"exp1_eta{eta:g}_sig{sg:g}.npz", **npz)
            print(f"  eta={eta:<4g} sig={sg:<4g} | CIF cov {row['cov_c']:.3f} w {row['w_c']:.3f} IS {row['is_c']:.3f} "
                  f"RMSE {row['rmse_c']:.4f} Post/Emp {row['ratio_c']:.2f} | dev {row['dev']:.3f} C {row['cidx']:.3f} IBS {row['ibs']:.3f}", flush=True)
    import pandas as pd; pd.DataFrame(rows).to_csv(OUT_DIR/"exp1_grid.csv", index=False)
    print(f"\nSaved to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
