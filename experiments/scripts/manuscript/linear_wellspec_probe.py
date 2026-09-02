#!/usr/bin/env python
"""Well-specified discrete-time logistic-hazard control.

Motivation (user, 2026-08-13): the squared-rate DGP + MLP + KM-grid/censoring
boundary may simply be too hard / misspecified. Strip all of that away:

  DGP  : single-risk discrete hazard  logit lambda_k(x) = alpha_k + x . beta_true
         (3 covariates, 3 true betas, intrinsic intervals -> at-risk mass at EVERY
         horizon, NO KM-quantile grid, NO continuous-time censoring boundary).
  Model: the SAME parameterization -- K baseline logits alpha_k + a shared p-dim
         beta.  WELL-SPECIFIED when p=3 (just estimate the betas); MISSPECIFIED
         when p=1 (drops two covariates).  Sampled by MALA from random init.

Reports per-horizon R^2 (fit quality) and 95%-CI coverage, for CIF and lambda.
Check R^2 first; if good, read coverage.  Env: SPEC={well|miss}, K, N, N_CHAINS,
MALA_ITERS, BURNIN, THIN, STEP, PRIOR_PREC, SEEDS, FIXED_TEST_SEED, TEST_N.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from diskd.uncertainty import gelman_rubin_rhat, effective_sample_size
from diskd.metrics import predictive_deviance, concordance_index, integrated_brier_score

DEVICE = "cpu"
D_COV = int(os.environ.get("D_COV", 3)); NBETA_TRUE = 3
SPEC = os.environ.get("SPEC", "well"); NBETA_FIT = NBETA_TRUE if SPEC == "well" else int(os.environ.get("NBETA_FIT", 1))
K = int(os.environ.get("NUM_DURATIONS", 10)); N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 400))
BETA_TRUE = np.array([float(b) for b in os.environ.get("BETA_TRUE", "0.8,-0.6,0.5").split(",")])[:NBETA_TRUE]
ALPHA_BASE = float(os.environ.get("ALPHA_BASE", -1.7346))        # logit(0.15) baseline hazard
ALPHA_TREND = float(os.environ.get("ALPHA_TREND", 0.0))          # per-interval drift in baseline logit
MALA_ITERS = int(os.environ.get("MALA_ITERS", 8000)); BURNIN = int(os.environ.get("BURNIN", 4000))
THIN = int(os.environ.get("THIN", 4)); N_CHAINS = int(os.environ.get("N_CHAINS", 4))
STEP0 = float(os.environ.get("STEP", 1e-3))
# Gaussian structural prior pi0=N(0, sigma^2 I): grad = theta/sigma^2 (ridge). "flat"/inf -> pi0 ∝ 1.
_PS = os.environ.get("PRIOR_SIGMA", os.environ.get("PRIOR_PREC_SIGMA", "10"))
if _PS.lower() in ("flat", "inf", "none"):
    PRIOR_PREC = 0.0; PRIOR_SIGMA_STR = "flat"
else:
    _sig = float(_PS); PRIOR_PREC = 1.0/_sig**2; PRIOR_SIGMA_STR = f"{_sig:g}"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", -1))
# HIERARCHICAL prior: sigma^2 ~ InvGamma(A0,B0), Gibbs-within-MALA -> sigma inferred from data (auto shrinkage).
HIER = int(os.environ.get("HIER", 0)); IG_A0 = float(os.environ.get("IG_A0", 1.0)); IG_B0 = float(os.environ.get("IG_B0", 1.0))
GIBBS_EVERY = int(os.environ.get("GIBBS_EVERY", 10))
# starting point for the chains: "random" (independent random init) or "adam" (AdamW MAP + small perturbation)
INIT = os.environ.get("INIT", "random"); ADAM_STEPS = int(os.environ.get("ADAM_STEPS", 800))
ADAM_LR = float(os.environ.get("ADAM_LR", 0.05)); ADAM_PERTURB = float(os.environ.get("ADAM_PERTURB", 0.1))
# Rung 1: teacher-guided generalized posterior  exp{-omega[NLL + eta*Q]},  Q = sum Y_ik BernKL(teacher||student).
ETA = float(os.environ.get("ETA", 0.0)); OMEGA = float(os.environ.get("OMEGA", 1.0))
TEACHER = os.environ.get("TEACHER", "none")           # none | oracle (true lambda) | fitted (larger cohort)
TEACHER_N = int(os.environ.get("TEACHER_N", 5000)); TEACHER_BIAS = float(os.environ.get("TEACHER_BIAS", 0.0))
CRITERIA = int(os.environ.get("CRITERIA", 0))          # 1 -> compute LPML / WAIC / DIC from the MALA draws
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_linear_wellspec"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def alpha_true(): return ALPHA_BASE + ALPHA_TREND * np.arange(K)

def true_lambda(x):                                              # [N,K] discrete hazard
    return sigmoid(alpha_true()[None, :] + (x[:, :NBETA_TRUE] @ BETA_TRUE)[:, None])
def true_cif(x):
    lam = true_lambda(x); return 1.0 - np.cumprod(1.0 - lam, axis=1)

def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D_COV)); lam = true_lambda(x)
    U = rng.random((n, K)); fires = U < lam
    has = fires.any(1); first = fires.argmax(1)
    d = np.where(has, first, K-1).astype(np.int64); e = np.where(has, 1, 0).astype(np.int64)   # admin-censor at K
    return x, d, e


def logits_of(theta, x):                                        # theta = [alpha(K), beta(NBETA_FIT)]
    alpha = theta[:K]; beta = theta[K:]
    lp = x[:, :NBETA_FIT] @ beta
    return alpha[None, :] + lp[:, None]                         # [N,K]

def nll_sum(theta, x, d, e):
    lg = logits_of(theta, x)                                    # [N,K]
    k = torch.arange(K, device=x.device)
    at_risk = (k[None, :] <= d[:, None]).double()               # rows k<=d_i are at risk
    target = torch.zeros_like(lg); ev = e.double()
    target[torch.arange(len(d)), d] = ev                        # event indicator at observed interval
    bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, target, reduction="none")
    return (bce * at_risk).sum()

def bern_kl(a, b):                                     # KL(Bernoulli(a) || Bernoulli(b)), elementwise
    a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6)
    return a*torch.log(a/b) + (1-a)*torch.log((1-a)/(1-b))

def q_sum(theta, x, d, e, teacher_lam):                # sum_i sum_k Y_ik BernKL(teacher || student)
    lam = torch.sigmoid(logits_of(theta, x))
    k = torch.arange(K, device=x.device); at_risk = (k[None, :] <= d[:, None]).double()
    return (bern_kl(teacher_lam, lam) * at_risk).sum()

def nll_persubj(theta, x, d, e):                       # per-subject single-risk NLL [N]
    lg = logits_of(theta, x); k = torch.arange(K, device=x.device); at_risk = (k[None, :] <= d[:, None]).double()
    target = torch.zeros_like(lg); target[torch.arange(len(d)), d] = e.double()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, target, reduction="none")
    return (bce * at_risk).sum(1)

def q_persubj(theta, x, d, e, teacher_lam):            # per-subject teacher KL Q_i [N]
    lam = torch.sigmoid(logits_of(theta, x)); k = torch.arange(K, device=x.device); at_risk = (k[None, :] <= d[:, None]).double()
    return (bern_kl(teacher_lam, lam) * at_risk).sum(1)

def _lme(a, axis):                                     # log-mean-exp
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)

def criteria(all_th, x, d, e, teacher_lam):
    """WAIC, DIC, LPML from pooled MALA draws (per-subject student NLL; generalized-posterior CPO for LPML)."""
    NLL, Q = [], []
    with torch.no_grad():
        for thv in all_th:
            th = torch.as_tensor(thv, dtype=torch.float64)
            NLL.append(nll_persubj(th, x, d, e).cpu().numpy())
            Q.append(q_persubj(th, x, d, e, teacher_lam).cpu().numpy() if teacher_lam is not None else np.zeros(x.shape[0]))
    NLL = np.stack(NLL); Q = np.stack(Q)               # [M,N]
    lppd = _lme(-NLL, 0).sum(); p_waic = NLL.var(0, ddof=1).sum(); waic = -2*(lppd - p_waic)
    Dbar = 2*NLL.sum(1).mean()
    with torch.no_grad():
        Dhat = 2*float(nll_persubj(torch.as_tensor(all_th.mean(0), dtype=torch.float64), x, d, e).sum())
    dic = Dbar + (Dbar - Dhat)                          # = Dbar + p_DIC
    a = ETA*Q; lpml = float((_lme(a, 0) - _lme(a + NLL, 0)).sum())   # CPO_i = E[e^{ηQ}]/E[e^{ηQ}/L_i]
    return dict(waic=float(waic), dic=float(dic), lpml=lpml, p_waic=float(p_waic))

def U_fn(theta, x, d, e, prec, teacher_lam=None):
    """Generalized posterior potential: -log Pi = omega[NLL + eta*Q] + 0.5*prec*||theta||^2 (prior NOT scaled by omega)."""
    base = nll_sum(theta, x, d, e)
    if ETA > 0 and teacher_lam is not None:
        base = base + ETA * q_sum(theta, x, d, e, teacher_lam)
    return OMEGA * base + 0.5 * prec * (theta @ theta)


def adam_map(x, d, e, seed, teacher_lam=None):
    """AdamW MAP of the (generalized) posterior -> the mode a warm-start would use."""
    torch.manual_seed(seed); th = (0.3*torch.randn(K+NBETA_FIT, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.AdamW([th], lr=ADAM_LR, weight_decay=0.0)
    prec = PRIOR_PREC if not HIER else 1.0
    for _ in range(ADAM_STEPS):
        opt.zero_grad(); U_fn(th, x, d, e, prec, teacher_lam).backward(); opt.step()
    return th.detach()


def mala(x, d, e, theta0, seed, teacher_lam=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    th = theta0.clone().requires_grad_(True); prec = PRIOR_PREC if not HIER else 1.0   # HIER inits sigma=1
    Uv = U_fn(th, x, d, e, prec, teacher_lam); g = torch.autograd.grad(Uv, th)[0].detach(); Uv = float(Uv)
    step = STEP0; draws = []; sig_tr = []; acc = 0; win = 0; p = th.numel()
    for t in range(MALA_ITERS):
        prop = (th.detach() - 0.5*step*g + step**0.5*torch.randn_like(th)).requires_grad_(True)
        Up = U_fn(prop, x, d, e, prec, teacher_lam); gp = torch.autograd.grad(Up, prop)[0].detach(); Up = float(Up)
        lq_f = -((prop.detach() - th.detach() + 0.5*step*g)**2).sum()/(2*step)
        lq_b = -((th.detach() - prop.detach() + 0.5*step*gp)**2).sum()/(2*step)
        if torch.log(torch.rand(())) < (-Up + Uv) + (lq_b - lq_f):
            th = prop.detach().requires_grad_(True); Uv = Up; g = gp; acc += 1; win += 1
        else:
            th = th.detach().requires_grad_(True)
        if HIER and (t+1) % GIBBS_EVERY == 0:                    # Gibbs: sigma^2 | theta ~ InvGamma(A0+p/2, B0+||theta||^2/2)
            s2 = 1.0 / rng.gamma(IG_A0 + p/2.0, 1.0/(IG_B0 + 0.5*float(th.detach() @ th.detach())))
            prec = 1.0/s2
            thl = th.detach().requires_grad_(True)              # recompute (Uv,g) under the new prec for the next step
            Uv = float(U_fn(thl, x, d, e, prec, teacher_lam)); g = torch.autograd.grad(U_fn(thl, x, d, e, prec, teacher_lam), thl)[0].detach(); th = thl
            if t >= BURNIN: sig_tr.append(s2**0.5)
        if t < BURNIN and (t+1) % 50 == 0:
            r = win/50.0; win = 0
            step *= 2.0 if r > 0.8 else 1.3 if r > 0.6 else 0.5 if r < 0.3 else 0.8 if r < 0.5 else 1.0
        if t >= BURNIN and (t-BURNIN) % THIN == 0: draws.append(th.detach().clone())
    return draws, acc/MALA_ITERS, (float(np.mean(sig_tr)) if sig_tr else float("nan"))


def rhat_ess(A):     # A: [C, ndraw, K] cohort-mean functional -> worst R-hat, min ESS over horizons
    rr, ee = [], []
    for j in range(A.shape[2]):
        try: rr.append(float(gelman_rubin_rhat(A[:, :, j]))); ee.append(float(effective_sample_size(A[:, :, j])))
        except Exception: pass
    return (float(np.nanmax(rr)) if rr else float("nan")), (float(np.nanmin(ee)) if ee else float("nan"))

def per_horizon(draws_np, truth):                               # draws [M,N,K], truth [N,K]
    pm = draws_np.mean(0); L = np.quantile(draws_np, .025, 0); Uq = np.quantile(draws_np, .975, 0)
    ss_res = ((pm-truth)**2).sum(0); ss_tot = ((truth-truth.mean(0, keepdims=True))**2).sum(0)
    r2 = 1.0 - ss_res/np.maximum(ss_tot, 1e-12)
    cov = ((truth >= L) & (truth <= Uq)).mean(0)
    return dict(r2=r2, cov=cov, bias=(pm-truth).mean(0), post_sd=draws_np.std(0).mean(0),
                rmse=np.sqrt(((pm-truth)**2).mean(0)), width=(Uq-L).mean(0),
                truth_range=(truth.max(0)-truth.min(0)), truth_mean=truth.mean(0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    print(f"=== Well-specified linear control === SPEC={SPEC} (fit {NBETA_FIT} of {NBETA_TRUE} betas) "
          f"K={K} N={N} sigma={PRIOR_SIGMA_STR} init={INIT} | teacher={TEACHER} eta={ETA} omega={OMEGA} T=1 | "
          f"{N_CHAINS}x{MALA_ITERS} MALA | seeds={SEEDS}", flush=True)
    P = K + NBETA_FIT
    agg = {kk: [] for kk in ["cif_r2", "cif_cov", "haz_r2", "haz_cov", "cif_rhat", "haz_rhat", "ess_c", "ess_h", "cif_psd", "haz_psd", "par_rhat", "cif_rmse", "waic", "dic", "lpml", "pdev", "cidx", "ibs", "cal_slope", "cal_intc"]}; hz = []
    for seed in SEEDS:
        x, d, e = simulate(N, 10*seed+2)
        xt, dte, ete = simulate(TEST_N, FIXED_TEST_SEED if FIXED_TEST_SEED >= 0 else 10*seed+3)
        xtt = torch.as_tensor(xt, dtype=torch.float64); tlam = true_lambda(xt); tcif = true_cif(xt)
        xt_torch = torch.as_tensor(x, dtype=torch.float64); dt = torch.as_tensor(d); et = torch.as_tensor(e)
        # teacher lambda on the STUDENT (training) covariates -> feeds the KL term Q
        teacher_lam = None
        if TEACHER == "oracle":
            tl = true_lambda(x)
            if TEACHER_BIAS != 0.0:                              # biased teacher: shift in logit space
                lt = np.log(tl/(1-tl)) + TEACHER_BIAS; tl = 1.0/(1.0+np.exp(-lt))
            teacher_lam = torch.as_tensor(tl, dtype=torch.float64)
        elif TEACHER == "fitted":                                # teacher = same form fit on a larger cohort
            xT, dT, eT = simulate(TEACHER_N, 10*seed+7)
            th_T = adam_map(torch.as_tensor(xT, dtype=torch.float64), torch.as_tensor(dT), torch.as_tensor(eT), seed+123, None)
            teacher_lam = torch.sigmoid(logits_of(th_T, xt_torch)).detach()
        map_th = adam_map(xt_torch, dt, et, seed, teacher_lam) if INIT == "adam" else None
        cif_ps, haz_ps, accs, cif_cm, haz_cm, sigs, th_cm = [], [], [], [], [], [], []
        for c in range(N_CHAINS):
            torch.manual_seed(1000*seed+c)
            if INIT == "adam":                                   # AdamW MAP + small independent perturbation per chain
                theta0 = map_th + ADAM_PERTURB*torch.randn(P, dtype=torch.float64)
            else:                                                # independent random init
                theta0 = 0.3*torch.randn(P, dtype=torch.float64)
            draws, acc, sig_hat = mala(xt_torch, dt, et, theta0, 7*seed+c, teacher_lam); accs.append(acc); sigs.append(sig_hat)
            th_cm.append(np.stack([t.numpy() for t in draws]))   # per-chain theta draws (for parameter R-hat)
            lam = np.stack([torch.sigmoid(logits_of(th, xtt)).numpy() for th in draws])          # [nd,N,K]
            cif = 1.0 - np.cumprod(1.0 - lam, axis=2)
            cif_ps.append(cif); haz_ps.append(lam)
            cif_cm.append(cif.mean(1)); haz_cm.append(lam.mean(1))                               # cohort-mean [nd,K]
        pooled_cif = np.concatenate(cif_ps, 0); pooled_haz = np.concatenate(haz_ps, 0)
        rhat_c, ess_c = rhat_ess(np.stack(cif_cm)); rhat_h, ess_h = rhat_ess(np.stack(haz_cm))
        rhat_p, _ = rhat_ess(np.stack(th_cm))                    # PARAMETER R-hat (vs functional) -> flags benign non-identifiability
        hc = per_horizon(pooled_cif, tcif); hh = per_horizon(pooled_haz, tlam)
        hz.append(dict(cif=hc, haz=hh))
        r2c = 1 - ((pooled_cif.mean(0)-tcif)**2).sum()/max(((tcif-tcif.mean())**2).sum(),1e-12)
        r2h = 1 - ((pooled_haz.mean(0)-tlam)**2).sum()/max(((tlam-tlam.mean())**2).sum(),1e-12)
        agg["cif_rhat"].append(rhat_c); agg["haz_rhat"].append(rhat_h); agg["ess_c"].append(ess_c); agg["ess_h"].append(ess_h); agg["par_rhat"].append(rhat_p)
        sig_str = f" sigma_hat={np.nanmean(sigs):.2f}" if HIER else ""
        print(f"  seed{seed}: acc={np.mean(accs):.2f}{sig_str} | CIF R-hat={rhat_c:.3f} (par {rhat_p:.2f}) R2={r2c:.3f} cov95={hc['cov'].mean():.3f} | "
              f"lambda R-hat={rhat_h:.3f} R2={r2h:.3f} cov95={hh['cov'].mean():.3f}", flush=True)
        agg["cif_r2"].append(r2c); agg["cif_cov"].append(hc["cov"].mean()); agg["cif_psd"].append(hc["post_sd"].mean()); agg["cif_rmse"].append(hc["rmse"].mean())
        agg["haz_r2"].append(r2h); agg["haz_cov"].append(hh["cov"].mean()); agg["haz_psd"].append(hh["post_sd"].mean())
        # test-set prediction outcomes (posterior-mean hazard on the fixed test set)
        mlam = pooled_haz.mean(0)                                # [Nt,K] posterior-mean hazard
        interval = np.stack([mlam, 1.0-mlam], axis=1)           # [Nt,2,K] (event, no-event)
        surv = np.cumprod(1.0-mlam, axis=1); cifL = 1.0 - surv[:, -1]
        agg["pdev"].append(predictive_deviance(interval, dte, ete, reduction="mean"))
        agg["cidx"].append(float(concordance_index(dte, ete, cifL)))
        agg["ibs"].append(float(integrated_brier_score(surv, dte, ete)))
        # calibration slope/intercept: logit(CIF_hat) ~ logit(CIF_true) pooled over horizons
        ph = np.clip(pooled_cif.mean(0).reshape(-1), 1e-4, 1-1e-4); pt = np.clip(tcif.reshape(-1), 1e-4, 1-1e-4)
        xg = np.log(pt/(1-pt)); yg = np.log(ph/(1-ph)); sl = float(np.polyfit(xg, yg, 1)[0])
        agg["cal_slope"].append(sl); agg["cal_intc"].append(float(yg.mean()-sl*xg.mean()))
        if CRITERIA:
            cr = criteria(np.concatenate(th_cm, 0), xt_torch, dt, et, teacher_lam)
            agg["waic"].append(cr["waic"]); agg["dic"].append(cr["dic"]); agg["lpml"].append(cr["lpml"])
            print(f"    criteria: WAIC={cr['waic']:.1f} DIC={cr['dic']:.1f} LPML={cr['lpml']:.1f} p_WAIC={cr['p_waic']:.1f}", flush=True)

    # per-horizon means over seeds
    def stack(fn, key): return np.stack([h[fn][key] for h in hz]).mean(0)
    print(f"\n--- per-horizon (mean over seeds), SPEC={SPEC} ---", flush=True)
    print(f"  CIF R2 : {np.round(stack('cif','r2'),3)}", flush=True)
    print(f"  CIF cov: {np.round(stack('cif','cov'),3)}", flush=True)
    print(f"  lam R2 : {np.round(stack('haz','r2'),3)}", flush=True)
    print(f"  lam cov: {np.round(stack('haz','cov'),3)}", flush=True)
    crit_str = (f" | WAIC {np.mean(agg['waic']):.1f} DIC {np.mean(agg['dic']):.1f} LPML {np.mean(agg['lpml']):.1f}" if CRITERIA else "")
    print(f"\n=== SPEC={SPEC} K={K} N={N} teacher={TEACHER} eta={ETA} omega={OMEGA}: "
          f"CIF R-hat {np.mean(agg['cif_rhat']):.3f} R2 {np.mean(agg['cif_r2']):.3f} RMSE {np.mean(agg['cif_rmse']):.4f} cov95 {np.mean(agg['cif_cov']):.3f} | "
          f"lambda R2 {np.mean(agg['haz_r2']):.3f} cov95 {np.mean(agg['haz_cov']):.3f}{crit_str} ===", flush=True)
    npz = {"cuts": np.arange(1, K+1)}
    for fn in ["cif", "haz"]:
        for key in ["r2", "cov", "bias", "post_sd", "rmse", "width", "truth_range", "truth_mean"]:
            npz[f"{fn}_{key}"] = stack(fn, key)
    np.savez(OUT_DIR/f"linear_{SPEC}_K{K}_N{N}_T{TEACHER}_eta{ETA:g}_om{OMEGA:g}.npz", **npz)
    # append one summary row per run (for the eta-vs-metric plots)
    import csv
    row = dict(eta=ETA, teacher=TEACHER, N=N, cif_r2=np.mean(agg['cif_r2']), cif_rmse=np.mean(agg['cif_rmse']),
               cif_cov=np.mean(agg['cif_cov']), haz_cov=np.mean(agg['haz_cov']), cif_width=np.mean([h["cif"]["width"].mean() for h in hz]),
               pdev=np.mean(agg['pdev']), cidx=np.mean(agg['cidx']), ibs=np.mean(agg['ibs']),
               cal_slope=np.mean(agg['cal_slope']), cal_intc=np.mean(agg['cal_intc']), cif_rhat=np.mean(agg['cif_rhat']),
               waic=(np.mean(agg['waic']) if CRITERIA else float('nan')),
               dic=(np.mean(agg['dic']) if CRITERIA else float('nan')),
               lpml=(np.mean(agg['lpml']) if CRITERIA else float('nan')))
    csvf = OUT_DIR/"eta_summary.csv"; header = not csvf.exists()
    with open(csvf, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header: w.writeheader()
        w.writerow(row)
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
