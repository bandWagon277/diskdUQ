#!/usr/bin/env python
"""Guideline simulation framework (references/guideline_implementation.md).

One common nonlinear discrete-time DGP; heterogeneity is injected via config so results stay
interpretable. Student = MLP (AdamW MAP + last-layer Laplace posterior); teacher = MLP trained on a
source cohort, predictions frozen (optionally distorted / feature-restricted). Three eta-selectors:
CV-Deviance, CV-C-index, LPML/CPO (+ WAIC, fixed eta, oracle). Central story: heterogeneity ->
teacher bias -> optimal borrowing decreases -> eta-selection prevents negative transfer.

EXP dispatch (env EXP): 1 control | 2 calibration-bias | 3 mean-shift | 6 baseline-shift |
7 concept-shift | 8 teacher-quality. Others reuse the same knobs. Env: R, N_T, N_S, N_TEST,
NUM_DURATIONS(K), H, EPOCHS, ETAS, SEED0, and per-experiment level knobs.
"""
from __future__ import annotations
import os, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from diskd.metrics import predictive_deviance, concordance_index, integrated_brier_score

# ------------------------------ config ---------------------------------------
EXP = os.environ.get("EXP", "1")
R = int(os.environ.get("R", 10)); SEED0 = int(os.environ.get("SEED0", 100))
P = 12; K = int(os.environ.get("NUM_DURATIONS", 20))
N_T = int(os.environ.get("N_T", 500)); N_S = int(os.environ.get("N_S", 10000)); N_TEST = int(os.environ.get("N_TEST", 5000))
H = int(os.environ.get("H", 32)); EPOCHS = int(os.environ.get("EPOCHS", 80)); LR = float(os.environ.get("LR", 1e-3))
PRIOR_PREC = float(os.environ.get("PRIOR_PREC", 1e-2)); LAP_M = int(os.environ.get("LAP_M", 400))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.1,0.25,0.5,1,2,5").split(",")]
HORIZONS = [int(h) for h in os.environ.get("HORIZONS", "5,10,15,20").split(",")]
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_guideline")))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
DEV = "cpu"

# ------------------------------ DGP ------------------------------------------
def fT(X):
    return (0.50*X[:,0] - 0.40*X[:,1] + 0.35*(X[:,2]**2 - 1) + 0.30*np.sin(X[:,3])
            + 0.30*X[:,4]*X[:,5] - 0.30*X[:,6] + 0.25*X[:,7] + 0.25*X[:,8]*X[:,9] + 0.20*X[:,10])
def fflip(X):
    return (-0.50*X[:,0] + 0.40*X[:,1] + 0.35*(X[:,2]**2 - 1) + 0.30*np.sin(X[:,3])
            - 0.30*X[:,4]*X[:,5] - 0.30*X[:,6] + 0.25*X[:,7] - 0.25*X[:,8]*X[:,9] + 0.20*X[:,10])
def f_mix(gamma):
    return (lambda X: (1-gamma)*fT(X) + gamma*fflip(X)) if gamma > 0 else fT

def cov(n, mu, rho, rng):
    idx = np.arange(P); Sig = rho ** np.abs(idx[:, None] - idx[None, :]); L = np.linalg.cholesky(Sig)
    m = np.zeros(P); m[:6] = mu
    return rng.normal(size=(n, P)) @ L.T + m[None, :]

def hazards(X, alpha, f):                                   # [n,K]
    return 1.0/(1.0 + np.exp(-(alpha[None, :] + f(X)[:, None])))

def calibrate_alpha(f, rng, target=0.37):
    Xc = cov(20000, 0.0, 0.3, rng)
    def rate(a):
        lam = 1.0/(1.0 + np.exp(-(a + f(Xc)))); lam = lam[:, None]*np.ones(K)[None, :]
        return 1.0 - np.prod(1.0 - lam, 1).mean()
    lo, hi = -8.0, 3.0
    for _ in range(40):
        mid = (lo+hi)/2
        if rate(mid) < target: lo = mid
        else: hi = mid
    return (lo+hi)/2 + np.zeros(K)

def calibrate_cens(alpha, f, rng, target=0.30):
    X = cov(20000, 0.0, 0.3, rng); lam = hazards(X, alpha, f)
    U = rng.random((20000, K)); fires = U < lam; has = fires.any(1); te = np.where(has, fires.argmax(1), K)
    def crate(pc):
        tc = np.minimum(rng.geometric(pc, 20000)-1, K-1); ev = (te <= tc) & (te < K); return 1.0 - ev.mean()
    lo, hi = 1e-3, 0.5
    for _ in range(40):
        mid = (lo+hi)/2
        if crate(mid) > target: hi = mid                    # smaller pc -> later censor -> less censoring? invert
        else: lo = mid
    return (lo+hi)/2

def simulate(n, rng, alpha, f, mu, rho, cens_p):
    X = cov(n, mu, rho, rng); lam = hazards(X, alpha, f)
    U = rng.random((n, K)); fires = U < lam; has = fires.any(1); te = np.where(has, fires.argmax(1), K)
    tc = np.minimum(rng.geometric(cens_p, n)-1, K-1)
    ev = (te <= tc) & (te < K); d = np.where(ev, te, np.minimum(tc, K-1)).astype(np.int64); e = ev.astype(np.int64)
    return X.astype(np.float64), d, e

def true_cif(X, alpha, f):                                  # [n,K]
    lam = hazards(X, alpha, f); return 1.0 - np.cumprod(1.0 - lam, axis=1)

# ------------------------------ model ----------------------------------------
class Net(nn.Module):
    def __init__(self, pin):
        super().__init__(); self.bb = nn.Sequential(nn.Linear(pin, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.head = nn.Linear(H, K)
    def forward(self, x): return self.head(self.bb(x))
    def feats(self, x): return self.bb(x)

def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_i(logits, d, e):
    tgt = torch.zeros_like(logits); tgt[torch.arange(len(d)), d] = e.double()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * _mask(d)).sum(1)
def _bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def kd_i(logits, tlam, d): return (_bkl(tlam, torch.sigmoid(logits)) * _mask(d)).sum(1)

BATCH = int(os.environ.get("BATCH", 64))
def train_student(X, d, e, tlam, eta, seed, pin=None):
    pin = pin or X.shape[1]; torch.manual_seed(seed)
    net = Net(pin).double().to(DEV)
    nn.init.constant_(net.head.bias, -3.0)                   # low baseline hazard init (~0.05) so CIF doesn't saturate
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    tb = torch.as_tensor(tlam, dtype=torch.float64) if tlam is not None else None
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    n = len(d); bs = min(BATCH, n)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad(); lg = net(xb[b]); loss = nll_i(lg, db[b], eb[b]).mean()
            if eta > 0 and tb is not None: loss = loss + eta*kd_i(lg, tb[b], db[b]).mean()
            loss = loss + 0.5*PRIOR_PREC*sum((p*p).sum() for p in net.parameters())/n
            loss.backward(); opt.step()
    return net

def head_vec(net): return torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
def logits_from_head(w, feat):                              # feat [n,H]
    W = w[:K*H].view(K, H); b = w[K*H:]; return feat @ W.T + b[None, :]

def laplace_draws(net, X, d, e, tlam, eta, seed):
    from torch.autograd.functional import hessian
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    feat = net.feats(xb).detach(); tb = torch.as_tensor(tlam, dtype=torch.float64) if tlam is not None else None
    w0 = head_vec(net)
    def U(w):
        lg = logits_from_head(w, feat); u = nll_i(lg, db, eb).sum()
        if eta > 0 and tb is not None: u = u + eta*kd_i(lg, tb, db).sum()
        return u + 0.5*PRIOR_PREC*(w @ w)
    Hs = hessian(U, w0).detach(); p = w0.numel()
    ridge = 1e-6*float(torch.diagonal(Hs).abs().mean()) + 1e-8
    covm = torch.linalg.inv(Hs + ridge*torch.eye(p, dtype=Hs.dtype)); covm = 0.5*(covm+covm.T)
    L = torch.linalg.cholesky(covm); g = torch.Generator().manual_seed(seed)
    z = torch.randn(LAP_M, p, generator=g, dtype=torch.float64)
    return [w0 + L @ z[m] for m in range(LAP_M)], feat, (tb, db, eb, w0)

# ------------------------------ teacher --------------------------------------
def make_teacher(Xs, ds, es, seed, feats_idx, distort=(0.0, 1.0)):
    """Train teacher MLP on source (optionally feature-restricted); return predict_haz(X)->[n,K] (distorted)."""
    net = train_student(Xs[:, feats_idx], ds, es, None, 0.0, seed, pin=len(feats_idx))
    a, b = distort
    def predict_haz(X):
        with torch.no_grad():
            lam = torch.sigmoid(net(torch.as_tensor(X[:, feats_idx], dtype=torch.float64))).numpy()
        if (a, b) != (0.0, 1.0):
            lg = np.log(np.clip(lam, 1e-6, 1-1e-6)/(1-np.clip(lam, 1e-6, 1-1e-6))); lam = 1.0/(1.0+np.exp(-(a+b*lg)))
        return lam
    return predict_haz

# ------------------------------ metrics --------------------------------------
def cif_from_draws(draws, feat):                            # -> pooled [M,n,K], mean CIF [n,K]
    with torch.no_grad():
        lam = np.stack([torch.sigmoid(logits_from_head(w, feat)).numpy() for w in draws])   # [M,n,K]
    return 1.0 - np.cumprod(1.0 - lam, axis=2)

def eval_metrics(cif_mean, cif_draws, X, d, e, alpha, f):
    tcif = true_cif(X, alpha, f)
    # predictive deviance from posterior-mean hazard implied by CIF: reconstruct hazard from CIF
    surv = 1.0 - cif_mean; lam = np.clip(1.0 - np.concatenate([surv[:, :1]/1.0, surv[:, 1:]/np.clip(surv[:, :-1], 1e-9, None)], 1), 1e-9, 1-1e-9)
    interval = np.stack([lam, 1.0 - lam], axis=1)            # [n,2,K] (event, no-event)
    dev = predictive_deviance(interval, d, e, reduction="mean")
    cidx = concordance_index(d, e, cif_mean[:, -1])
    ibs = integrated_brier_score(surv, d, e)
    rmse = float(np.sqrt(((cif_mean - tcif)**2).mean()))
    L = np.quantile(cif_draws, .025, 0); Uq = np.quantile(cif_draws, .975, 0)
    cov = float(((tcif >= L) & (tcif <= Uq)).mean()); width = float((Uq - L).mean())
    # calibration slope/intercept (logit CIF-hat ~ logit true-CIF at horizons)
    hz = [h-1 for h in HORIZONS if h <= K] or [K-1]
    p_hat = np.clip(cif_mean[:, hz].reshape(-1), 1e-4, 1-1e-4); p_tru = np.clip(tcif[:, hz].reshape(-1), 1e-4, 1-1e-4)
    xg = np.log(p_tru/(1-p_tru)); yg = np.log(p_hat/(1-p_hat))
    slope = float(np.polyfit(xg, yg, 1)[0]); intc = float(yg.mean() - slope*xg.mean())
    return dict(dev=dev, cidx=float(cidx), ibs=float(ibs), rmse=rmse, cov=cov, width=width,
                cal_slope=slope, cal_intc=intc)

# ------------------------------ selectors ------------------------------------
def cv_scores(Xt, dt, et, tlam, seed):
    """5-fold: per eta return summed held-out deviance and mean held-out C-index."""
    n = len(dt); rng = np.random.default_rng(seed); perm = rng.permutation(n); folds = np.array_split(perm, 5)
    dev = {e: 0.0 for e in ETAS}; cidx = {e: [] for e in ETAS}
    for e in ETAS:
        for v in range(5):
            va = folds[v]; tr = np.concatenate([folds[j] for j in range(5) if j != v])
            tl = tlam[tr] if tlam is not None else None
            net = train_student(Xt[tr], dt[tr], et[tr], tl, e, seed+v)
            with torch.no_grad():
                lam = torch.sigmoid(net(torch.as_tensor(Xt[va], dtype=torch.float64))).numpy()
            cifh = 1.0 - np.cumprod(1.0 - lam, 1); interval = np.stack([lam, 1-lam], 1)
            dev[e] += predictive_deviance(interval, dt[va], et[va], reduction="sum")
            try: cidx[e].append(concordance_index(dt[va], et[va], cifh[:, -1]))
            except Exception: pass
    return dev, {e: (np.mean(v) if v else np.nan) for e, v in cidx.items()}

def lpml_waic(Xt, dt, et, tlam, seed):
    """LPML (generalized CPO) and WAIC per eta from a full-data last-layer Laplace posterior."""
    lp = {}; wa = {}
    for e in ETAS:
        net = train_student(Xt, dt, et, tlam, e, seed)
        draws, feat, (tb, db, eb, w0) = laplace_draws(net, Xt, dt, et, tlam, e, seed)
        NLL, Q = [], []
        with torch.no_grad():
            for w in draws:
                lg = logits_from_head(w, feat); NLL.append(nll_i(lg, db, eb).numpy())
                Q.append(kd_i(lg, tb, db).numpy() if tb is not None else np.zeros(len(dt)))
        NLL = np.stack(NLL); Q = np.stack(Q)
        def lme(a): m = a.max(0, keepdims=True); return (m + np.log(np.mean(np.exp(a-m), 0, keepdims=True))).squeeze(0)
        a = e*Q; lp[e] = float((lme(a) - lme(a + NLL)).sum())
        lppd = lme(-NLL).sum(); pw = NLL.var(0, ddof=1).sum(); wa[e] = float(-2*(lppd - pw))
    return lp, wa

# ------------------------------ one replicate --------------------------------
def run_replicate(rep, het):
    rng = np.random.default_rng(SEED0 + rep)
    fS = f_mix(het.get("gamma", 0.0)); alphaS_off = het.get("delta0", 0.0)
    alpha = calibrate_alpha(fT, rng); cens_p = calibrate_cens(alpha, fT, rng)
    # target train/test (always target DGP)
    Xt, dt, et = simulate(N_T, rng, alpha, fT, 0.0, 0.3, cens_p)
    Xte, dte, ete = simulate(N_TEST, rng, alpha, fT, 0.0, 0.3, cens_p)
    # source (teacher training): heterogeneity in mu/rho/concept/baseline
    alphaS = alpha + alphaS_off
    Xs, ds, es = simulate(N_S, rng, alphaS, fS, het.get("mu", 0.0), het.get("rho", 0.3), cens_p)
    feats_idx = het.get("teacher_feats", np.arange(P))
    teacher = make_teacher(Xs, ds, es, SEED0+rep, feats_idx, distort=het.get("distort", (0.0, 1.0)))
    tlam_tr = teacher(Xt); tlam_te = teacher(Xte)
    feat_te_cache = {}
    def test_eval(eta):                                     # train full student at eta, eval on test
        net = train_student(Xt, dt, et, tlam_tr, eta, SEED0+rep)
        draws, _, _ = laplace_draws(net, Xt, dt, et, tlam_tr, eta, SEED0+rep)
        with torch.no_grad():
            feat_te = net.feats(torch.as_tensor(Xte, dtype=torch.float64)).detach()
        cd = cif_from_draws(draws, feat_te); return eval_metrics(cd.mean(0), cd, Xte, dte, ete, alpha, fT)
    return Xt, dt, et, tlam_tr, alpha, teacher, test_eval

# ------------------------------ experiments ----------------------------------
def _selected(dev, cidx, lpml, waic, test_by_eta):
    hatCVd = min(dev, key=dev.get); hatCVc = max(cidx, key=lambda k: (cidx[k] if not np.isnan(cidx[k]) else -1))
    hatLP = max(lpml, key=lpml.get); hatWA = min(waic, key=waic.get)
    orc = min(test_by_eta, key=lambda k: test_by_eta[k]["dev"])
    return dict(CVdev=hatCVd, CVc=hatCVc, LPML=hatLP, WAIC=hatWA, oracle=orc)

def experiment(het, selectors, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== EXP {EXP} [{tag}] het={het} | R={R} N_T/N_S/N_test={N_T}/{N_S}/{N_TEST} K={K} etas={ETAS} ===", flush=True)
    rows = []
    for rep in range(R):
        Xt, dt, et, tlam_tr, alpha, teacher, test_eval = run_replicate(rep, het)
        test_by_eta = {e: test_eval(e) for e in ETAS}
        dev, cidx = cv_scores(Xt, dt, et, tlam_tr, SEED0+rep)
        lpml, waic = lpml_waic(Xt, dt, et, tlam_tr, SEED0+rep) if any(s in selectors for s in ("LPML", "WAIC")) else ({e: np.nan for e in ETAS}, {e: np.nan for e in ETAS})
        sel = _selected(dev, cidx, lpml, waic, test_by_eta)
        dev0 = test_by_eta[0.0]["dev"]
        for name in selectors + ["fixed1", "internal"]:
            eh = {"fixed1": 1.0, "internal": 0.0}.get(name, sel.get(name))
            if eh is None: continue
            m = test_by_eta[eh]; regret = m["dev"] - min(v["dev"] for v in test_by_eta.values())
            rows.append(dict(rep=rep, selector=name, eta_hat=eh, **m, nt=int(m["dev"] > dev0), regret=regret))
        print(f"  rep{rep}: CVdev={sel['CVdev']:g} CVc={sel['CVc']:g} LPML={sel['LPML']:g} oracle={sel['oracle']:g}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(OUT_DIR/f"exp{EXP}_{tag}.csv", index=False)
    g = df.groupby("selector")
    print(f"\n--- EXP {EXP} [{tag}] summary (mean over {R} reps) ---", flush=True)
    print(f"{'selector':10} {'mean_eta':>8} {'P(eta=0)':>8} {'dev':>7} {'C':>6} {'IBS':>6} {'RMSE':>6} {'cov':>5} {'NTrate':>6} {'regret':>7}", flush=True)
    for s, gg in g:
        print(f"{s:10} {gg['eta_hat'].mean():8.2f} {(gg['eta_hat']==0).mean():8.2f} {gg['dev'].mean():7.3f} "
              f"{gg['cidx'].mean():6.3f} {gg['ibs'].mean():6.3f} {gg['rmse'].mean():6.3f} {gg['cov'].mean():5.2f} "
              f"{gg['nt'].mean():6.2f} {gg['regret'].mean():7.3f}", flush=True)
    print(f"Saved {OUT_DIR}/exp{EXP}_{tag}.csv", flush=True)


def main():
    if EXP == "1":                                          # well-specified control-style: teacher conditions, fixed eta
        experiment(dict(), ["oracle"], "control")
    elif EXP == "2":                                        # calibration bias: a in logit(p*)=a+b logit(p)
        for a, b, lab in [(0.0, 1.0, "C0"), (0.5, 1.0, "C1"), (1.0, 1.0, "C2"), (0.0, 1.5, "C3")]:
            experiment(dict(distort=(a, b)), ["CVdev", "CVc", "LPML", "WAIC", "oracle"], lab)
    elif EXP == "3":                                        # marginal predictor mean shift
        for dl in [0.0, 0.5, 1.0]:
            experiment(dict(mu=dl), ["CVdev", "CVc", "LPML", "oracle"], f"delta{dl}")
    elif EXP == "6":                                        # baseline-risk shift
        for d0 in [0.0, 0.5, 1.0]:
            experiment(dict(delta0=d0), ["CVdev", "CVc", "LPML", "oracle"], f"d0_{d0}")
    elif EXP == "7":                                        # concept shift
        for gm in [0.0, 0.5, 1.0]:
            experiment(dict(gamma=gm), ["CVdev", "CVc", "LPML", "oracle"], f"gamma{gm}")
    elif EXP == "8":                                        # teacher quality (feature restriction)
        feats = {"good": np.arange(12), "fair": np.array([0,1,2,4,5,6,8,9,10]), "poor": np.array([0,1,4,5,8,9])}
        for lab, fi in feats.items():
            experiment(dict(teacher_feats=fi), ["CVdev", "CVc", "LPML", "oracle"], lab)
    else:
        raise SystemExit(f"EXP {EXP} not implemented")


if __name__ == "__main__":
    main()
