#!/usr/bin/env python
"""NN-student analogues of E1/E3/E4/E5 on the ORIGINAL DiSKD squared-rate DGP.

Student = MLP backbone (AdamW MAP) with a Bayesian last-layer head (Laplace posterior of the
generalized potential  U(w) = NLL_sum + eta*Q_KL_sum + 0.5*prec*||w||^2 ; convex variant divides the
data part by (1+eta)).  The backbone is frozen at its MAP, so the head posterior is a convex logistic
problem that mixes cleanly -- this is the convergent NN scheme that reproduces the historical baseline
(full-network MALA does NOT converge on this DGP).

Same knobs as the GLM probes:
  ETA / SIGMA (grid)         -> E1 (well-specified NN)
  SANDWICH=1                 -> E3 (per-horizon raw vs Godambe sandwich)
  NFEAT<12 (drop shared grp) -> E4 (omitted-covariate misspecification)
  CONVEX=1                   -> E5 (convex Bayesian DiSKD)
  TEACHER=oracle|fitted      -> oracle (closed-form true hazard) or MLP trained on external data

Reports the full per-time-interval CIF metric suite: coverage, width, MeanPostSD, EmpSD,
MeanPostSD/EmpSD, interval score IS_0.05, bias, RMSE; plus C^td, IBS, predictive deviance.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd.functional import hessian, jacobian
from diskd.preprocessing import fit_time_grid, transform_durations
from diskd.metrics import predictive_deviance, concordance_index, integrated_brier_score

# ------------------------------ config ---------------------------------------
D = 12
K = int(os.environ.get("NUM_DURATIONS", 10))
H = int(os.environ.get("HIDDEN", 24))
NFEAT = int(os.environ.get("NFEAT", D))                 # < 12 (=8 drops the shared z3 group) -> E4 misspec
CONVEX = int(os.environ.get("CONVEX", 0))               # E5: data part /(1+eta)
SANDWICH = int(os.environ.get("SANDWICH", 0))           # E3: also compute Godambe sandwich + per-horizon
TEACHER = os.environ.get("TEACHER", "oracle")           # oracle | fitted
TEACHER_N = int(os.environ.get("TEACHER_N", 2000))
N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 2000)); R = int(os.environ.get("R", 20))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.5,1").split(",")]
SIGMAS = [float(s) for s in os.environ.get("SIGMAS", "1,5,10,30").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", 250)); LR = float(os.environ.get("LR", 0.02)); BATCH = int(os.environ.get("BATCH", 64))
LAP_M = int(os.environ.get("LAP_M", 4000))
CENSOR_MAX = float(os.environ.get("CENSOR_MAX", 0.05))
BETA_R1 = float(os.environ.get("BETA_R1", 2.0)); BETA_R2 = float(os.environ.get("BETA_R2", 2.0)); BETA_SHARED = float(os.environ.get("BETA_SHARED", 8.0))
SEED0 = int(os.environ.get("SEED0", 42)); FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", 999))
GRID_SCHEME = os.environ.get("GRID_SCHEME", "quantiles")
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_expNN")))
DEV = torch.device("cpu")

# ------------------------------ squared-rate DGP -----------------------------
def rates(x):                                            # original DiSKD squared rates (single-risk total)
    z1 = x[:, 0:4].sum(1); z2 = x[:, 4:8].sum(1); z3 = x[:, 8:12].sum(1)
    r1 = np.clip((BETA_R1*z1)**2 + (BETA_SHARED*z3)**2, 1e-3, None)
    r2 = np.clip((BETA_R2*z2)**2 + (BETA_SHARED*z3)**2, 1e-3, None)
    return np.stack([r1, r2], axis=1)

def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); r = rates(x)
    t = rng.exponential(scale=1.0/r); et = t.min(1)
    cens = rng.uniform(0.0, CENSOR_MAX, size=n); censored = cens < et
    dur = np.where(censored, cens, et); event = np.where(censored, 0, 1).astype(np.int64)
    return x, dur.astype(float), event

def true_cif_haz(x, cuts):                               # closed-form single-risk truth on the (KM) grid
    rt = rates(x).sum(1, keepdims=True)                  # total event rate [N,1]
    cuts = np.asarray(cuts); prev = np.concatenate([[0.0], cuts[:-1]]); delta = (cuts - prev)[None, :]
    cif = 1.0 - np.exp(-rt*cuts[None, :])                # [N,K]
    haz = 1.0 - np.exp(-rt*delta)                        # [N,K]
    return cif, haz

# ------------------------------ MLP + last layer -----------------------------
class Net(nn.Module):
    def __init__(self, pin):
        super().__init__(); self.bb = nn.Sequential(nn.Linear(pin, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.head = nn.Linear(H, K)
    def forward(self, x): return self.head(self.bb(x))
    def feats(self, x): return self.bb(x)

def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_i(logits, d, e):
    tgt = torch.zeros_like(logits); tgt[torch.arange(len(d)), d] = e.double()
    return (nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")*_mask(d)).sum(1)
def _bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def kd_i(logits, tlam, d): return (_bkl(tlam, torch.sigmoid(logits))*_mask(d)).sum(1)

def train_map(X, d, e, tlam, eta, prec, seed, pin):
    torch.manual_seed(seed); net = Net(pin).double().to(DEV)
    nn.init.constant_(net.head.bias, -3.0)
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    tb = torch.as_tensor(tlam, dtype=torch.float64) if tlam is not None else None
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0); n = len(d); bs = min(BATCH, n)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad(); lg = net(xb[b]); loss = nll_i(lg, db[b], eb[b]).mean()
            if eta > 0 and tb is not None: loss = loss + eta*kd_i(lg, tb[b], db[b]).mean()
            loss = loss + 0.5*prec*sum((p*p).sum() for p in net.parameters())/n
            loss.backward(); opt.step()
    return net

def head_vec(net): return torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
def logits_from_head(w, feat): W = w[:K*H].view(K, H); b = w[K*H:]; return feat @ W.T + b[None, :]

def head_potential_pieces(w, feat, db, eb, tb, eta):
    """Return (nll_sum, q_sum) for the head weights w given frozen features."""
    lg = logits_from_head(w, feat)
    ns = nll_i(lg, db, eb).sum()
    qs = kd_i(lg, tb, db).sum() if (eta > 0 and tb is not None) else torch.zeros((), dtype=torch.float64)
    return ns, qs

def fit_head_posterior(net, X, d, e, tlam, eta, prec, seed, want_sandwich):
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    feat = net.feats(xb).detach(); tb = torch.as_tensor(tlam, dtype=torch.float64) if tlam is not None else None
    w0 = head_vec(net); p = w0.numel()

    def U(w):
        ns, qs = head_potential_pieces(w, feat, db, eb, tb, eta)
        data = ns + eta*qs
        if CONVEX: data = data/(1.0+eta)
        return data + 0.5*prec*(w @ w)
    Hs = hessian(U, w0).detach(); ridge = 1e-6*float(torch.diagonal(Hs).abs().mean()) + 1e-8
    I = torch.eye(p, dtype=torch.float64); H_eta = Hs + ridge*I
    S_raw = torch.linalg.inv(H_eta); S_raw = 0.5*(S_raw+S_raw.T)
    g = torch.Generator().manual_seed(seed); z = torch.randn(LAP_M, p, generator=g, dtype=torch.float64)
    Lr = torch.linalg.cholesky(S_raw + 1e-12*I); raw = [w0 + Lr @ z[m] for m in range(LAP_M)]
    sand = None
    if want_sandwich:
        def score_i(w):                                  # per-subject score of the (convex-scaled) data potential
            lg = logits_from_head(w, feat); s = nll_i(lg, db, eb) + eta*(kd_i(lg, tb, db) if (eta > 0 and tb is not None) else 0.0)
            return s/(1.0+eta) if CONVEX else s
        G = jacobian(score_i, w0).detach()               # [N,p]
        J = G.T @ G; V = S_raw @ J @ S_raw; V = 0.5*(V+V.T)
        Ls = torch.linalg.cholesky(V + 1e-12*I); sand = [w0 + Ls @ z[m] for m in range(LAP_M)]
    return raw, sand, feat

def cif_lam_draws(draws, feat):
    with torch.no_grad(): lam = np.stack([torch.sigmoid(logits_from_head(w, feat)).numpy() for w in draws])  # [M,Nt,K]
    return 1.0 - np.cumprod(1.0 - lam, axis=2), lam

# ------------------------------ teacher --------------------------------------
def oracle_teacher(X, cuts): return true_cif_haz(X, cuts)[1]      # [N,K] discrete hazard

def fitted_teacher_factory(seed, cuts):
    Xs, ds, es = simulate(TEACHER_N, 10_000 + seed)
    dur_idx = transform_durations(ds, _grid_from_cuts(cuts))
    net = train_map(Xs[:, :NFEAT], dur_idx, es, None, 0.0, 1.0, 7_000 + seed, pin=NFEAT)
    def predict(X):
        with torch.no_grad(): return torch.sigmoid(net(torch.as_tensor(X[:, :NFEAT], dtype=torch.float64))).numpy()
    return predict

class _G:                                                # tiny shim so transform_durations sees a .cuts grid
    def __init__(self, cuts): self.cuts = np.asarray(cuts)
def _grid_from_cuts(cuts): return _G(cuts)

# ------------------------------ metrics --------------------------------------
def per_h(draws, truth):                                 # draws [M,Nt,K], truth [Nt,K]
    pm = draws.mean(0); L = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    cov = ((truth >= L) & (truth <= Uq)).mean(0); width = (Uq-L).mean(0); psd = draws.std(0).mean(0)
    isc = ((Uq-L) + (2/0.05)*(L-truth)*(truth < L) + (2/0.05)*(truth-Uq)*(truth > Uq)).mean(0)
    return dict(pm=pm, cov=cov, width=width, psd=psd, bias=(pm-truth).mean(0),
                rmse=np.sqrt(((pm-truth)**2).mean(0)), isc=isc)

def cover_wh(draws, truth):
    L = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    return ((truth >= L) & (truth <= Uq)).mean(0), (Uq-L).mean(0)

# ------------------------------ main -----------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
    xt, dte_raw, ete = simulate(TEST_N, FIXED_TEST_SEED)
    # KM-style grid fit once on a large reference draw so the estimand grid is fixed across replicates
    xg, dg, eg = simulate(max(N, 4000), 7)
    tg = fit_time_grid(dg, K, scheme=GRID_SCHEME); cuts = np.asarray(tg.cuts)
    dte = transform_durations(dte_raw, tg)
    tcif, tlam = true_cif_haz(xt, cuts)
    tag = f"NN squared+MLP | TEACHER={TEACHER} NFEAT={NFEAT} CONVEX={CONVEX} SANDWICH={SANDWICH}"
    print(f"=== {tag} === K={K} N={N} R={R} H={H} | etas={ETAS} sigmas={SIGMAS}", flush=True)
    print(f"    grid={GRID_SCHEME} cuts[0,mid,-1]=[{cuts[0]:.4f},{cuts[K//2]:.4f},{cuts[-1]:.4f}] "
          f"true CIF range [{tcif.min():.3f},{tcif.max():.3f}]", flush=True)
    rows = []
    for eta in ETAS:
        for sg in SIGMAS:
            prec = 1.0/sg**2
            pm_c = []; acc = {k: [] for k in ["cov_c","w_c","psd_c","bias_c","rmse_c","is_c","cidx","ibs","dev","mae"]}
            hz_c = []; sand_cov = []; sand_w = []; raw_cov = []; raw_w = []
            for r in range(R):
                x, d_raw, e = simulate(N, SEED0+r); d = transform_durations(d_raw, tg)
                if TEACHER == "oracle": tlam_tr = oracle_teacher(x, cuts)
                else: tlam_tr = fitted_teacher_factory(SEED0+r, cuts)(x)
                tlam_tr = tlam_tr if eta > 0 else None
                net = train_map(x[:, :NFEAT], d, e, tlam_tr, eta, prec, 1000*(SEED0+r), pin=NFEAT)
                raw, sand, _ = fit_head_posterior(net, x[:, :NFEAT], d, e, tlam_tr, eta, prec, 1000*(SEED0+r)+1, SANDWICH)
                xtt_feat = None
                cif, lam = cif_lam_draws(raw, net.feats(torch.as_tensor(xt[:, :NFEAT], dtype=torch.float64)).detach())
                hc = per_h(cif, tcif); hz_c.append(hc); pm_c.append(hc["pm"])
                acc["cov_c"].append(hc["cov"].mean()); acc["w_c"].append(hc["width"].mean()); acc["psd_c"].append(hc["psd"].mean())
                acc["bias_c"].append(np.abs(hc["bias"]).mean()); acc["rmse_c"].append(hc["rmse"].mean()); acc["is_c"].append(hc["isc"].mean())
                mlam = lam.mean(0); interval = np.stack([mlam, 1-mlam], 1); surv = np.cumprod(1-mlam, 1)
                acc["dev"].append(predictive_deviance(interval, dte, ete, reduction="mean"))
                acc["cidx"].append(float(concordance_index(dte, ete, hc["pm"][:, -1])))
                acc["ibs"].append(float(integrated_brier_score(surv, dte, ete)))
                acc["mae"].append(float(np.abs(hc["pm"]-tcif).mean()))
                rc, rw = cover_wh(cif, tcif); raw_cov.append(rc); raw_w.append(rw)
                if SANDWICH and sand is not None:
                    cifs, _ = cif_lam_draws(sand, net.feats(torch.as_tensor(xt[:, :NFEAT], dtype=torch.float64)).detach())
                    sc, sw = cover_wh(cifs, tcif); sand_cov.append(sc); sand_w.append(sw)
            empsd_c = np.stack(pm_c).std(0).mean(0); mpsd_c = np.stack([h["psd"] for h in hz_c]).mean(0)
            ratio_c = (mpsd_c/np.maximum(empsd_c, 1e-9)).mean()
            row = dict(eta=eta, sigma=sg, **{k: float(np.mean(v)) for k, v in acc.items()},
                       empsd_c=float(empsd_c.mean()), ratio_c=float(ratio_c))
            rows.append(row)
            npz = {f"cif_{k}": np.stack([h[k] for h in hz_c]).mean(0) for k in ["cov","width","psd","bias","rmse","isc"]}
            npz["cif_empsd"] = empsd_c; npz["cif_rawcov"] = np.stack(raw_cov).mean(0); npz["cif_raww"] = np.stack(raw_w).mean(0)
            if SANDWICH and sand_cov:
                npz["cif_sandcov"] = np.stack(sand_cov).mean(0); npz["cif_sandw"] = np.stack(sand_w).mean(0)
            np.savez(OUT_DIR/f"expNN_eta{eta:g}_sig{sg:g}.npz", **npz)
            extra = ""
            if SANDWICH and sand_cov: extra = f" | SAND cov {np.stack(sand_cov).mean():.3f} w {np.stack(sand_w).mean():.3f}"
            print(f"  eta={eta:<4g} sig={sg:<4g} | CIF cov {row['cov_c']:.3f} w {row['w_c']:.3f} IS {row['is_c']:.3f} "
                  f"RMSE {row['rmse_c']:.4f} Post/Emp {row['ratio_c']:.2f} | dev {row['dev']:.3f} C {row['cidx']:.3f} IBS {row['ibs']:.3f}{extra}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR/"expNN_grid.csv", index=False)
    print(f"\nSaved to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
