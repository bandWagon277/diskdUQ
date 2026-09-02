#!/usr/bin/env python
"""MLP student on the E1 LINEAR well-specified DGP -- a one-variable swap vs the GLM student.

Everything else identical to E1: linear logistic DGP (alpha_k=logit(0.15), beta=[.8,-.6,.5]), oracle
teacher, K=10, sigma=10, same N and eta grid. Only the student's functional form changes: GLM ->
2-hidden-layer MLP. Full-network MALA over an MLP does not converge, so we use the convergent
Bayesian-last-layer scheme (AdamW-MAP the backbone with teacher KD, freeze it, Laplace over the head).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from diskd.metrics import predictive_deviance, concordance_index, integrated_brier_score

K = int(os.environ.get("NUM_DURATIONS", 10)); D = 3; H = int(os.environ.get("HIDDEN", 16))
N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 1000)); R = int(os.environ.get("R", 20))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.5,1").split(",")]
SIGMA = float(os.environ.get("SIGMA", 10)); PREC = 1.0/SIGMA**2
EPOCHS = int(os.environ.get("EPOCHS", 300)); LR = float(os.environ.get("LR", 0.02)); BATCH = int(os.environ.get("BATCH", 64))
LAP_M = int(os.environ.get("LAP_M", 3000)); SEED0 = int(os.environ.get("SEED0", 42)); FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", 999))
BETA = np.array([0.8, -0.6, 0.5]); AB = -1.7346
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_expMLP")))

def sig(z): return 1.0/(1.0+np.exp(-z))
def true_lam(x): return sig(AB + x @ BETA)[:, None]*np.ones(K)[None, :]
def true_cif(x): return 1.0 - np.cumprod(1.0 - true_lam(x), 1)
def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); lam = true_lam(x)
    U = rng.random((n, K)); fires = U < lam; has = fires.any(1)
    return x, np.where(has, fires.argmax(1), K-1).astype(np.int64), np.where(has, 1, 0).astype(np.int64)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU()); self.head = nn.Linear(H, K)
    def forward(self, x): return self.head(self.bb(x))
    def feats(self, x): return self.bb(x)

def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_i(lg, d, e):
    tgt = torch.zeros_like(lg); tgt[torch.arange(len(d)), d] = e.double()
    return (nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none")*_mask(d)).sum(1)
def bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def kd_i(lg, tl, d): return (bkl(tl, torch.sigmoid(lg))*_mask(d)).sum(1)

def train_map(X, d, e, tl, eta, seed):
    torch.manual_seed(seed); net = Net().double(); nn.init.constant_(net.head.bias, AB)
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    tb = torch.as_tensor(tl, dtype=torch.float64) if tl is not None else None
    opt = torch.optim.AdamW(net.parameters(), lr=LR); n = len(d); bs = min(BATCH, n)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad(); lg = net(xb[b]); loss = nll_i(lg, db[b], eb[b]).mean()
            if eta > 0 and tb is not None: loss = loss + eta*kd_i(lg, tb[b], db[b]).mean()
            loss = loss + 0.5*PREC*sum((p*p).sum() for p in net.parameters())/n
            loss.backward(); opt.step()
    return net

def head_vec(net): return torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
def logits_head(w, feat): W = w[:K*H].view(K, H); b = w[K*H:]; return feat @ W.T + b[None, :]

def laplace(net, X, d, e, tl, eta, seed):
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    feat = net.feats(xb).detach(); tb = torch.as_tensor(tl, dtype=torch.float64) if tl is not None else None
    w0 = head_vec(net)
    def U(w):
        lg = logits_head(w, feat); u = nll_i(lg, db, eb).sum()
        if eta > 0 and tb is not None: u = u + eta*kd_i(lg, tb, db).sum()
        return u + 0.5*PREC*(w @ w)
    Hs = hessian(U, w0).detach(); p = w0.numel(); rg = 1e-6*float(torch.diagonal(Hs).abs().mean()) + 1e-8
    cov = torch.linalg.inv(Hs + rg*torch.eye(p, dtype=Hs.dtype)); cov = 0.5*(cov+cov.T)
    L = torch.linalg.cholesky(cov); g = torch.Generator().manual_seed(seed); z = torch.randn(LAP_M, p, generator=g, dtype=torch.float64)
    return [w0 + L @ z[m] for m in range(LAP_M)], feat

def cif_lam(draws, feat):
    with torch.no_grad(): lam = np.stack([torch.sigmoid(logits_head(w, feat)).numpy() for w in draws])
    return 1.0 - np.cumprod(1.0 - lam, 2), lam

def per_h(draws, truth):
    pm = draws.mean(0); L = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    return dict(pm=pm, cov=((truth >= L) & (truth <= Uq)).mean(0), width=(Uq-L).mean(0), psd=draws.std(0).mean(0),
                rmse=np.sqrt(((pm-truth)**2).mean(0)),
                isc=((Uq-L) + (2/.05)*(L-truth)*(truth < L) + (2/.05)*(truth-Uq)*(truth > Uq)).mean(0))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
    xt, dte, ete = simulate(TEST_N, FIXED_TEST_SEED); tcif = true_cif(xt); xtt = torch.as_tensor(xt, dtype=torch.float64)
    print(f"=== MLP student (last-layer Laplace), LINEAR well-spec DGP, oracle teacher === "
          f"K={K} N={N} R={R} H={H} sigma={SIGMA} | head_dim={K*H+K} | etas={ETAS}", flush=True)
    rows = []
    for eta in ETAS:
        accov, acw, acr, acp, acis, emp, dv, ib, cx, hz = [], [], [], [], [], [], [], [], [], []
        for r in range(R):
            x, d, e = simulate(N, SEED0+r); tl = true_lam(x) if eta > 0 else None
            net = train_map(x, d, e, tl, eta, 1000*(SEED0+r))
            draws, _ = laplace(net, x, d, e, tl, eta, 1000*(SEED0+r)+1)
            feat_te = net.feats(xtt).detach(); cif, lam = cif_lam(draws, feat_te)
            h = per_h(cif, tcif); accov.append(h["cov"].mean()); acw.append(h["width"].mean()); acr.append(h["rmse"].mean())
            acp.append(h["psd"].mean()); acis.append(h["isc"].mean()); emp.append(h["pm"]); hz.append(h)
            mlam = lam.mean(0); surv = np.cumprod(1-mlam, 1); iv = np.stack([mlam, 1-mlam], 1)
            dv.append(predictive_deviance(iv, dte, ete, reduction="mean")); ib.append(float(integrated_brier_score(surv, dte, ete)))
            cx.append(float(concordance_index(dte, ete, h["pm"][:, -1])))
        empsd = np.stack(emp).std(0).mean(); ratio = np.mean(acp)/max(empsd, 1e-9)
        np.savez(OUT_DIR/f"expMLP_eta{eta:g}.npz",                       # per-horizon means over R (for plotting)
                 cif_cov=np.stack([h["cov"] for h in hz]).mean(0), cif_width=np.stack([h["width"] for h in hz]).mean(0),
                 cif_rmse=np.stack([h["rmse"] for h in hz]).mean(0), cif_psd=np.stack([h["psd"] for h in hz]).mean(0))
        row = dict(eta=eta, cov=np.mean(accov), width=np.mean(acw), IS=np.mean(acis), rmse=np.mean(acr),
                   post_emp=ratio, dev=np.mean(dv), C=np.mean(cx), IBS=np.mean(ib))
        rows.append(row)
        print(f"  eta={eta:<4g} | CIF cov {row['cov']:.3f} w {row['width']:.3f} IS {row['IS']:.3f} RMSE {row['rmse']:.4f} "
              f"Post/Emp {row['post_emp']:.2f} | dev {row['dev']:.3f} C {row['C']:.3f} IBS {row['IBS']:.3f}", flush=True)
    import pandas as pd; pd.DataFrame(rows).to_csv(OUT_DIR/"expMLP_grid.csv", index=False)
    print(f"\nSaved to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
