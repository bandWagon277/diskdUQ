#!/usr/bin/env python
"""Stage 1 of the GLM->Neural DiSKD plan: architecture change, SAME linear DGP & population.

Configurable teacher and student architecture (glm | mlp), fitted teacher on D_ext, Bayesian student
via MAP + last-layer Laplace (GLM: full 13-dim Laplace; MLP: frozen backbone + head Laplace) with the
eta*KL distillation term.  Same linear well-specified DGP as G0/E1: logit lambda_k = alpha_k + X^T beta.

  G1 : TEACHER_ARCH=glm STUDENT_ARCH=mlp
  G2 : TEACHER_ARCH=mlp STUDENT_ARCH=mlp
  G3a: TEACHER_ARCH=mlp STUDENT_ARCH=glm
  G3b: TEACHER_ARCH=glm STUDENT_ARCH=mlp   (= G1)
  G3c: TEACHER_ARCH=mlp TEACHER_H=64,32  STUDENT_ARCH=mlp STUDENT_H=16

Reports per-horizon CIF coverage/width/rmse/psd + C/IBS/dev + Post/Emp + teacher quality on test.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from diskd.metrics import predictive_deviance, concordance_index, integrated_brier_score

K = int(os.environ.get("NUM_DURATIONS", 10)); D = 3
N = int(os.environ.get("N", 500)); N_EXT = int(os.environ.get("N_EXT", 10000)); N_TEST = int(os.environ.get("N_TEST", 5000))
R = int(os.environ.get("R", 20)); ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.5,1").split(",")]
SIGMA = float(os.environ.get("SIGMA", 10)); PREC = 1.0/SIGMA**2
TEACHER_ARCH = os.environ.get("TEACHER_ARCH", "glm"); STUDENT_ARCH = os.environ.get("STUDENT_ARCH", "mlp")
TEACHER_H = [int(h) for h in os.environ.get("TEACHER_H", "32").split(",")]
STUDENT_H = [int(h) for h in os.environ.get("STUDENT_H", "16").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", 400)); LR = float(os.environ.get("LR", 0.02)); BATCH = int(os.environ.get("BATCH", 64))
WD = float(os.environ.get("WD", 1e-2))                    # backbone weight decay (regularize MLP toward the linear truth)
LAP_M = int(os.environ.get("LAP_M", 3000)); ADAM_STEPS = int(os.environ.get("ADAM_STEPS", 2000))
SEED0 = int(os.environ.get("SEED0", 42)); FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", 999)); TEACHER_SEED0 = int(os.environ.get("TEACHER_SEED0", 600000))
BETA = np.array([0.8, -0.6, 0.5]); AB = -1.7346
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_gstage")))
TAG = os.environ.get("TAG", f"{TEACHER_ARCH}2{STUDENT_ARCH}")

def sig(z): return 1.0/(1.0+np.exp(-z))
def true_lam(x): return sig(AB + x @ BETA)[:, None]*np.ones(K)[None, :]
def true_cif(x): return 1.0 - np.cumprod(1.0 - true_lam(x), 1)
def simulate(n, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); lam = true_lam(x)
    U = rng.random((n, K)); fires = U < lam; has = fires.any(1)
    return x, np.where(has, fires.argmax(1), K-1).astype(np.int64), np.where(has, 1, 0).astype(np.int64)

def _mask(d): k = torch.arange(K); return (k[None, :] <= d[:, None]).double()
def nll_i(lg, d, e):
    tgt = torch.zeros_like(lg); tgt[torch.arange(len(d)), d] = e.double()
    return (nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none")*_mask(d)).sum(1)
def bkl(a, b): a = a.clamp(1e-6, 1-1e-6); b = b.clamp(1e-6, 1-1e-6); return a*torch.log(a/b)+(1-a)*torch.log((1-a)/(1-b))
def kd_i(lg, tl, d): return (bkl(tl, torch.sigmoid(lg))*_mask(d)).sum(1)

# ---------------- MLP with separate alpha_k + last-layer head ----------------
class MLP(nn.Module):
    def __init__(self, hs):
        super().__init__(); layers = []; pin = D
        for h in hs: layers += [nn.Linear(pin, h), nn.ReLU()]; pin = h
        self.bb = nn.Sequential(*layers); self.Hd = pin; self.head = nn.Linear(pin, K)
    def forward(self, x): return self.head(self.bb(x))
    def feats(self, x): return self.bb(x)

def train_mlp(X, d, e, tl, eta, seed, hs, epochs=EPOCHS):
    torch.manual_seed(seed); net = MLP(hs).double(); nn.init.constant_(net.head.bias, AB)
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    tb = torch.as_tensor(tl, dtype=torch.float64) if tl is not None else None
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); n = len(d); bs = min(BATCH, n)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad(); lg = net(xb[b]); loss = nll_i(lg, db[b], eb[b]).mean()
            if eta > 0 and tb is not None: loss = loss + eta*kd_i(lg, tb[b], db[b]).mean()
            loss.backward(); opt.step()
    return net

# ---------------- GLM (13-param) MAP + logits --------------------------------
def glm_logits(th, x): return th[:K][None, :] + (torch.as_tensor(x, dtype=torch.float64) @ th[K:])[:, None]
def train_glm(X, d, e, tl, eta, seed):
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    tb = torch.as_tensor(tl, dtype=torch.float64) if tl is not None else None
    torch.manual_seed(seed); th = (0.3*torch.randn(K+D, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.AdamW([th], lr=0.05)
    for _ in range(ADAM_STEPS):
        opt.zero_grad(); lg = glm_logits(th, xb); u = nll_i(lg, db, eb).sum() + 0.5*PREC*(th@th)
        if eta > 0 and tb is not None: u = u + eta*kd_i(lg, tb, db).sum()
        u.backward(); opt.step()
    return th.detach()

# ---------------- teacher (fitted on D_ext) ----------------------------------
def make_teacher(seed):
    xe, de, ee = simulate(N_EXT, TEACHER_SEED0 + seed)
    if TEACHER_ARCH == "glm":
        th = train_glm(xe, de, ee, None, 0.0, TEACHER_SEED0 + seed)
        return lambda x: torch.sigmoid(glm_logits(th, x)).detach().numpy()
    net = train_mlp(xe, de, ee, None, 0.0, TEACHER_SEED0 + seed, TEACHER_H)
    return lambda x: torch.sigmoid(net(torch.as_tensor(x, dtype=torch.float64))).detach().numpy()

# ---------------- Bayesian student (MAP + last-layer Laplace) ----------------
def student_draws(X, d, e, tl, eta, seed):
    xb = torch.as_tensor(X, dtype=torch.float64); db = torch.as_tensor(d); eb = torch.as_tensor(e)
    tb = torch.as_tensor(tl, dtype=torch.float64) if tl is not None else None
    if STUDENT_ARCH == "glm":
        w0 = train_glm(X, d, e, tl, eta, seed)
        def U(w):
            lg = glm_logits(w, xb); u = nll_i(lg, db, eb).sum()
            if eta > 0 and tb is not None: u = u + eta*kd_i(lg, tb, db).sum()
            return u + 0.5*PREC*(w@w)
        pred = lambda w, xt: torch.sigmoid(glm_logits(w, xt))
    else:
        net = train_mlp(X, d, e, tl, eta, seed, STUDENT_H)
        feat = net.feats(xb).detach(); Hd = net.Hd; w0 = torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
        def logits_head(w, f): return f @ w[:K*Hd].view(K, Hd).T + w[K*Hd:][None, :]
        def U(w):
            lg = logits_head(w, feat); u = nll_i(lg, db, eb).sum()
            if eta > 0 and tb is not None: u = u + eta*kd_i(lg, tb, db).sum()
            return u + 0.5*PREC*(w@w)
        def pred(w, xt):
            f = net.feats(torch.as_tensor(xt, dtype=torch.float64)).detach(); return torch.sigmoid(logits_head(w, f))
    Hs = hessian(U, w0).detach(); p = w0.numel(); rg = 1e-6*float(torch.diagonal(Hs).abs().mean()) + 1e-8
    cov = torch.linalg.inv(Hs + rg*torch.eye(p, dtype=Hs.dtype)); cov = 0.5*(cov+cov.T)
    L = torch.linalg.cholesky(cov); g = torch.Generator().manual_seed(seed+1); z = torch.randn(LAP_M, p, generator=g, dtype=torch.float64)
    return [w0 + L@z[m] for m in range(LAP_M)], pred

def per_h(draws, truth):
    pm = draws.mean(0); Lq = np.quantile(draws, .025, 0); Uq = np.quantile(draws, .975, 0)
    return dict(pm=pm, cov=((truth >= Lq) & (truth <= Uq)).mean(0), width=(Uq-Lq).mean(0), psd=draws.std(0).mean(0),
                rmse=np.sqrt(((pm-truth)**2).mean(0)))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
    xt, dte, ete = simulate(N_TEST, FIXED_TEST_SEED); tcif = true_cif(xt); tlam = true_lam(xt)
    print(f"=== Gstage {TAG} === teacher={TEACHER_ARCH}{TEACHER_H if TEACHER_ARCH=='mlp' else ''} "
          f"student={STUDENT_ARCH}{STUDENT_H if STUDENT_ARCH=='mlp' else ''} | N={N} N_ext={N_EXT} R={R} sigma={SIGMA} etas={ETAS}", flush=True)
    # teacher quality on test (avg over R retrained teachers)
    tq = {k: [] for k in ["r2c", "rmse", "dev", "ibs", "c"]}
    teachers = [make_teacher(SEED0+r) for r in range(R)]
    for tf in teachers:
        lam = tf(xt); cif = 1-np.cumprod(1-lam, 1); surv = np.cumprod(1-lam, 1); iv = np.stack([lam, 1-lam], 1)
        tq["r2c"].append(1-((cif-tcif)**2).sum()/((tcif-tcif.mean())**2).sum()); tq["rmse"].append(np.sqrt(((cif-tcif)**2).mean()))
        tq["dev"].append(predictive_deviance(iv, dte, ete, reduction="mean")); tq["ibs"].append(float(integrated_brier_score(surv, dte, ete)))
        tq["c"].append(float(concordance_index(dte, ete, cif[:, -1])))
    print(f"    TEACHER on test: R2_CIF {np.mean(tq['r2c']):.4f} CIF-RMSE {np.mean(tq['rmse']):.4f} dev {np.mean(tq['dev']):.3f} "
          f"IBS {np.mean(tq['ibs']):.3f} C {np.mean(tq['c']):.3f}", flush=True)
    rows = []
    for eta in ETAS:
        acc = {k: [] for k in ["cov", "w", "rmse", "psd", "dev", "ibs", "c"]}; emp = []; hz = []
        for r in range(R):
            x, d, e = simulate(N, SEED0+r); tl = teachers[r](x) if eta > 0 else None
            draws, pred = student_draws(x, d, e, tl, eta, 1000*(SEED0+r))
            with torch.no_grad(): cifd = np.stack([1-np.cumprod(1-pred(w, xt).numpy(), 1) for w in draws]); lamd = np.stack([pred(w, xt).numpy() for w in draws])
            h = per_h(cifd, tcif); hz.append(h); emp.append(h["pm"])
            acc["cov"].append(h["cov"].mean()); acc["w"].append(h["width"].mean()); acc["rmse"].append(h["rmse"].mean()); acc["psd"].append(h["psd"].mean())
            mlam = lamd.mean(0); surv = np.cumprod(1-mlam, 1); iv = np.stack([mlam, 1-mlam], 1)
            acc["dev"].append(predictive_deviance(iv, dte, ete, reduction="mean")); acc["ibs"].append(float(integrated_brier_score(surv, dte, ete)))
            acc["c"].append(float(concordance_index(dte, ete, h["pm"][:, -1])))
        empsd = np.stack(emp).std(0).mean(); ratio = np.mean(acc["psd"])/max(empsd, 1e-9)
        np.savez(OUT_DIR/f"gstage_{TAG}_eta{eta:g}.npz",
                 cif_cov=np.stack([h["cov"] for h in hz]).mean(0), cif_width=np.stack([h["width"] for h in hz]).mean(0),
                 cif_rmse=np.stack([h["rmse"] for h in hz]).mean(0), cif_psd=np.stack([h["psd"] for h in hz]).mean(0))
        row = dict(tag=TAG, eta=eta, cov=np.mean(acc["cov"]), width=np.mean(acc["w"]), rmse=np.mean(acc["rmse"]),
                   post_emp=ratio, dev=np.mean(acc["dev"]), C=np.mean(acc["c"]), IBS=np.mean(acc["ibs"]))
        rows.append(row)
        print(f"  eta={eta:<4g} | CIF cov {row['cov']:.3f} w {row['width']:.3f} RMSE {row['rmse']:.4f} Post/Emp {row['post_emp']:.2f} "
              f"| dev {row['dev']:.3f} C {row['C']:.3f} IBS {row['IBS']:.3f}", flush=True)
    import pandas as pd; pd.DataFrame(rows).to_csv(OUT_DIR/f"gstage_{TAG}.csv", index=False)
    print(f"\nSaved to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
