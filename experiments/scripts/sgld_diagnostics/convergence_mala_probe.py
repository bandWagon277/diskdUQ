#!/usr/bin/env python
"""Convergence probe: eta=0 base model, plain FC-MLP, MALA from random init.

User's convergence recipe (2026-08-11): if the full model won't converge, first
SIMPLIFY THE STRUCTURE (plain fully-connected MLP, not the time-embedding /
transformer backbone), use MALA (exact, Metropolis-adjusted Langevin) from RANDOM
initialization (not warm-start-from-MAP), and walk a simplification ladder ---
current competing-risk settings -> homogeneous -> single-risk --- WITHOUT changing
the DGP / lambda rate (keep the original DiSKD squared rates).

This isolates "can the eta=0 posterior converge at all" from the teacher term.
MALA gives an EXACT stationary law, and random independent inits make R-hat a
genuine multi-basin convergence test (not a within-basin diagnostic).

Env: RISK={competing|single}, HIDDEN, HIDDEN_LAYERS, N (student), COV_SCALE
(homogeneity: <1 shrinks covariate spread, same rate form), MALA_ITERS, BURNIN,
THIN, N_CHAINS, STEP (initial; auto-adapted in burn-in to ~0.57 acceptance),
PRIOR_PREC, SEEDS.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call

from diskd import fit_time_grid, transform_durations
from diskd.networks import build_backbone
from diskd.losses import CompetingRiskNLLLoss, SingleRiskNLLLoss
from diskd.utils import competing_cif, competing_interval_probs
from diskd.uncertainty import gelman_rubin_rhat, effective_sample_size

DEVICE = os.environ.get("DEVICE", "cpu")
RISK = os.environ.get("RISK", "competing"); J = 2 if RISK == "competing" else 1
D = int(os.environ.get("D", 12)); K = int(os.environ.get("NUM_DURATIONS", 20))
N = int(os.environ.get("N", 500)); TEST_N = int(os.environ.get("TEST_N", 500))
HIDDEN = int(os.environ.get("HIDDEN", 16)); HIDDEN_LAYERS = int(os.environ.get("HIDDEN_LAYERS", 1))
COV_SCALE = float(os.environ.get("COV_SCALE", 1.0))               # homogeneity knob (same rate form)
CENSOR_MAX = float(os.environ.get("CENSOR_MAX", 0.05))
BETA_R1 = float(os.environ.get("BETA_R1", 2.0)); BETA_R2 = float(os.environ.get("BETA_R2", 2.0))
BETA_SHARED = float(os.environ.get("BETA_SHARED", 8.0))
MALA_ITERS = int(os.environ.get("MALA_ITERS", 4000)); BURNIN = int(os.environ.get("BURNIN", 2000))
THIN = int(os.environ.get("THIN", 5)); N_CHAINS = int(os.environ.get("N_CHAINS", 4))
STEP0 = float(os.environ.get("STEP", 1e-6)); PRIOR_PREC = float(os.environ.get("PRIOR_PREC", 1e-2))
# chain starting point: "random" (independent random weights) or "adam" (AdamW MAP + perturbation)
INIT = os.environ.get("INIT", "random"); ADAM_STEPS = int(os.environ.get("ADAM_STEPS", 400))
ADAM_LR = float(os.environ.get("ADAM_LR", 0.02)); ADAM_PERTURB = float(os.environ.get("ADAM_PERTURB", 0.05))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
FIXED_TEST_SEED = int(os.environ.get("FIXED_TEST_SEED", -1))     # >=0 -> same test set (estimand) for all replicates
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_convergence"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def rates(x):  # original DiSKD squared rates (unchanged); covariate spread scaled by COV_SCALE upstream
    z1 = x[:, 0:4].sum(1); z2 = x[:, 4:8].sum(1); z3 = x[:, 8:12].sum(1)
    r1 = np.clip((BETA_R1*z1)**2 + (BETA_SHARED*z3)**2, 1e-3, None)
    r2 = np.clip((BETA_R2*z2)**2 + (BETA_SHARED*z3)**2, 1e-3, None)
    return np.stack([r1, r2], axis=1)

def simulate(n, seed):
    rng = np.random.default_rng(seed); x = COV_SCALE*rng.normal(size=(n, D)); r = rates(x)
    t = rng.exponential(scale=1.0/r); cause = t.argmin(1); et = t.min(1)
    cens = rng.uniform(0.0, CENSOR_MAX, size=n); censored = cens < et
    dur = np.where(censored, cens, et); event = np.where(censored, 0, cause+1).astype(np.int64)
    if RISK == "single":
        event = (event > 0).astype(np.int64)                     # any event -> 1 (drop competing risk)
    df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(D)]); df["duration"] = dur.astype(float); df["event"] = event
    return df


def new_net(seed):
    torch.manual_seed(seed)
    net = build_backbone("mlp", in_features=D, num_durations=K, num_risks=J,
                         hidden_dim=HIDDEN, hidden_layers=HIDDEN_LAYERS, dropout=0.0).to(DEVICE).double()
    net.eval()
    return net


def make_U(net, x, idx, ev):
    names = [n for n, _ in net.named_parameters()]; shapes = [p.shape for p in net.parameters()]
    numels = [p.numel() for p in net.parameters()]
    lossfn = CompetingRiskNLLLoss() if RISK == "competing" else SingleRiskNLLLoss()
    def unflat(vec):
        out = {}; i = 0
        for n_, sh, k in zip(names, shapes, numels):
            out[n_] = vec[i:i+k].view(sh); i += k
        return out
    def logits_of(vec):
        o = functional_call(net, unflat(vec), (x,))              # [Ntest, J*K]
        return o.view(-1, J, K) if RISK == "competing" else o    # [N,J,K] or [N,K]
    def U(vec):
        lg = functional_call(net, unflat(vec), (x,))
        lg = lg.view(-1, J, K) if RISK == "competing" else lg
        return lossfn(lg, idx, ev, reduction="sum") + 0.5*PRIOR_PREC*(vec@vec)
    return U, logits_of, sum(numels)


def flat_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).double()


def adam_map_net(x, idx, ev, seed):
    """AdamW MAP on a fresh net: minimize NLL_sum + ridge -> returns the flat MAP weight vector."""
    net = new_net(seed)
    for p in net.parameters(): p.requires_grad_(True)
    lossfn = CompetingRiskNLLLoss() if RISK == "competing" else SingleRiskNLLLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=ADAM_LR, weight_decay=0.0)
    for _ in range(ADAM_STEPS):
        opt.zero_grad()
        out = net(x); lg = out.view(-1, J, K) if RISK == "competing" else out
        loss = lossfn(lg, idx, ev, reduction="sum") + 0.5*PRIOR_PREC*sum((p*p).sum() for p in net.parameters())
        loss.backward(); opt.step()
    return flat_params(net)


def mala(U, vec0, seed):
    torch.manual_seed(seed)
    vec = vec0.clone().requires_grad_(True)
    Uv = U(vec); g = torch.autograd.grad(Uv, vec)[0].detach(); Uv = float(Uv)
    step = STEP0; draws = []; acc = 0; win = 0
    for t in range(MALA_ITERS):
        prop = (vec.detach() - 0.5*step*g + step**0.5*torch.randn_like(vec)).requires_grad_(True)
        Up = U(prop); gp = torch.autograd.grad(Up, prop)[0].detach(); Up = float(Up)
        logq_fwd = -((prop.detach() - vec.detach() + 0.5*step*g)**2).sum()/(2*step)
        logq_bwd = -((vec.detach() - prop.detach() + 0.5*step*gp)**2).sum()/(2*step)
        logalpha = (-Up + Uv) + (logq_bwd - logq_fwd)
        if torch.log(torch.rand(())) < logalpha:
            vec = prop.detach().requires_grad_(True); Uv = Up; g = gp; acc += 1; win += 1
        else:
            vec = vec.detach().requires_grad_(True)
        # step adaptation during burn-in (target MALA acceptance ~0.55; aggressive so it can travel orders of magnitude)
        if t < BURNIN and (t+1) % 50 == 0:
            rate = win/50.0; win = 0
            if rate > 0.8:   step *= 2.0
            elif rate > 0.6: step *= 1.3
            elif rate < 0.3: step *= 0.5
            elif rate < 0.5: step *= 0.8
        if t >= BURNIN and (t-BURNIN) % THIN == 0:
            draws.append(vec.detach().clone())
    return draws, acc/MALA_ITERS, step


def cif_subj(logits):  # per-subject CIF [N,J,K] (competing) or [N,K] (single)
    if RISK == "competing":
        return competing_cif(competing_interval_probs(logits))                  # [N,J,K]
    haz = torch.sigmoid(logits)                                                  # [N,K]
    return 1.0 - torch.cumprod(1.0 - haz, dim=1)                                 # [N,K]

def haz_subj(logits):  # per-subject discrete HAZARD (lambda) [N,J,K] or [N,K]
    if RISK == "competing":
        return competing_interval_probs(logits)[:, :J, :]                        # [N,J,K]
    return torch.sigmoid(logits)                                                 # [N,K]

def true_cif_haz(x_np, cuts):
    """Closed-form truth on the test covariates. Returns (CIF, hazard), shapes match subj functionals."""
    r = rates(x_np); rt = r.sum(1, keepdims=True)                                # [N,J], [N,1]
    cuts = np.asarray(cuts); prev = np.concatenate([[0.0], cuts[:-1]]); delta = (cuts - prev)[None, :]
    if RISK == "competing":
        cif = (r/rt)[:, :, None] * (1.0 - np.exp(-rt*cuts[None, :]))[:, None, :]     # [N,J,K]
        haz = (r/rt)[:, :, None] * (1.0 - np.exp(-rt*delta))[:, None, :]             # [N,J,K]
    else:  # single risk: total event rate rt
        cif = 1.0 - np.exp(-rt*cuts[None, :])                                        # [N,K]
        haz = 1.0 - np.exp(-rt*delta)                                                # [N,K]
    return cif, haz

def coverage(draws, truth):
    """draws: [M, N, ...] pooled; truth: [N, ...]. Fraction of truths in the 95% credible interval."""
    L = np.quantile(draws, 0.025, axis=0); Uq = np.quantile(draws, 0.975, axis=0)
    return float(((truth >= L) & (truth <= Uq)).mean()), float((Uq - L).mean())

def r2_overall(draws, truth):
    """R^2 of the posterior-mean fit vs closed-form truth over all (subject, [cause,] horizon)."""
    pm = draws.mean(0)
    return float(1.0 - ((pm - truth)**2).sum() / max(((truth - truth.mean())**2).sum(), 1e-12))

def horizon_diag(draws, truth):
    """Per-horizon bias/variance decomposition. draws [M,N,...,K], truth [N,...,K].
    Reduces over subjects (and causes) -> each returned array is length K."""
    postmean = draws.mean(0); postsd = draws.std(0)
    L = np.quantile(draws, 0.025, axis=0); Uq = np.quantile(draws, 0.975, axis=0)
    inside = ((truth >= L) & (truth <= Uq)).astype(np.float64)
    err = postmean - truth
    red = tuple(range(truth.ndim - 1))                           # all axes except the last (horizon)
    # per-horizon R^2 of the posterior-mean fit vs truth (explained subject-level variation)
    tmean = truth.mean(red, keepdims=True)
    ss_res = (err**2).sum(red); ss_tot = ((truth - tmean)**2).sum(red)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return dict(post_sd=postsd.mean(red), post_var=(postsd**2).mean(red),
                bias=err.mean(red), abs_bias=np.abs(err).mean(red), rmse=np.sqrt((err**2).mean(red)),
                cov=inside.mean(red), width=(Uq - L).mean(red), r2=r2, tvar=(truth.var(red)))

def rhat_ess(A):     # A: [C, ndraw, *dims] -> worst R-hat, min ESS over all components
    Af = A.reshape(A.shape[0], A.shape[1], int(np.prod(A.shape[2:])))
    rr, ee = [], []
    for j in range(Af.shape[2]):
        try: rr.append(float(gelman_rubin_rhat(Af[:, :, j]))); ee.append(float(effective_sample_size(Af[:, :, j])))
        except Exception: pass
    return (float(np.nanmax(rr)) if rr else float("nan")), (float(np.nanmin(ee)) if ee else float("nan"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    print(f"=== MALA convergence probe === RISK={RISK} struct=MLP(hidden={HIDDEN}x{HIDDEN_LAYERS}) "
          f"cov_scale={COV_SCALE} | {N_CHAINS} chains x {MALA_ITERS} iters (burn {BURNIN}), random init, "
          f"eta=0 | seeds={SEEDS}", flush=True)
    rows = []; hz_rows = []
    for seed in SEEDS:
        # FIXED_TEST_SEED: hold the test set (estimand) constant across training replicates so that
        # coverage across `seeds` is a proper frequentist coverage over R independent posteriors.
        df = simulate(N, 10*seed+2)
        test_df = simulate(TEST_N, FIXED_TEST_SEED if FIXED_TEST_SEED >= 0 else 10*seed+3)
        tg = fit_time_grid(df["duration"].values, K)
        x = torch.as_tensor(df[[f"x{i+1}" for i in range(D)]].to_numpy(), dtype=torch.float64, device=DEVICE)
        xt = torch.as_tensor(test_df[[f"x{i+1}" for i in range(D)]].to_numpy(), dtype=torch.float64, device=DEVICE)
        idx = torch.as_tensor(transform_durations(df["duration"].values, tg), dtype=torch.long, device=DEVICE)
        ev = torch.as_tensor(df["event"].values.copy(), dtype=torch.long, device=DEVICE)
        net = new_net(seed); U, logits_of, P = make_U(net, x, idx, ev)
        _, logits_test, _ = make_U(net, xt, idx, ev)             # reuse net for test logits
        tcif, thaz = true_cif_haz(test_df[[f"x{i+1}" for i in range(D)]].to_numpy(), np.asarray(tg.cuts))
        cif_cm, haz_cm = [], []                                  # per-chain cohort-mean (for R-hat)
        cif_ps, haz_ps = [], []                                  # pooled per-subject draws (for coverage)
        map_vec = adam_map_net(x, idx, ev, seed) if INIT == "adam" else None
        accs = []; laststep = STEP0
        for c in range(N_CHAINS):
            if INIT == "adam":                                   # AdamW MAP + small perturbation per chain
                torch.manual_seed(1000*seed + c); vec0 = map_vec + ADAM_PERTURB*torch.randn_like(map_vec)
            else:                                                # INDEPENDENT random init per chain
                vec0 = flat_params(new_net(1000*seed + c))
            draws, acc, step = mala(U, vec0, 7*seed + c); accs.append(acc); laststep = step
            cs = np.stack([cif_subj(logits_test(th)).detach().cpu().numpy().astype(np.float32) for th in draws])
            hs = np.stack([haz_subj(logits_test(th)).detach().cpu().numpy().astype(np.float32) for th in draws])
            cif_cm.append(cs.mean(1)); haz_cm.append(hs.mean(1))  # cohort mean over subjects -> [ndraw,...]
            cif_ps.append(cs); haz_ps.append(hs)
        rhat_c, ess_c = rhat_ess(np.stack(cif_cm))               # CIF convergence (cohort-mean functional)
        rhat_h, ess_h = rhat_ess(np.stack(haz_cm))               # lambda (hazard) convergence
        pooled_cif = np.concatenate(cif_ps, 0); pooled_haz = np.concatenate(haz_ps, 0)
        cov_c, wid_c = coverage(pooled_cif, tcif)                # 95% CI coverage vs closed-form truth
        cov_h, wid_h = coverage(pooled_haz, thaz)
        r2_c = r2_overall(pooled_cif, tcif); r2_h = r2_overall(pooled_haz, thaz)   # fit quality
        hz_rows.append(dict(seed=seed, cif=horizon_diag(pooled_cif, tcif), haz=horizon_diag(pooled_haz, thaz)))
        rec = dict(seed=seed, K=K, N=N, params=P, rhat_cif=rhat_c, ess_cif=ess_c, rhat_haz=rhat_h, ess_haz=ess_h,
                   cov_cif=cov_c, wid_cif=wid_c, cov_haz=cov_h, wid_haz=wid_h, r2_cif=r2_c, r2_haz=r2_h,
                   acc=float(np.mean(accs)))
        rows.append(rec)
        print(f"  seed{seed}: p={P} | CIF: R-hat={rhat_c:.2f} cov95={cov_c:.2f} R2={r2_c:.3f} | "
              f"lambda: R-hat={rhat_h:.2f} cov95={cov_h:.2f} R2={r2_h:.3f} | acc={np.mean(accs):.2f}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{RISK}_K{K}_N{N}_h{HIDDEN}x{HIDDEN_LAYERS}_cov{COV_SCALE}"
    df.to_csv(OUT_DIR/f"convergence_{tag}.csv", index=False)
    # per-horizon bias/variance dump (mean over seeds) for CIF and lambda
    keys = ["post_sd", "post_var", "bias", "abs_bias", "rmse", "cov", "width", "r2", "tvar"]
    npz = {"cuts": np.asarray(fit_time_grid(simulate(N, 10*SEEDS[0]+2)["duration"].values, K).cuts)}
    for fn in ["cif", "haz"]:
        for k in keys:
            npz[f"{fn}_{k}"] = np.stack([r[fn][k] for r in hz_rows]).mean(0)   # [K]
    np.savez(OUT_DIR/f"horizon_{tag}.npz", **npz)
    conv_c = int((df['rhat_cif'] < 1.1).sum()); conv_h = int((df['rhat_haz'] < 1.1).sum())
    R = len(df); ft = "fixed" if FIXED_TEST_SEED >= 0 else "per-seed"
    print(f"\n=== {RISK} K={K} N={N} MLP({HIDDEN}x{HIDDEN_LAYERS}) cov={COV_SCALE} | R={R} replicates ({ft} test): "
          f"CIF R-hat {df['rhat_cif'].mean():.2f} conv {conv_c}/{R} cov95 {df['cov_cif'].mean():.3f}+/-{df['cov_cif'].std():.3f} R2 {df['r2_cif'].mean():.3f} | "
          f"lambda R-hat {df['rhat_haz'].mean():.2f} conv {conv_h}/{R} cov95 {df['cov_haz'].mean():.3f}+/-{df['cov_haz'].std():.3f} R2 {df['r2_haz'].mean():.3f} | "
          f"acc={df['acc'].mean():.2f} ===", flush=True)
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
