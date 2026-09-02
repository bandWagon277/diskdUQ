"""Convergence-meaningful re-run of the clean-DGP diagnostic.

Fixes the two reliability caveats so that R-hat is trustworthy:
  (1) DECAYING step size (canonical SGLD): eps_t = a (b+t)^{-gamma}, tunable gamma,
      instead of a fixed epsilon (which samples a biased diffusion).
  (2) OVER-DISPERSED / independent initialization: each chain starts from the AdamW
      MAP plus an independent Gaussian perturbation (INIT_PERTURB), so R-hat measures
      real between-chain mixing, not agreement within one basin. Set COLD_START=1 for
      fully random initialization (the strictest test).
  Plus a long budget with burn-in.

R-hat / ESS are reported per functional over ALL horizons and both causes (cohort-mean
CIF), and the worst value is the headline convergence number. Only if R-hat < 1.1 do we
treat the coverage / PostSD / under-dispersion numbers as meaningful.

Outputs -> responses_converged/.
"""
from __future__ import annotations

import copy, os, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from diskd import DiSKDStudent, DiscreteSurvivalModel, fit_time_grid, transform_durations
from diskd.uncertainty import _iter_samples, effective_sample_size, gelman_rubin_rhat
from diskd.networks import sinusoidal_time_embedding
from diskd.losses import CompetingRiskNLLLoss
from diskd.utils import (competing_cif, competing_interval_probs, full_probs_from_event_probs,
                         make_at_risk_mask, temperature_scale_probs)

ARMS = os.environ.get("ARMS", "full,lastlayer").split(",")

D = int(os.environ.get("D", 6)); J = 2
K = int(os.environ.get("NUM_DURATIONS", 20))
TEACHER_N = int(os.environ.get("TEACHER_N", 5000)); STUDENT_N = int(os.environ.get("STUDENT_N", 1000))
TEST_N = int(os.environ.get("TEST_N", 1000)); HID = int(os.environ.get("HIDDEN", 32)); BATCH = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100)); ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 60))
SNR_B = float(os.environ.get("SNR_B", 0.5)); BASE_RATE = float(os.environ.get("BASE_RATE", 1.0))
CENSOR_RATE = float(os.environ.get("CENSOR_RATE", 0.3))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,1").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
# --- convergence-meaningful sampler config ---
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 2000)); BURNIN = int(os.environ.get("BURNIN", 1000))
N_CHAINS = int(os.environ.get("N_CHAINS", 6)); SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 1e-3)); EPS_T = float(os.environ.get("EPS_T", 1e-5))
GAMMA = float(os.environ.get("GAMMA", 0.55)); DRIFT = os.environ.get("DRIFT", "welling_teh")
INIT_PERTURB = float(os.environ.get("INIT_PERTURB", 0.1)); COLD_START = int(os.environ.get("COLD_START", 0))
R_REP = int(os.environ.get("R_REP", 15)); Z = 1.959963985
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_converged"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


# DGP mode: "loglinear" = clean homogeneous PH DGP (our modification);
#           "squared"   = ORIGINAL DiSKD DGP, r_j = (beta_j z_j)^2 + (beta_sh z3)^2 (simulation.py:45).
DGP = os.environ.get("DGP", "loglinear")
CENSOR_MAX = float(os.environ.get("CENSOR_MAX", 0.05))
BETA_R1 = float(os.environ.get("BETA_R1", 2.0)); BETA_R2 = float(os.environ.get("BETA_R2", 2.0))
BETA_SHARED = float(os.environ.get("BETA_SHARED", 8.0))

def dgp_params():
    if DGP == "squared":
        return np.array([BETA_R1, BETA_R2]), np.array([BETA_SHARED])
    rng = np.random.default_rng(1234); return np.log(BASE_RATE) + np.zeros(J), SNR_B * rng.normal(size=(J, D))
def rates(x, a, b):
    if DGP == "squared":  # original DiSKD squared rates; covariate groups fixed as in simulation.py
        z1 = x[:, 0:4].sum(1); z2 = x[:, 4:8].sum(1); z3 = x[:, 8:12].sum(1)
        r1 = np.clip((a[0]*z1)**2 + (b[0]*z3)**2, 1e-3, None)
        r2 = np.clip((a[1]*z2)**2 + (b[0]*z3)**2, 1e-3, None)
        return np.stack([r1, r2], axis=1)
    return np.exp(a[None, :] + x @ b.T)
def simulate(n, a, b, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); r = rates(x, a, b)
    t = rng.exponential(scale=1.0/r); cause = t.argmin(1); et = t.min(1)
    if DGP == "squared":  # original DiSKD uses uniform censoring on [0, censor_max]
        cens = rng.uniform(0.0, CENSOR_MAX, size=n)
    else:
        cens = rng.exponential(scale=1.0/max(CENSOR_RATE, 1e-6), size=n)
    censored = cens < et
    dur = np.where(censored, cens, et); event = np.where(censored, 0, cause+1).astype(np.int64)
    df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(D)]); df["duration"] = dur.astype(float); df["event"] = event
    return df
def true_cif(df, cuts, a, b):
    x = df[[f"x{i+1}" for i in range(D)]].to_numpy(); r = rates(x, a, b); rt = r.sum(1, keepdims=True)
    return (r/rt)[:, :, None] * (1.0-np.exp(-rt*np.asarray(cuts)[None, :]))[:, None, :]


def build_sgld(teacher, eta):
    common = dict(num_risks=J, num_durations=K, hidden_dim=HID, epochs=SGLD_EPOCHS, batch_size=BATCH,
                  device=DEVICE, optimizer="sgld", sgld_step_size=EPS0, sgld_final_step_size=EPS_T,
                  sgld_gamma=GAMMA, sgld_drift_mode=DRIFT, sgld_noise_scale=1.0, sgld_burnin_epochs=BURNIN,
                  sgld_samples_per_chain=SAMPLES_PER_CHAIN)
    return DiscreteSurvivalModel(**common) if eta == 0 else DiSKDStudent(
        teacher_model=teacher, teacher_type="competing", eta=eta, temperature=2.0, **common)

def fit_adamw(df, teacher, tg, eta, seed, feats):
    torch.manual_seed(seed); np.random.seed(seed)
    common = dict(num_risks=J, num_durations=K, hidden_dim=HID, epochs=ADAMW_EPOCHS, batch_size=BATCH,
                  device=DEVICE, optimizer="adamw")
    m = DiscreteSurvivalModel(**common) if eta == 0 else DiSKDStudent(
        teacher_model=teacher, teacher_type="competing", eta=eta, temperature=2.0, **common)
    m.time_grid = tg; m.fit(df, feature_cols=feats); return m


def run_chains(teacher, tg, student_df, test_df, eta, mapstate, seed, feats):
    """Over-dispersed (or cold) multi-chain SGLD with the decaying schedule. Returns
    pooled per-draw CIF, per-chain cohort-mean CIF functionals, posterior-mean hazard."""
    all_draws, chain_fn, hz_sum, ndraw = [], [], None, 0
    for c in range(N_CHAINS):
        m = build_sgld(teacher, eta); m.time_grid = tg
        torch.manual_seed(100*seed + c); np.random.seed(100*seed + c)
        if not COLD_START:
            pert = {k: v + INIT_PERTURB * torch.randn_like(v.float()).to(v.dtype) for k, v in mapstate.items()}
            m._warm_start_state = pert                         # over-dispersed warm start
        m.fit(student_df, feature_cols=feats)                  # COLD_START -> random init (no warm state)
        cc = []
        for _, mm in _iter_samples(m, m.posterior_samples):
            cif = np.asarray(mm.predict_cif(test_df)); all_draws.append(cif); cc.append(cif)
            h = np.asarray(mm.predict_hazard(test_df)); hz_sum = h if hz_sum is None else hz_sum + h; ndraw += 1
        chain_fn.append(np.stack(cc).mean(1))                  # [ndraw, J, K] cohort-mean CIF this chain
    return np.stack(all_draws), np.stack(chain_fn), hz_sum / ndraw


# ---------- LAST-LAYER SGLD (frozen backbone; 66-dim posterior that can converge) ----------

def backbone_feats(net, x):
    net.eval()
    with torch.no_grad():
        h = net.feature_projection(x).unsqueeze(1)
        emb = sinusoidal_time_embedding(net.num_durations, h.shape[-1], device=x.device, dtype=x.dtype)
        h = net.blocks(h + emb.unsqueeze(0))
    return h.double()                                          # [N,K,HID]

def head_fns(feat, idx, ev, teacher_full, eta):
    Hd = HID; nll = CompetingRiskNLLLoss(); mask = make_at_risk_mask(idx, K).double()
    def logits_of(th, f):
        W = th[:J*Hd].view(J, Hd); bb = th[J*Hd:]
        return torch.einsum("nkd,jd->njk", f, W) + bb.view(1, J, 1)
    def sum_loss(th):                                          # -log of the omega=1 generalized posterior
        lg = logits_of(th, feat); r = nll(lg, idx, ev, reduction="sum")
        if eta == 0: return r
        T = 2.0; tt = temperature_scale_probs(teacher_full, T); logt = tt.clamp_min(1e-12).log()
        st = temperature_scale_probs(competing_interval_probs(lg), T)
        kl = (tt * (logt - st.clamp_min(1e-12).log())).sum(1)
        return r + eta * (T * T) * (kl * mask).sum(1).sum()
    return logits_of, sum_loss

def decay_eps(t, T, eps0, epsT, gamma):
    bb = T / max((eps0/epsT) ** (1.0/gamma) - 1.0, 1e-9); aa = eps0 * (bb ** gamma)
    return aa * (bb + t) ** (-gamma)

def sgld_lastlayer(sum_loss, theta0, seed):
    torch.manual_seed(seed)
    th = theta0.clone(); draws = []
    total = SGLD_EPOCHS; thin = max(1, (total - BURNIN) // SAMPLES_PER_CHAIN)
    for t in range(total):
        eps = decay_eps(t, total, EPS0, EPS_T, GAMMA)
        thr = th.clone().requires_grad_(True)
        g = torch.autograd.grad(sum_loss(thr), thr)[0]
        th = (th - 0.5 * eps * g + (eps ** 0.5) * torch.randn_like(th)).detach()
        if t >= BURNIN and (t - BURNIN) % thin == 0 and len(draws) < SAMPLES_PER_CHAIN:
            draws.append(th.clone())
    return draws

def run_lastlayer(adam, teacher, tg, student_df, test_df, eta, seed):
    net = adam.net
    xtr = torch.as_tensor(adam.preprocessor.transform(student_df), dtype=torch.float32, device=DEVICE)
    idx = torch.as_tensor(transform_durations(student_df["duration"].values, tg), dtype=torch.long, device=DEVICE)
    ev = torch.as_tensor(student_df["event"].values.copy(), dtype=torch.long, device=DEVICE)
    tp = teacher.predict_interval_probs(student_df)[:, :J, :]
    tfull = full_probs_from_event_probs(torch.as_tensor(tp, dtype=torch.float64, device=DEVICE))
    feat = backbone_feats(net, xtr); fte = backbone_feats(net, torch.as_tensor(
        adam.preprocessor.transform(test_df), dtype=torch.float32, device=DEVICE))
    logits_of, sum_loss = head_fns(feat, idx, ev, tfull, eta)
    theta_map = torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
    Nt = fte.shape[0]; all_draws, chain_fn = [], []
    for c in range(N_CHAINS):
        th0 = theta_map + (0.0 if COLD_START else INIT_PERTURB) * torch.randn_like(theta_map)
        draws = sgld_lastlayer(sum_loss, th0, 100*seed + c)
        cc = []
        for th in draws:
            cif = competing_cif(competing_interval_probs(logits_of(th, fte))).detach().cpu().numpy()  # [Nt,J,K]
            all_draws.append(cif); cc.append(cif)
        chain_fn.append(np.stack(cc).mean(1))
    return np.stack(all_draws), np.stack(chain_fn)


def rhat_ess_allhorizons(chain_fn):
    """chain_fn [nchains, ndraw, J, K]; worst R-hat and min ESS over all (cause,horizon)."""
    rr, ee = [], []
    for j in range(J):
        for k in range(K):
            A = chain_fn[:, :, j, k]
            try: rr.append(float(gelman_rubin_rhat(A))); ee.append(float(effective_sample_size(A)))
            except Exception: pass
    return (float(np.nanmax(rr)) if rr else float("nan")), (float(np.nanmin(ee)) if ee else float("nan"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    feats = [f"x{i+1}" for i in range(D)]; a, b = dgp_params()
    init = "COLD (random)" if COLD_START else f"over-dispersed warm (perturb={INIT_PERTURB})"
    print(f"=== Convergence-meaningful probe (K={K}) === decay eps {EPS0:.0e}->{EPS_T:.0e} gamma={GAMMA}, "
          f"{N_CHAINS} chains x {SGLD_EPOCHS}ep ({BURNIN} burnin), init={init}, etas={ETAS} seeds={SEEDS}", flush=True)
    rows = []
    for seed in SEEDS:
        teacher_df = simulate(TEACHER_N, a, b, 10*seed+1); student_df = simulate(STUDENT_N, a, b, 10*seed+2)
        test_df = simulate(TEST_N, a, b, 10*seed+3)
        tg = fit_time_grid(np.concatenate([teacher_df["duration"].values, student_df["duration"].values]), K)
        cuts = np.asarray(tg.cuts); tcif = true_cif(test_df, cuts, a, b)
        teacher = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=HID, epochs=TEACHER_EPOCHS,
                                        batch_size=BATCH, device=DEVICE, time_grid=tg).fit(teacher_df, feature_cols=feats)
        risk = tcif[:, :, -1].mean(1); order = np.argsort(risk); n = len(order)
        strata = {"low": order[:int(0.2*n)], "mid": order[int(0.2*n):int(0.8*n)], "high": order[int(0.8*n):]}
        for eta in ETAS:
            adam = fit_adamw(student_df, teacher, tg, eta, 1000*seed, feats)
            mapstate = {k: v.detach().clone() for k, v in adam.net.state_dict().items()}
            reps = [np.asarray(fit_adamw(simulate(STUDENT_N, a, b, 500000+1000*seed+r), teacher, tg, eta, 7000+r, feats).predict_cif(test_df))
                    for r in range(R_REP)]
            emp_sd = np.stack(reps).std(0)
            for arm in ARMS:
                t0 = time.time()
                if arm == "full":
                    draws, chain_fn, _ = run_chains(teacher, tg, student_df, test_df, eta, mapstate, seed, feats)
                else:
                    draws, chain_fn = run_lastlayer(adam, teacher, tg, student_df, test_df, eta, seed)
                q = np.quantile(draws, [0.025, 0.5, 0.975], axis=0); L, med, U = q[0], q[1], q[2]
                post_sd = draws.std(0); rhat, ess = rhat_ess_allhorizons(chain_fn)
                inside = (tcif >= L) & (tcif <= U)
                Ls = np.clip(med + 1.2*(L-med), 0, 1); Us = np.clip(med + 1.2*(U-med), 0, 1)
                cov12 = float(((tcif >= Ls) & (tcif <= Us)).mean())
                low = strata["low"]
                rec = dict(seed=seed, eta=eta, arm=arm, rhat=rhat, ess=ess, cov=float(inside.mean()),
                           cov_front=float(inside[:, :, :3].mean()), cov12=cov12,
                           ratio=float(post_sd.mean()/max(emp_sd.mean(), 1e-9)), bias=float((med-tcif).mean()),
                           low_front=float(inside[low][:, :, :3].mean()), low_missb=float((tcif[low] < L[low]).mean()))
                rows.append(rec)
                print(f"  seed{seed} eta={eta} [{arm}]: Rhat(worst)={rhat:.2f} ESS={ess:.0f} | "
                      f"cov={rec['cov']:.3f} (1.2x->{cov12:.3f}) ratio={rec['ratio']:.2f} "
                      f"low_front={rec['low_front']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR/"converged.csv", index=False)
    md = ["# Convergence-meaningful re-run (decaying step + over-dispersed init)", ""]
    md.append(f"Clean DGP, K={K}. SGLD: eps {EPS0:.0e}->{EPS_T:.0e}, gamma={GAMMA}, {N_CHAINS} chains x "
              f"{SGLD_EPOCHS} epochs ({BURNIN} burn-in), init = {init}. R-hat/ESS are the WORST/MIN over all "
              f"{J}x{K} cohort-mean CIF functionals. Arms: full-network vs last-layer (frozen backbone). "
              f"Seeds {SEEDS}. Convergence threshold R-hat<1.1.")
    md.append("")
    md.append("| arm | eta | R-hat worst (mean+/-SD) | ESS min | coverage | cov 1.2x | PostSD/EmpSD | low-risk front cov |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for eta in ETAS:
            s = df[(df.eta == eta) & (df.arm == arm)]
            if len(s) == 0: continue
            md.append(f"| {arm} | {eta:g} | {s['rhat'].mean():.2f} +/- {s['rhat'].std():.2f} | {s['ess'].mean():.0f} | "
                      f"{s['cov'].mean():.3f} +/- {s['cov'].std():.3f} | {s['cov12'].mean():.3f} | "
                      f"{s['ratio'].mean():.2f} +/- {s['ratio'].std():.2f} | {s['low_front'].mean():.3f} |")
    md.append("")
    for arm in ARMS:
        s = df[df.arm == arm]; nconv = (s['rhat'] < 1.1).sum()
        verdict = ("**converges** -> its coverage / PostSD / under-dispersion numbers are MEANINGFUL."
                   if nconv > len(s)/2 else
                   "does **NOT** converge even with the decaying schedule + over-dispersed init.")
        md.append(f"- **{arm}:** {nconv}/{len(s)} cells reach R-hat<1.1; {verdict}")
    md.append("")
    md.append("Per-seed rows: `converged.csv`.")
    (OUT_DIR/"converged_report.md").write_text("\n".join(md)+"\n")
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
