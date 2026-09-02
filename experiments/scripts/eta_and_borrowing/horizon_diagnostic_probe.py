"""Per-horizon bias-vs-variance-vs-propagation diagnostic (K=20, clean DGP).

Implements the diagnostic framework:
  For each horizon t_k report
    F0(t_k)  (true CIF),  Fhat(t_k) (posterior mean),  [L(t_k), U(t_k)] (95% interval),
    Coverage(t_k),  Width(t_k),  Bias(t_k)=E[Fhat-F0],
    miss_below = P{F0 < L}  (=> risk OVER-estimated),
    miss_above = P{F0 > U}  (=> risk UNDER-estimated),
    EmpSD(t_k)  = SD of Fhat across REPEATED simulated datasets (estimator sampling SD),
    PostSD(t_k) = posterior SD WITHIN one fitted dataset.

Verdicts:
  - mostly F0<L (miss_below dominant) + Bias>0  -> systematic early-risk OVER-estimation (A: bias)
  - mostly F0>U (miss_above dominant) + Bias<0  -> systematic UNDER-estimation (A: bias)
  - misses both sides + PostSD < EmpSD          -> UNDER-DISPERSION (B: variance)
  - Width & PostSD increase with horizon        -> cumulative-propagation pattern (C)

Uses the clean log-linear exponential competing-risk DGP (closed-form CIF/hazard),
homogeneous teacher/student, welling_teh SGLD. Outputs to responses_horizon/.
"""
from __future__ import annotations

import os, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from diskd import (DiSKDStudent, DiscreteSurvivalModel, WarmStartMultiChainSampler,
                   fit_time_grid, transform_durations)
from diskd.uncertainty import _iter_samples, effective_sample_size, gelman_rubin_rhat

D = int(os.environ.get("D", 6)); J = 2
K = int(os.environ.get("NUM_DURATIONS", 20))
TEACHER_N = int(os.environ.get("TEACHER_N", 5000)); STUDENT_N = int(os.environ.get("STUDENT_N", 1000))
TEST_N = int(os.environ.get("TEST_N", 1000)); HID = int(os.environ.get("HIDDEN", 32)); BATCH = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100)); ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 60))
SNR_B = float(os.environ.get("SNR_B", 0.5)); BASE_RATE = float(os.environ.get("BASE_RATE", 1.0))
CENSOR_RATE = float(os.environ.get("CENSOR_RATE", 0.3))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,1").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",")]
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500)); N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500)); EPS = float(os.environ.get("EPS", 2e-4))
R_REP = int(os.environ.get("R_REP", 20)); DRIFT = os.environ.get("DRIFT", "welling_teh")
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_horizon"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


# DGP mode: "loglinear" = clean homogeneous PH DGP; "squared" = ORIGINAL DiSKD DGP
# r_j = (beta_j z_j)^2 + (beta_sh z3)^2 (simulation.py:45), identical to writeup Section 1.3.
DGP = os.environ.get("DGP", "loglinear")
CENSOR_MAX = float(os.environ.get("CENSOR_MAX", 0.05))
BETA_R1 = float(os.environ.get("BETA_R1", 2.0)); BETA_R2 = float(os.environ.get("BETA_R2", 2.0))
BETA_SHARED = float(os.environ.get("BETA_SHARED", 8.0))

def dgp_params():
    if DGP == "squared":
        return np.array([BETA_R1, BETA_R2]), np.array([BETA_SHARED])
    rng = np.random.default_rng(1234)
    return np.log(BASE_RATE) + np.zeros(J), SNR_B * rng.normal(size=(J, D))

def rates(x, a, b):
    if DGP == "squared":  # original DiSKD squared rates; covariate groups fixed as in simulation.py
        z1 = x[:, 0:4].sum(1); z2 = x[:, 4:8].sum(1); z3 = x[:, 8:12].sum(1)
        r1 = np.clip((a[0]*z1)**2 + (b[0]*z3)**2, 1e-3, None)
        r2 = np.clip((a[1]*z2)**2 + (b[0]*z3)**2, 1e-3, None)
        return np.stack([r1, r2], axis=1)
    return np.exp(a[None, :] + x @ b.T)

def simulate(n, a, b, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); r = rates(x, a, b)
    t = rng.exponential(scale=1.0 / r); cause = t.argmin(1); et = t.min(1)
    if DGP == "squared":  # original DiSKD uses uniform censoring on [0, censor_max]
        cens = rng.uniform(0.0, CENSOR_MAX, size=n)
    else:
        cens = rng.exponential(scale=1.0 / max(CENSOR_RATE, 1e-6), size=n)
    censored = cens < et; dur = np.where(censored, cens, et)
    event = np.where(censored, 0, cause + 1).astype(np.int64)
    df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(D)]); df["duration"] = dur.astype(float); df["event"] = event
    return df

def true_cif(df, cuts, a, b):
    x = df[[f"x{i+1}" for i in range(D)]].to_numpy(); r = rates(x, a, b); rt = r.sum(1, keepdims=True)
    return (r / rt)[:, :, None] * (1.0 - np.exp(-rt * np.asarray(cuts)[None, :]))[:, None, :]

def true_hazard(df, cuts, a, b):
    """Discrete interval hazard lambda_j(tau_k|z) = (r_j/r_tot)(1-exp(-r_tot*Delta_k))."""
    x = df[[f"x{i+1}" for i in range(D)]].to_numpy(); r = rates(x, a, b); rt = r.sum(1, keepdims=True)
    cuts = np.asarray(cuts); prev = np.concatenate([[0.0], cuts[:-1]]); delta = (cuts - prev)[None, :]
    return (r / rt)[:, :, None] * (1.0 - np.exp(-rt * delta))[:, None, :]


def fit_student(df, teacher, tg, eta, seed, optimizer, epochs, feats):
    torch.manual_seed(seed); np.random.seed(seed)
    common = dict(num_risks=J, num_durations=K, hidden_dim=HID, epochs=epochs, batch_size=BATCH,
                  device=DEVICE, optimizer=optimizer)
    if optimizer == "sgld":
        common.update(sgld_step_size=EPS, sgld_final_step_size=EPS, sgld_gamma=0.55, sgld_drift_mode=DRIFT,
                      sgld_noise_scale=1.0, sgld_burnin_epochs=0, sgld_samples_per_chain=SAMPLES_PER_CHAIN)
    m = DiscreteSurvivalModel(**common) if eta == 0 else DiSKDStudent(
        teacher_model=teacher, teacher_type="competing", eta=eta, temperature=2.0, **common)
    m.time_grid = tg; m.fit(df, feature_cols=feats); return m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    feats = [f"x{i+1}" for i in range(D)]; a, b = dgp_params()
    print(f"=== Per-horizon diagnostic (K={K}, DGP={DGP}, {DRIFT}) === etas={ETAS} seeds={SEEDS} R={R_REP}")
    print(f"CHAINS={N_CHAINS}, init=warm-start (ALL chains from the same AdamW MAP) -> R-hat is a "
          f"WITHIN-BASIN mixing diagnostic, not full-posterior exploration.", flush=True)
    agg = {}; med0_store = {}
    for seed in SEEDS:
        teacher_df = simulate(TEACHER_N, a, b, 10*seed+1); student_df = simulate(STUDENT_N, a, b, 10*seed+2)
        test_df = simulate(TEST_N, a, b, 10*seed+3)
        tg = fit_time_grid(np.concatenate([teacher_df["duration"].values, student_df["duration"].values]), K)
        cuts = np.asarray(tg.cuts); tcif = true_cif(test_df, cuts, a, b)               # [N,J,K]
        thz = true_hazard(test_df, cuts, a, b)                                          # [N,J,K] true discrete hazard
        teacher = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=HID, epochs=TEACHER_EPOCHS,
                                        batch_size=BATCH, device=DEVICE, time_grid=tg).fit(teacher_df, feature_cols=feats)
        for eta in ETAS:
            t0 = time.time()
            adam = fit_student(student_df, teacher, tg, eta, 1000*seed, "adamw", ADAMW_EPOCHS, feats)
            pretrained = {k: v.clone() for k, v in adam.net.state_dict().items()}
            # SGLD posterior on the main fit
            bs = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=HID, epochs=SGLD_EPOCHS, batch_size=BATCH,
                    device=DEVICE, optimizer="sgld", sgld_step_size=EPS, sgld_final_step_size=EPS, sgld_gamma=0.55,
                    sgld_drift_mode=DRIFT, sgld_noise_scale=1.0, sgld_burnin_epochs=0, sgld_samples_per_chain=SAMPLES_PER_CHAIN) \
                 if eta == 0 else DiSKDStudent(teacher_model=teacher, teacher_type="competing", eta=eta, temperature=2.0,
                    num_risks=J, num_durations=K, hidden_dim=HID, epochs=SGLD_EPOCHS, batch_size=BATCH, device=DEVICE,
                    optimizer="sgld", sgld_step_size=EPS, sgld_final_step_size=EPS, sgld_gamma=0.55, sgld_drift_mode=DRIFT,
                    sgld_noise_scale=1.0, sgld_burnin_epochs=0, sgld_samples_per_chain=SAMPLES_PER_CHAIN)
            bs.time_grid = tg
            sampler = WarmStartMultiChainSampler(bs, pretrained, n_chains=N_CHAINS,
                                                 seeds=[100*seed+c for c in range(N_CHAINS)]).fit(student_df, feature_cols=feats)
            lead = sampler.lead_chain; draws = []; hz_sum = None; ndraw = 0
            fn1, fn2, fnh = [], [], []                                                 # per-chain,per-draw functionals
            for chain in sampler.chains:
                g1, g2, gh = [], [], []
                for _, m in _iter_samples(lead, chain.posterior_samples):
                    c = np.asarray(m.predict_cif(test_df)); draws.append(c)
                    h = np.asarray(m.predict_hazard(test_df)); hz_sum = h if hz_sum is None else hz_sum + h; ndraw += 1
                    g1.append(c[:, 0, -1].mean()); g2.append(c[:, 1, -1].mean()); gh.append(h[:, 0, -1].mean())
                fn1.append(g1); fn2.append(g2); fnh.append(gh)
            draws = np.stack(draws)                                                   # [M,N,J,K]
            hz_hat = hz_sum / ndraw                                                   # posterior-mean hazard
            q = np.quantile(draws, [0.025, 0.5, 0.975], axis=0); L, med, Uq = q[0], q[1], q[2]
            fhat = draws.mean(0); post_sd = draws.std(0)
            # EmpSD: repeated-dataset sampling SD of the point estimate (R independent student datasets)
            reps = []
            for r in range(R_REP):
                sr = simulate(STUDENT_N, a, b, 500000 + 1000*seed + r)
                reps.append(np.asarray(fit_student(sr, teacher, tg, eta, 7000+r, "adamw", ADAMW_EPOCHS, feats).predict_cif(test_df)))
            emp_sd = np.stack(reps).std(0)
            # multi-functional convergence (F_c1, F_c2, hazard_c1 at last horizon); worst R-hat / min ESS
            def rh_ess(fn):
                A = np.asarray(fn)
                try: return float(gelman_rubin_rhat(A)), float(effective_sample_size(A))
                except Exception: return float("nan"), float("nan")
            rr = [rh_ess(f) for f in (fn1, fn2, fnh)]
            rhat = max(r_ for r_, _ in rr); ess = min(e_ for _, e_ in rr)
            def ph(x): return x.mean(axis=(0, 1))
            inside = (tcif >= L) & (tcif <= Uq)
            # STEP 3: 1.2x widening (center on median) -> per-horizon coverage
            def widen_cov(scale):
                Ls = np.clip(med + scale * (L - med), 0, 1)
                Us = np.clip(med + scale * (Uq - med), 0, 1)
                return ((tcif >= Ls) & (tcif <= Us)).mean(axis=(0, 1))
            cov12 = widen_cov(1.2)
            # STEP 5: risk strata (per-subject true CIF at last horizon): low 20% / mid 60% / high 20%
            risk = tcif[:, :, -1].mean(1); order = np.argsort(risk); n = len(order)
            strata = {"low": order[:int(0.2*n)], "mid": order[int(0.2*n):int(0.8*n)], "high": order[int(0.8*n):]}
            def strat(idx):
                return dict(cov=float(inside[idx].mean()), front_cov=float(inside[idx][:, :, :3].mean()),
                            missb=float((tcif[idx] < L[idx]).mean()), missa=float((tcif[idx] > Uq[idx]).mean()),
                            bias=float((med[idx] - tcif[idx]).mean()))
            strat_res = {k: strat(v) for k, v in strata.items()}
            # STEP 6: teacher-induced shift (eta>0 vs stored eta=0): Delta=Fhat1-Fhat0 vs residual R=F0-Fhat0
            away = None
            if eta != 0 and seed in med0_store:
                med0 = med0_store[seed]; Delta = med - med0; Resid = tcif - med0
                away = ph((Delta * Resid < 0).astype(float))                            # per-horizon: shift AWAY from truth
            if eta == 0:
                med0_store[seed] = med.copy()
            rec = dict(
                cov=ph(inside.astype(float)), width=ph(Uq - L), bias=ph(med - tcif),
                miss_below=ph((tcif < L).astype(float)), miss_above=ph((tcif > Uq).astype(float)),
                post_sd=ph(post_sd), emp_sd=ph(emp_sd), F0=ph(tcif), Fhat=ph(fhat), Lm=ph(L), Um=ph(Uq),
                rhat=rhat, ess=ess, cif_mse=ph((med - tcif) ** 2), hz_bias=ph(hz_hat - thz), hz_mse=ph((hz_hat - thz) ** 2),
                cov12=cov12, strat=strat_res, away=away)
            agg.setdefault((eta), []).append(rec)
            awstr = f"away={away.mean():.2f} " if away is not None else ""
            print(f"  seed{seed} eta={eta}: cov={rec['cov'].mean():.3f} (1.2x->{cov12.mean():.3f}) "
                  f"bias={rec['bias'].mean():+.4f} postSD/empSD={rec['post_sd'].mean()/rec['emp_sd'].mean():.2f} "
                  f"missB={rec['miss_below'].mean():.3f} missA={rec['miss_above'].mean():.3f} "
                  f"strat_lowfront={strat_res['low']['front_cov']:.2f} {awstr}"
                  f"Rhat={rhat:.2f} ESS={ess:.0f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- plots ----
    x = np.arange(1, K+1)
    for eta in ETAS:
        recs = agg[eta]
        M = {k: np.stack([r[k] for r in recs]).mean(0) for k in
             ["cov","width","bias","miss_below","miss_above","post_sd","emp_sd","F0","Fhat","Lm","Um",
              "cif_mse","hz_bias","hz_mse","cov12"]}
        fig, ax = plt.subplots(2, 2, figsize=(13, 8)); ax = ax.ravel()
        # Plot 1: truth / posterior mean / band (cohort-averaged)
        ax[0].plot(x, M["F0"], "k-o", ms=3, label="true CIF $F_0$")
        ax[0].plot(x, M["Fhat"], "b-o", ms=3, label="posterior mean $\\hat F$")
        ax[0].fill_between(x, M["Lm"], M["Um"], color="b", alpha=0.15, label="95% interval")
        ax[0].set_title("Cohort-avg CIF: truth vs posterior"); ax[0].legend(fontsize=8)
        # Plot 2: coverage, and EmpSD vs PostSD (the key under-dispersion test)
        ax[1].plot(x, M["cov"], "g-o", ms=3, label="coverage (raw)"); ax[1].axhline(0.95, ls="--", c="gray", lw=.8)
        ax[1].plot(x, M["cov12"], "-^", color="darkgreen", ms=3, label="coverage (1.2x widened)")
        ax[1].plot(x, np.clip(M["post_sd"]/M["emp_sd"], 0, 1.6), "m-s", ms=3, label="PostSD / EmpSD")
        ax[1].axhline(1.0, ls=":", c="m", lw=.8)
        ax[1].set_title("Coverage (raw & 1.2x) & PostSD/EmpSD"); ax[1].legend(fontsize=7); ax[1].set_ylim(0, 1.6)
        # Plot 3: directional misses + bias
        ax[2].plot(x, M["miss_below"], "r-o", ms=3, label="P{F0<L} (over-est)")
        ax[2].plot(x, M["miss_above"], "-o", color="orange", ms=3, label="P{F0>U} (under-est)")
        ax[2].plot(x, M["bias"], "b--", label="bias E[$\\hat F$-F0]")
        ax[2].axhline(0, ls=":", c="gray", lw=.6); ax[2].set_title("Directional misses & CIF bias"); ax[2].legend(fontsize=8)
        # Plot 4: propagation -- LOCAL hazard MSE vs CUMULATIVE CIF MSE
        ax[3].plot(x, M["hz_mse"], "-o", color="teal", ms=3, label="hazard MSE (local, per-interval)")
        ax[3].plot(x, M["cif_mse"], "-o", color="purple", ms=3, label="CIF MSE (cumulative)")
        ax[3].plot(x, np.cumsum(M["hz_mse"]), ":", color="teal", label="cumsum(hazard MSE)")
        ax[3].set_title("Error propagation: hazard $\\to$ CIF"); ax[3].legend(fontsize=8)
        for a_ in ax: a_.set_xlabel("horizon"); a_.grid(ls=":", alpha=.4)
        fig.suptitle(f"Per-horizon diagnostic (K={K}, DGP={DGP}, eta={eta:g})", fontsize=12)
        fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig(OUT_DIR/f"horizon_diag_eta{eta:g}.png", dpi=140, bbox_inches="tight")

    # ---- report ----
    md = [f"# Per-horizon bias/variance/propagation diagnostic (K={K}, DGP={DGP})", ""]
    md.append(f"Log-linear exponential DGP, D={D}, SNR_B={SNR_B}, homogeneous teacher/student (hidden {HID}), "
              f"{DRIFT} SGLD ({N_CHAINS}x{SAMPLES_PER_CHAIN}), R={R_REP} repeated datasets for EmpSD. Seeds {SEEDS}. Nominal 0.95.")
    md.append("")
    md.append(f"**Chains:** {N_CHAINS}, all warm-started from the SAME AdamW MAP -> R-hat/ESS below are a "
              f"WITHIN-BASIN mixing diagnostic (worst of F_c1, F_c2, hazard_c1 at last horizon), NOT full-posterior "
              f"exploration. Cov and ratio shown as mean +/- SD across the {len(SEEDS)} independent seeds.")
    md.append("")
    md.append("| eta | cov (mean+/-SD) | cov 1.2x-widened | PostSD/EmpSD (mean+/-SD) | miss_below (over) | "
              "miss_above (under) | CIF bias | R-hat | ESS |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for eta in ETAS:
        recs = agg[eta]
        covs = np.array([r["cov"].mean() for r in recs]); ratios = np.array([r["post_sd"].mean()/r["emp_sd"].mean() for r in recs])
        c12 = np.array([r["cov12"].mean() for r in recs])
        g = lambda k: np.stack([r[k] for r in recs]).mean()
        md.append(f"| {eta:g} | {covs.mean():.3f} +/- {covs.std():.3f} | {c12.mean():.3f} | "
                  f"{ratios.mean():.2f} +/- {ratios.std():.2f} | {g('miss_below'):.3f} | {g('miss_above'):.3f} | "
                  f"{g('bias'):+.4f} | {np.nanmean([r['rhat'] for r in recs]):.2f} | {np.nanmean([r['ess'] for r in recs]):.0f} |")
    md.append("")
    # ---- Section-1.3-style headline: does borrowing beat the internal model on coverage? ----
    md.append("## Borrowing vs internal model (Section 1.3 style)")
    md.append("")
    md.append("Coverage by follow-up region (mean over seeds; nominal 0.95). front = first 3 horizons, "
              "back = last 3, overall = all K. width is the mean 95% interval length. eta=0 is the "
              "internal (student-only) model; eta>0 borrows from the teacher.")
    md.append("")
    md.append("| eta | model | front cov | back cov | overall cov | overall width |")
    md.append("|---:|---|---:|---:|---:|---:|")
    def _region(recs, k, sl):
        return float(np.mean([r[k][sl].mean() for r in recs]))
    for eta in ETAS:
        recs = agg[eta]
        label = "internal (no teacher)" if eta == 0 else f"borrowing (eta={eta:g})"
        md.append(f"| {eta:g} | {label} | {_region(recs,'cov',slice(0,3)):.3f} | "
                  f"{_region(recs,'cov',slice(-3,None)):.3f} | {_region(recs,'cov',slice(None)):.3f} | "
                  f"{_region(recs,'width',slice(None)):.3f} |")
    md.append("")
    if 0.0 in [float(e) for e in ETAS] and any(e != 0 for e in ETAS):
        base = agg[0.0]; b_ov = _region(base, 'cov', slice(None)); b_fr = _region(base, 'cov', slice(0, 3))
        best_eta = max([e for e in ETAS if e != 0], key=lambda e: _region(agg[e], 'cov', slice(None)))
        be = agg[best_eta]
        md.append(f"**Headline:** internal overall coverage {b_ov:.3f} (front {b_fr:.3f}); best borrowing "
                  f"eta={best_eta:g} overall {_region(be,'cov',slice(None)):.3f} (front {_region(be,'cov',slice(0,3)):.3f}). "
                  f"Borrowing {'improves' if _region(be,'cov',slice(None))>b_ov else 'does not improve'} calibration "
                  f"over the internal model, mirroring Section 1.3.")
    md.append("")
    md.append("**STEP 3 (1.2x widening) verdict:** raw->1.2x. If it reaches ~0.93-0.95, the eta=0 failure is spread "
              "calibration; if it only reaches ~0.82, bias/shape/heterogeneity remains.")
    md.append("")
    md.append("## STEP 5: risk-stratified coverage (low 20% / mid 60% / high 20% by true CIF at last horizon)")
    md.append("")
    md.append("| eta | stratum | cov | FRONT cov (t1-t3) | miss_below | miss_above | bias |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for eta in ETAS:
        recs = agg[eta]
        for s in ["low", "mid", "high"]:
            gv = lambda k: float(np.mean([r["strat"][s][k] for r in recs]))
            md.append(f"| {eta:g} | {s} | {gv('cov'):.3f} | {gv('front_cov'):.3f} | {gv('missb'):.3f} | "
                      f"{gv('missa'):.3f} | {gv('bias'):+.4f} |")
    md.append("")
    md.append("Expected under the boundary-bias hypothesis: **low-risk, front** subjects show high miss_below "
              "(F0<L, over-estimation) because their true early CIF ~ 0 but the interval floor sits above it.")
    md.append("")
    if any(e != 0 for e in ETAS):
        md.append("## STEP 6: teacher-induced shift (eta>0 vs eta=0)")
        md.append("")
        md.append("`away` = fraction of entries where the teacher shift (Fhat_eta - Fhat_0) moves in the direction "
                  "OPPOSITE to the correction needed (F0 - Fhat_0), i.e. the teacher pulls the prediction AWAY from truth. "
                  ">0.5 = teacher hurts on balance.")
        md.append("")
        md.append("| eta | away (all horizons) | away (front t1-t3) |")
        md.append("|---:|---:|---:|")
        for eta in ETAS:
            if eta == 0: continue
            recs = [r for r in agg[eta] if r["away"] is not None]
            if not recs: continue
            aw = np.stack([r["away"] for r in recs]).mean(0)
            md.append(f"| {eta:g} | {aw.mean():.3f} | {aw[:3].mean():.3f} |")
        md.append("")
    md.append("**Verdict key:** miss_below>>miss_above = over-estimation; PostSD<EmpSD (ratio<1) = under-dispersion; "
              "CIF MSE flat while cumsum(hazard MSE) rises = propagation shapes width not error.")
    md.append("")
    for eta in ETAS:
        md.append(f"![eta{eta:g}](horizon_diag_eta{eta:g}.png)")
    (OUT_DIR/"horizon_diagnostic_report.md").write_text("\n".join(md)+"\n")
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
