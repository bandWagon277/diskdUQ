"""Experiment 1+2: variance-vs-bias decomposition of CIF undercoverage (Phases 3-4).

Question. When an interval misses the true CIF, is it (a) too narrow for the estimator's own
sampling variability [a VARIANCE / representation-uncertainty problem the sandwich/subnet
correction could fix], or (b) the right width but centred on a biased target [a BIAS problem
only conformal / a better teacher can fix]?

Design. At the headline cell, fixed teacher, eta in {0,1}:
  - Fit the deployed student; build 4 UQ intervals for the CIF on a fixed test set:
    last-layer Laplace (H^-1), last-layer Godambe (H^-1 J H^-1), deep ensemble, patient bootstrap.
  - PSEUDO-TARGET (Phase 3): refit the student on R fresh student cohorts (same DGP, FIXED
    teacher, FIXED test set). This gives the estimator's actual sampling distribution:
      pseudo_mean = E[F(theta_hat)]  (repeated-training mean; NOT F(theta_eta) exactly),
      replicate_sd = sd over replicates  (the TRUE per-entry sampling variability).
    A large-n student fit gives an F(theta_eta) proxy for the population minimiser.
  - Key diagnostic: method_sd / replicate_sd. <1 => the method underestimates the estimator's
    own variability (a real variance gap). Coverage is reported against BOTH pseudo_mean and the
    true CIF; the gap between them is the teacher/approximation BIAS.
  - Split-conformal: scale each method's width on a calibration split to hit nominal coverage of
    the pseudo-target, then measure the residual true-CIF coverage (the irreducible bias).

Outputs (OUT_DIR, default responses_cov/): coverage_decomposition_report.md, .csv
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import jacfwd, hessian

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent, DiscreteSurvivalModel, competing_risk_c_index,
    fit_time_grid, simulate_competing_risk_cohorts, transform_durations,
)
from diskd._ground_truth import true_cif_at_grid
from diskd.networks import sinusoidal_time_embedding
from diskd.utils import competing_cif, competing_interval_probs
from diskd.losses import CompetingRiskNLLLoss
from diskd.utils import full_probs_from_event_probs, make_at_risk_mask, temperature_scale_probs

# ---------- config ----------
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
TEST_N = int(os.environ.get("TEST_N", 500))
QUALITY = os.environ.get("TEACHER_FEATURE_QUALITY", "full")
J, K = 2, 12
HID = int(os.environ.get("STUDENT_HIDDEN", 32))
TEACHER_HID = 128
BATCH = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))
TEMPERATURE = 2.0
ETAS = [float(e) for e in os.environ.get("ETAS", "0,1").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",")]
R_REP = int(os.environ.get("R_REP", 20))          # pseudo-target replicates
N_ENS = int(os.environ.get("N_ENS", 10))          # deep-ensemble / bootstrap members
LARGE_N = int(os.environ.get("LARGE_N", 8000))    # population (theta_eta) proxy
Z = 1.959963985
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_cov"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def build_student(teacher, eta, epochs=ADAMW_EPOCHS, hidden=HID, dropout=0.0):
    return DiSKDStudent(
        teacher_model=teacher, teacher_type="competing", eta=eta, temperature=TEMPERATURE,
        num_risks=J, num_durations=K, hidden_dim=hidden, epochs=epochs, batch_size=BATCH,
        device=DEVICE, optimizer="adamw", dropout=dropout)


def fit_on(student_df, teacher, tg, eta, seed, epochs=ADAMW_EPOCHS, hidden=HID, dropout=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build_student(teacher, eta, epochs, hidden, dropout)
    m.time_grid = tg
    m.fit(student_df, feature_cols=list(student_df.columns.drop(["duration", "event"])))
    return m


def cif_on(model, test_df):
    return np.asarray(model.predict_cif(test_df))               # [N,J,K]


# ---------- last-layer Laplace / Godambe (exact, 66 params) ----------

def backbone_feats(net, x):
    net.eval()
    with torch.no_grad():
        h = net.feature_projection(x).unsqueeze(1)
        emb = sinusoidal_time_embedding(net.num_durations, h.shape[-1], device=x.device, dtype=x.dtype)
        h = net.blocks(h + emb.unsqueeze(0))
    return h


def last_layer_intervals(model, teacher, tg, student_df, test_df, eta, true_cif):
    """Return dict name -> (median_cif[N,J,K], sd[N,J,K]) for Laplace and Godambe."""
    net = model.net
    Hd = HID
    xtr = torch.as_tensor(model.preprocessor.transform(student_df), dtype=torch.float32, device=DEVICE)
    idx = torch.as_tensor(transform_durations(student_df["duration"].values, tg), dtype=torch.long, device=DEVICE)
    ev = torch.as_tensor(student_df["event"].values.copy(), dtype=torch.long, device=DEVICE)
    tp = teacher.predict_interval_probs(student_df)[:, :J, :]
    tfull = full_probs_from_event_probs(torch.as_tensor(tp, dtype=torch.float64, device=DEVICE))
    feat = backbone_feats(net, xtr).double()
    theta = torch.cat([net.head.weight.detach().reshape(-1), net.head.bias.detach().reshape(-1)]).double()
    n = feat.shape[0]

    mask = make_at_risk_mask(idx, K).to(torch.float64)
    t_t = temperature_scale_probs(tfull, TEMPERATURE); log_t = t_t.clamp_min(1e-12).log()
    nll = CompetingRiskNLLLoss()

    def logits_of(th, f):
        W = th[:J * Hd].view(J, Hd); b = th[J * Hd:]
        return torch.einsum("nkd,jd->njk", f, W) + b.view(1, J, 1)

    def r_vec(th):
        return nll(logits_of(th, feat), idx, ev, reduction="none")

    def q_vec(th):
        st = temperature_scale_probs(competing_interval_probs(logits_of(th, feat)), TEMPERATURE)
        kl = t_t * (log_t - st.clamp_min(1e-12).log())
        return (TEMPERATURE ** 2) * (kl.sum(1) * mask).sum(1)

    def m_mean(th):
        return (r_vec(th) + eta * q_vec(th)).mean()

    # polish to stationarity
    th = theta.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([th], max_iter=150, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)
    opt.step(lambda: _clos(opt, m_mean, th))
    theta = th.detach()

    S_R = jacfwd(r_vec)(theta); S_Q = jacfwd(q_vec)(theta)
    cR = S_R - S_R.mean(0, keepdim=True); cQ = S_Q - S_Q.mean(0, keepdim=True)
    J_eta = (cR.T @ cR + eta * (cR.T @ cQ + cQ.T @ cR) + eta * eta * (cQ.T @ cQ)) / n
    H = hessian(m_mean)(theta); H = 0.5 * (H + H.T)
    Hinv = _psd_pinv(H)
    V_lap = Hinv / n
    V_sand = Hinv @ J_eta @ Hinv / n

    xte = torch.as_tensor(model.preprocessor.transform(test_df), dtype=torch.float32, device=DEVICE)
    fte = backbone_feats(net, xte).double()
    Nt = xte.shape[0]

    def cif_flat(th):
        return competing_cif(competing_interval_probs(logits_of(th, fte))).reshape(-1)

    cif_pt = cif_flat(theta).reshape(Nt, J, K)
    D = jacfwd(cif_flat)(theta)                                  # [Nt*J*K, p]
    out = {}
    for nm, V in (("last_layer_laplace", V_lap), ("last_layer_godambe", V_sand)):
        var = ((D @ V) * D).sum(-1).clamp_min(0).reshape(Nt, J, K)
        out[nm] = (cif_pt.detach().cpu().numpy(), var.sqrt().detach().cpu().numpy())
    return out


def _clos(opt, fn, th):
    opt.zero_grad(); loss = fn(th); loss.backward(); return loss


def _psd_pinv(M, rcond=1e-6):
    ev, evec = torch.linalg.eigh(M)
    keep = ev > rcond * ev.max().clamp_min(0)
    inv = torch.where(keep, 1.0 / ev.clamp_min(rcond * ev.max().clamp_min(1e-12)), torch.zeros_like(ev))
    return (evec * inv) @ evec.T


# ---------- ensemble / bootstrap intervals ----------

def ensemble_intervals(cif_stack):
    """cif_stack [M,N,J,K] -> (median, sd)."""
    return np.median(cif_stack, 0), cif_stack.std(0)


# ---------- coverage / decomposition ----------

def eval_method(median, sd, pseudo_mean, replicate_sd, true_cif, calib_frac=0.3, seed=0):
    """Per-entry coverage vs pseudo-target and truth, variance ratio, + split-conformal."""
    lo = np.clip(median - Z * sd, 0, 1); hi = np.clip(median + Z * sd, 0, 1)
    cov_pseudo = ((pseudo_mean >= lo) & (pseudo_mean <= hi)).mean()
    cov_true = ((true_cif >= lo) & (true_cif <= hi)).mean()
    Nt = median.shape[0]
    inside_true = (true_cif >= lo) & (true_cif <= hi)
    simult_true = inside_true.reshape(Nt, -1).all(1).mean()
    mae = np.abs(median - true_cif).mean()
    var_ratio = (sd.mean() / max(replicate_sd.mean(), 1e-9))
    # split-conformal: pick a scale on a calibration split so pseudo-target coverage hits 0.95,
    # then measure true-CIF coverage on the held-out split (deployable target = pseudo mean).
    rng = np.random.default_rng(seed)
    Ncal = int(calib_frac * Nt)
    perm = rng.permutation(Nt); cal, te = perm[:Ncal], perm[Ncal:]
    resid = np.abs(pseudo_mean[cal] - median[cal]) / np.clip(sd[cal], 1e-6, None)
    scale = float(np.quantile(resid, 0.95))
    lo_c = np.clip(median[te] - scale * sd[te], 0, 1); hi_c = np.clip(median[te] + scale * sd[te], 0, 1)
    conf_cov_pseudo = ((pseudo_mean[te] >= lo_c) & (pseudo_mean[te] <= hi_c)).mean()
    conf_cov_true = ((true_cif[te] >= lo_c) & (true_cif[te] <= hi_c)).mean()
    conf_width = (hi_c - lo_c).mean()
    return dict(cov_pseudo=cov_pseudo, cov_true=cov_true, simult_true=simult_true, mae=mae,
                width=(hi - lo).mean(), var_ratio=var_ratio, conf_scale=scale,
                conf_cov_pseudo=conf_cov_pseudo, conf_cov_true=conf_cov_true, conf_width=conf_width)


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    print(f"=== Experiment 1+2: coverage decomposition (device={DEVICE}) ===")
    print(f"seeds={SEEDS} etas={ETAS} R_rep={R_REP} n_ens={N_ENS}")
    rows = []
    for seed in SEEDS:
        cohorts = simulate_competing_risk_cohorts(
            n_teacher=TEACHER_N, n_student=STUDENT_N, n_test=TEST_N, seed=seed,
            teacher_feature_quality=QUALITY)
        dur = np.concatenate([cohorts.teacher["duration"].values, cohorts.student["duration"].values])
        tg = fit_time_grid(dur, K)
        teacher = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=TEACHER_HID,
                                        epochs=TEACHER_EPOCHS, batch_size=BATCH, device=DEVICE,
                                        time_grid=tg).fit(cohorts.teacher, feature_cols=cohorts.teacher_features)
        test = cohorts.test
        true_cif = true_cif_at_grid(test, tg, num_risks=J)
        tdur, tev = test["duration"].to_numpy(), test["event"].to_numpy()

        for eta in ETAS:
            t0 = time.time()
            main_model = fit_on(cohorts.student, teacher, tg, eta, 1000 * seed)

            # pseudo-target: R fresh student cohorts, FIXED teacher + FIXED test set
            rep = []
            for r in range(R_REP):
                sr = simulate_competing_risk_cohorts(
                    n_teacher=200, n_student=STUDENT_N, n_test=2, seed=100000 + 1000 * seed + r,
                    teacher_feature_quality=QUALITY).student
                mr = fit_on(sr, teacher, tg, eta, 7000 + r)
                rep.append(cif_on(mr, test))
            rep = np.stack(rep, 0)                                  # [R,N,J,K]
            pseudo_mean = rep.mean(0); replicate_sd = rep.std(0)
            # theta_eta proxy: one large-n student fit
            big = simulate_competing_risk_cohorts(n_teacher=200, n_student=LARGE_N, n_test=2,
                                                  seed=200000 + seed, teacher_feature_quality=QUALITY).student
            cif_pop = cif_on(fit_on(big, teacher, tg, eta, 999), test)
            bias_vs_pop = float(np.abs(pseudo_mean - cif_pop).mean())
            bias_vs_truth = float(np.abs(pseudo_mean - true_cif).mean())

            methods = {}
            methods.update(last_layer_intervals(main_model, teacher, tg, cohorts.student, test, eta, true_cif))
            # deep ensemble (different inits, same data)
            ens = np.stack([cif_on(fit_on(cohorts.student, teacher, tg, eta, 3000 + m), test)
                            for m in range(N_ENS)], 0)
            methods["deep_ensemble"] = ensemble_intervals(ens)
            # patient bootstrap
            bs = []
            rng = np.random.default_rng(seed)
            for m in range(N_ENS):
                bidx = rng.integers(0, STUDENT_N, STUDENT_N)
                bs.append(cif_on(fit_on(cohorts.student.iloc[bidx].reset_index(drop=True),
                                        teacher, tg, eta, 4000 + m), test))
            methods["bootstrap"] = ensemble_intervals(np.stack(bs, 0))

            ctd_main = competing_risk_c_index(cif_on(main_model, test), tdur, tev)
            for nm, (med, sd) in methods.items():
                r = eval_method(med, sd, pseudo_mean, replicate_sd, true_cif, seed=seed)
                r.update(dict(seed=seed, eta=eta, method=nm, ctd1=float(ctd_main[0]),
                              bias_vs_pop=bias_vs_pop, bias_vs_truth=bias_vs_truth,
                              replicate_sd_mean=float(replicate_sd.mean())))
                rows.append(r)
                print(f"  seed{seed} eta={eta} {nm:20s} cov_pseudo={r['cov_pseudo']:.3f} "
                      f"cov_true={r['cov_true']:.3f} var_ratio={r['var_ratio']:.2f} "
                      f"conf_true={r['conf_cov_true']:.3f}", flush=True)
            print(f"  seed{seed} eta={eta}: bias(pseudo vs truth)={bias_vs_truth:.4f} "
                  f"replicate_sd={float(replicate_sd.mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "coverage_decomposition.csv", index=False)
    agg = df.groupby(["eta", "method"]).mean(numeric_only=True).reset_index()

    md = ["# Experiment 1+2 — Coverage Decomposition (variance vs bias)", ""]
    md.append(f"Cell: teacher {TEACHER_N} / student {STUDENT_N} / test {TEST_N}, quality={QUALITY}. "
              f"R={R_REP} pseudo-target replicates (fixed teacher, fixed test), {N_ENS} ensemble/bootstrap "
              f"members. Seeds {SEEDS}. Nominal 0.95.")
    md.append("")
    md.append("`var_ratio` = method_sd / replicate_sd (the estimator's TRUE sampling sd). <1 => the "
              "method underestimates sampling variability (a variance gap). `cov_pseudo` = coverage of "
              "the repeated-training mean E[F(theta_hat)]; `cov_true` = coverage of the closed-form CIF. "
              "A large cov_pseudo - cov_true gap = teacher/approximation BIAS. `conf_cov_true` = coverage "
              "after split-conformal scaling to the pseudo-target.")
    md.append("")
    md.append("| eta | method | cov_pseudo | cov_true | simult_true | var_ratio | width | MAE | conf_cov_pseudo | conf_cov_true | conf_width |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        md.append(f"| {r['eta']:g} | {r['method']} | {r['cov_pseudo']:.3f} | {r['cov_true']:.3f} | "
                  f"{r['simult_true']:.3f} | {r['var_ratio']:.2f} | {r['width']:.3f} | {r['mae']:.4f} | "
                  f"{r['conf_cov_pseudo']:.3f} | {r['conf_cov_true']:.3f} | {r['conf_width']:.3f} |")
    md.append("")
    md.append("## Bias (pseudo-target vs truth), per eta")
    md.append("")
    md.append("| eta | bias(pseudo vs pop) | bias(pseudo vs truth) | replicate_sd |")
    md.append("|---:|---:|---:|---:|")
    for eta in ETAS:
        sub = df[df.eta == eta]
        md.append(f"| {eta:g} | {sub['bias_vs_pop'].mean():.4f} | {sub['bias_vs_truth'].mean():.4f} | "
                  f"{sub['replicate_sd_mean'].mean():.4f} |")
    md.append("")
    md.append("Raw rows: `coverage_decomposition.csv`.")
    (OUT_DIR / "coverage_decomposition_report.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
