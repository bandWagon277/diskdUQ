"""Teacher audit: is the teacher itself biased, and is eta=1 too strong? (clean DGP)

Before concluding "negative transfer", audit the teacher and the borrowing strength.
Implements the requested checks:
  1. Teacher accuracy vs the ORACLE truth, per horizon AND per risk stratum
     (teacher CIF bias/MSE, teacher hazard bias/MSE). Is the teacher risk-compressed
     (low-risk over-predict, high-risk under-predict)?
  4. Numeric magnitude of the two loss pieces at eta=1: |NLL|, eta*|KL| (with the T^2
     factor), and their gradient norms -> the TRUE borrowing strength (not just set eta).
  6. eta-path {0, 0.1, 0.25, 0.5, 1}: per-eta CIF MSE, bias, teacher-away, coverage
     (5-member deep ensemble) -> is it a standard bias-variance tradeoff (best at
     intermediate eta) rather than "experiment is bad"?
  7 (light). Teacher-seed sensitivity: a few teacher realizations.
Plus the FOUR-CURVE diagnostic: F0, F_teacher, Fhat(eta=0), Fhat(eta=1) per low/mid/high
risk stratum -- reveals at a glance whether the teacher is biased, the student is biased,
or KL pulls a good student toward a biased teacher.

Outputs -> responses_teacher/.
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
from diskd import (DiSKDStudent, DiscreteSurvivalModel, competing_risk_c_index,
                   fit_time_grid, transform_durations)
from diskd.losses import CompetingRiskNLLLoss
from diskd.utils import competing_interval_probs, full_probs_from_event_probs, make_at_risk_mask, temperature_scale_probs

D = int(os.environ.get("D", 6)); J = 2
K = int(os.environ.get("NUM_DURATIONS", 20))
TEACHER_N = int(os.environ.get("TEACHER_N", 5000)); STUDENT_N = int(os.environ.get("STUDENT_N", 1000))
TEST_N = int(os.environ.get("TEST_N", 1000)); HID = int(os.environ.get("HIDDEN", 32)); BATCH = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100)); ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 60))
SNR_B = float(os.environ.get("SNR_B", 0.5)); BASE_RATE = float(os.environ.get("BASE_RATE", 1.0))
CENSOR_RATE = float(os.environ.get("CENSOR_RATE", 0.3)); TEMP = float(os.environ.get("TEMPERATURE", 2.0))
ETAS = [float(e) for e in os.environ.get("ETAS", "0,0.1,0.25,0.5,1").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
N_ENS = int(os.environ.get("N_ENS", 5)); TEACHER_SEEDS = int(os.environ.get("TEACHER_SEEDS", 3))
Z = 1.959963985
_DEFAULT = Path(__file__).resolve().parent.parent / "responses_teacher"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT)))


def dgp_params():
    rng = np.random.default_rng(1234); return np.log(BASE_RATE) + np.zeros(J), SNR_B * rng.normal(size=(J, D))

def rates(x, a, b): return np.exp(a[None, :] + x @ b.T)

def simulate(n, a, b, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(n, D)); r = rates(x, a, b)
    t = rng.exponential(scale=1.0/r); cause = t.argmin(1); et = t.min(1)
    cens = rng.exponential(scale=1.0/max(CENSOR_RATE, 1e-6), size=n); censored = cens < et
    dur = np.where(censored, cens, et); event = np.where(censored, 0, cause+1).astype(np.int64)
    df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(D)]); df["duration"] = dur.astype(float); df["event"] = event
    return df

def true_cif(df, cuts, a, b):
    x = df[[f"x{i+1}" for i in range(D)]].to_numpy(); r = rates(x, a, b); rt = r.sum(1, keepdims=True)
    return (r/rt)[:, :, None] * (1.0-np.exp(-rt*np.asarray(cuts)[None, :]))[:, None, :]

def true_hazard(df, cuts, a, b):
    x = df[[f"x{i+1}" for i in range(D)]].to_numpy(); r = rates(x, a, b); rt = r.sum(1, keepdims=True)
    cuts = np.asarray(cuts); prev = np.concatenate([[0.0], cuts[:-1]]); delta = (cuts-prev)[None, :]
    return (r/rt)[:, :, None] * (1.0-np.exp(-rt*delta))[:, None, :]

def fit_adamw(df, teacher, tg, eta, seed, feats):
    torch.manual_seed(seed); np.random.seed(seed)
    common = dict(num_risks=J, num_durations=K, hidden_dim=HID, epochs=ADAMW_EPOCHS, batch_size=BATCH,
                  device=DEVICE, optimizer="adamw")
    m = DiscreteSurvivalModel(**common) if eta == 0 else DiSKDStudent(
        teacher_model=teacher, teacher_type="competing", eta=eta, temperature=TEMP, **common)
    m.time_grid = tg; m.fit(df, feature_cols=feats); return m

def strata_idx(tcif):
    risk = tcif[:, :, -1].mean(1); order = np.argsort(risk); n = len(order)
    return {"low": order[:int(0.2*n)], "mid": order[int(0.2*n):int(0.8*n)], "high": order[int(0.8*n):]}


def kl_nll_magnitudes(student, student_df, teacher, tg, feats):
    """|NLL|, eta*|KL| (T^2-scaled) and their gradient norms at the fitted eta=1 student."""
    x = torch.as_tensor(student.preprocessor.transform(student_df), dtype=torch.float32, device=DEVICE)
    idx = torch.as_tensor(transform_durations(student_df["duration"].values, tg), dtype=torch.long, device=DEVICE)
    ev = torch.as_tensor(student_df["event"].values.copy(), dtype=torch.long, device=DEVICE)
    tp = teacher.predict_interval_probs(student_df)[:, :J, :]
    tfull = full_probs_from_event_probs(torch.as_tensor(tp, dtype=torch.float32, device=DEVICE))
    net = student.net; net.zero_grad()
    logits = student._reshape_logits(net(x)) if hasattr(student, "_reshape_logits") else net(x)
    nll = CompetingRiskNLLLoss()(logits, idx, ev, reduction="mean")
    mask = make_at_risk_mask(idx, K).to(logits.dtype)
    sfull = competing_interval_probs(logits)
    tt = temperature_scale_probs(tfull, TEMP); st = temperature_scale_probs(sfull, TEMP)
    kl = (tt * (tt.clamp_min(1e-7).log() - st.clamp_min(1e-7).log())).sum(1)
    kl_i = (TEMP * TEMP) * (kl * mask).sum(1)
    kl_mean = kl_i.mean()
    gn = lambda loss: float(torch.sqrt(sum((g**2).sum() for g in torch.autograd.grad(loss, net.parameters(), retain_graph=True))).item())
    return dict(nll=nll.item(), kl=kl_mean.item(), grad_nll=gn(nll), grad_kl=gn(kl_mean))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
    feats = [f"x{i+1}" for i in range(D)]; a, b = dgp_params()
    print(f"=== Teacher audit (K={K}, clean DGP, T={TEMP}) === etas={ETAS} seeds={SEEDS}", flush=True)

    teach_rows = []; path_rows = []; fourcurve = {}
    for seed in SEEDS:
        student_df = simulate(STUDENT_N, a, b, 10*seed+2); test_df = simulate(TEST_N, a, b, 10*seed+3)
        # teacher-seed sensitivity: a few teacher realizations on this data seed
        teachers = []
        for ts in range(TEACHER_SEEDS):
            tdf = simulate(TEACHER_N, a, b, 77000 + 10*seed + ts)
            tg = fit_time_grid(np.concatenate([tdf["duration"].values, student_df["duration"].values]), K)
            tt_model = DiscreteSurvivalModel(num_risks=J, num_durations=K, hidden_dim=HID, epochs=TEACHER_EPOCHS,
                                             batch_size=BATCH, device=DEVICE, time_grid=tg).fit(tdf, feature_cols=feats)
            teachers.append((tt_model, tg))
        teacher, tg = teachers[0]
        cuts = np.asarray(tg.cuts); tcif = true_cif(test_df, cuts, a, b); thz = true_hazard(test_df, cuts, a, b)
        strata = strata_idx(tcif)

        # --- 1. teacher accuracy vs oracle (per horizon + per stratum) ---
        for ti, (tt_, tgi) in enumerate(teachers):
            tc = np.asarray(tt_.predict_cif(test_df)); th = np.asarray(tt_.predict_hazard(test_df))
            row = dict(seed=seed, teacher=ti, cif_bias=float((tc-tcif).mean()), cif_mse=float(((tc-tcif)**2).mean()),
                       hz_bias=float((th-thz).mean()), hz_mse=float(((th-thz)**2).mean()),
                       ctd1=float(competing_risk_c_index(tc, test_df["duration"].values, test_df["event"].values)[0]))
            for s, idx in strata.items():
                row[f"cif_bias_{s}"] = float((tc[idx]-tcif[idx]).mean())
                row[f"cif_bias_{s}_front"] = float((tc[idx]-tcif[idx])[:, :, :3].mean())
            teach_rows.append(row)
            if ti == 0: teacher_cif0 = tc

        # --- 6. eta-path with deep-ensemble coverage + teacher-away, and 4-curve centers ---
        point0 = None
        for eta in ETAS:
            ens = np.stack([np.asarray(fit_adamw(student_df, teacher, tg, eta, 3000+m, feats).predict_cif(test_df))
                            for m in range(N_ENS)], 0)                        # [E,N,J,K]
            med = np.median(ens, 0); lo = np.quantile(ens, 0.025, 0); hi = np.quantile(ens, 0.975, 0)
            if eta == 0: point0 = med
            away = float(((med - point0) * (tcif - point0) < 0).mean()) if eta != 0 else 0.0
            cov = ((tcif >= lo) & (tcif <= hi))
            row = dict(seed=seed, eta=eta, cif_mse=float(((med-tcif)**2).mean()), bias=float((med-tcif).mean()),
                       away=away, cov=float(cov.mean()), front_cov=float(cov[:, :, :3].mean()))
            for s, idx in strata.items():
                row[f"bias_{s}"] = float((med[idx]-tcif[idx]).mean())
            path_rows.append(row)
            if eta in (0.0, 1.0):
                fourcurve.setdefault(seed, {})[eta] = med

        # --- 4. KL/NLL magnitude at eta=1 ---
        s1 = fit_adamw(student_df, teacher, tg, 1.0, 12345, feats)
        mags = kl_nll_magnitudes(s1, student_df, teacher, tg, feats)
        print(f"  seed{seed}: teacher cif_bias(low/mid/high)="
              f"{teach_rows[-TEACHER_SEEDS]['cif_bias_low']:+.3f}/{teach_rows[-TEACHER_SEEDS]['cif_bias_mid']:+.3f}/"
              f"{teach_rows[-TEACHER_SEEDS]['cif_bias_high']:+.3f} | "
              f"eta=1 |NLL|={mags['nll']:.3f} eta*|KL|={mags['kl']:.3f} "
              f"gradNLL={mags['grad_nll']:.2f} gradKL={mags['grad_kl']:.2f} "
              f"(ratio KL/NLL grad={mags['grad_kl']/max(mags['grad_nll'],1e-9):.2f})", flush=True)
        path_rows[-1]["_mags"] = mags  # stash on last row (eta=1)
        # store four-curve teacher + truth for this seed
        fourcurve.setdefault(seed, {})["teacher"] = teacher_cif0; fourcurve[seed]["truth"] = tcif; fourcurve[seed]["strata"] = strata

    # ---- aggregate + report ----
    td = pd.DataFrame(teach_rows); pd.DataFrame([{k: v for k, v in r.items() if k != "_mags"} for r in path_rows]).to_csv(OUT_DIR/"teacher_audit_path.csv", index=False)
    td.to_csv(OUT_DIR/"teacher_accuracy.csv", index=False)
    x = np.arange(1, K+1)

    # four-curve stratified plot (first seed)
    s0 = SEEDS[0]; fc = fourcurve[s0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for pi, s in enumerate(["low", "mid", "high"]):
        idx = fc["strata"][s]
        ax[pi].plot(x, fc["truth"][idx].mean((0,1)), "k-o", ms=3, label="$F_0$ truth")
        ax[pi].plot(x, fc["teacher"][idx].mean((0,1)), "-s", color="crimson", ms=3, label="$\\tilde F$ teacher")
        ax[pi].plot(x, fc[0.0][idx].mean((0,1)), "-^", color="gray", ms=3, label="$\\hat F$ eta=0")
        ax[pi].plot(x, fc[1.0][idx].mean((0,1)), "-v", color="blue", ms=3, label="$\\hat F$ eta=1")
        ax[pi].set_title(f"{s}-risk stratum"); ax[pi].set_xlabel("horizon"); ax[pi].grid(ls=":", alpha=.4); ax[pi].legend(fontsize=8)
    fig.suptitle(f"Four-curve center diagnostic (seed {s0}): truth / teacher / student(eta=0) / student(eta=1)", fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(OUT_DIR/"four_curve.png", dpi=140, bbox_inches="tight")

    # eta-path plot
    pdf = pd.DataFrame([{k: v for k, v in r.items() if k != "_mags"} for r in path_rows])
    ag = pdf.groupby("eta").mean(numeric_only=True).reset_index()
    fig2, ax2 = plt.subplots(1, 4, figsize=(18, 4))
    for j2, (col, ttl) in enumerate([("cif_mse", "CIF MSE"), ("bias", "CIF bias"), ("away", "teacher-away frac"), ("cov", "coverage")]):
        ax2[j2].plot(ag["eta"], ag[col], "o-"); ax2[j2].set_title(ttl); ax2[j2].set_xlabel("eta"); ax2[j2].grid(ls=":", alpha=.4)
    ax2[3].axhline(0.95, ls="--", c="gray", lw=.8); ax2[2].axhline(0.5, ls=":", c="r", lw=.8)
    fig2.suptitle("eta-path (deep-ensemble): bias-variance tradeoff", fontsize=12)
    fig2.tight_layout(rect=(0,0,1,0.93)); fig2.savefig(OUT_DIR/"eta_path.png", dpi=140, bbox_inches="tight")

    md = [f"# Teacher audit (K={K}, clean homogeneous DGP, T={TEMP})", ""]
    md.append(f"D={D}, SNR_B={SNR_B}, teacher {TEACHER_N} / student {STUDENT_N}, hidden {HID} (homogeneous). "
              f"Seeds {SEEDS}, {TEACHER_SEEDS} teacher realizations each, {N_ENS}-member ensemble for the eta-path. "
              f"Truth = closed-form CIF & hazard.")
    md.append("")
    md.append("## 1. Is the teacher itself biased? (teacher vs oracle, mean over seeds/teachers)")
    md.append("")
    md.append("| cif_bias | cif_mse | hz_bias | hz_mse | Ctd1 | bias low | bias mid | bias high | bias low FRONT |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    g = lambda k: td[k].mean()
    md.append(f"| {g('cif_bias'):+.4f} | {g('cif_mse'):.2e} | {g('hz_bias'):+.4f} | {g('hz_mse'):.2e} | {g('ctd1'):.3f} | "
              f"{g('cif_bias_low'):+.4f} | {g('cif_bias_mid'):+.4f} | {g('cif_bias_high'):+.4f} | {g('cif_bias_low_front'):+.4f} |")
    md.append("")
    md.append("If **bias low > 0 (over-predict) and bias high < 0 (under-predict)** the teacher is RISK-COMPRESSED and "
              "passes this to the student. Teacher-realization spread (SD of cif_bias_low across teachers): "
              f"{td.groupby('seed')['cif_bias_low'].std().mean():.4f}.")
    md.append("")
    md.append("## 4. Effective borrowing strength at eta=1 (magnitudes, mean over seeds)")
    md.append("")
    mg = [r["_mags"] for r in path_rows if "_mags" in r]
    if mg:
        mm = {k: float(np.mean([m[k] for m in mg])) for k in mg[0]}
        md.append(f"|NLL| = {mm['nll']:.3f}, eta*|KL| (T^2-scaled) = {mm['kl']:.3f}, ||grad NLL|| = {mm['grad_nll']:.3f}, "
                  f"eta*||grad KL|| = {mm['grad_kl']:.3f}. **KL/NLL gradient ratio = {mm['grad_kl']/max(mm['grad_nll'],1e-9):.2f}** "
                  f"(the T^2={TEMP**2:.0f} factor inflates the teacher pull; ratio>>1 means eta=1 is effectively strong borrowing).")
    md.append("")
    md.append("## 6. eta-path (bias-variance tradeoff)")
    md.append("")
    md.append("| eta | CIF MSE | bias | teacher-away | coverage | front cov | bias low | bias high |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ag.iterrows():
        md.append(f"| {r['eta']:g} | {r['cif_mse']:.2e} | {r['bias']:+.4f} | {r['away']:.3f} | {r['cov']:.3f} | "
                  f"{r['front_cov']:.3f} | {r['bias_low']:+.4f} | {r['bias_high']:+.4f} |")
    md.append("")
    md.append("If CIF MSE improves then worsens across eta (best at intermediate eta), it is a standard bias-variance "
              "tradeoff, not a broken experiment. teacher-away < 0.5 = teacher helps the point estimate on balance.")
    md.append("")
    md.append("![four-curve](four_curve.png)")
    md.append("![eta-path](eta_path.png)")
    (OUT_DIR/"teacher_audit_report.md").write_text("\n".join(md)+"\n")
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
