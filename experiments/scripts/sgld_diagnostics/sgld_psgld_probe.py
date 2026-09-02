"""pSGLD probe: preconditioned SGLD vs literal SGLD on both simulations.

Runs CR->CR with:
  - literal SGLD (baseline)
  - pSGLD (per-parameter adaptive preconditioning, Li et al. 2016)

On both:
  - Old simulation (N=5000 shared cohort, 8 student features)
  - New simulation (teacher N=10k, student N=500, 12 features)

Reports best-chain metrics: Ctd1, Ctd2, deviance, CIF coverage, CIF width.

    PYTHONPATH=src python examples/sgld_psgld_probe.py
"""
from __future__ import annotations

import copy, os, time
from pathlib import Path

import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

from diskd import (
    DiSKDStudent, DiscreteSurvivalModel, MultiChainSampler,
    competing_risk_c_index, fit_time_grid, gelman_rubin_rhat,
    predictive_deviance, simulate_competing_risks, transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import _iter_samples

SEED = 42
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 2000))
SGLD_STEP_SIZE = float(os.environ.get("SGLD_STEP_SIZE", 1e-3))
SGLD_FINAL = float(os.environ.get("SGLD_FINAL", 1e-5))
SGLD_GAMMA = float(os.environ.get("SGLD_GAMMA", 1.0))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

MODES_STR = os.environ.get("PROBE_MODES", "literal,psgld")
MODES = [m.strip() for m in MODES_STR.split(",")]

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))


def evaluate_chains(sampler, test, true_cif, test_dur, test_ev, test_idx, num_risks):
    lead = sampler.lead_chain
    chain_results = []
    for c_idx, chain in enumerate(sampler.chains):
        cifs, ctds, devs = [], [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            cifs.append(cif)
            ctds.append(competing_risk_c_index(cif, test_dur, test_ev))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_ev))
        if not cifs:
            chain_results.append(None)
            continue
        arr = np.stack(cifs, axis=0)
        q = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
        cov = coverage_from_quantiles(q, true_cif)
        ctd_arr = np.array(ctds)
        dev_arr = np.array(devs)
        chain_results.append({
            "n": len(cifs),
            "cif_cov": cov["overall"], "cif_wid": cov["mean_interval_width"],
            "ctd1": np.quantile(ctd_arr[:, 0], [0.025, 0.5, 0.975]),
            "ctd2": np.quantile(ctd_arr[:, 1], [0.025, 0.5, 0.975]),
            "dev": np.quantile(dev_arr, [0.025, 0.5, 0.975]),
            "loss": chain.history.losses[-1],
        })
    return chain_results


def best_chain(results):
    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    if not valid:
        return None, None
    idx, r = min(valid, key=lambda x: x[1]["dev"][1])
    return idx, r


def fmt(q):
    return f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"


def run_simulation(label, teacher_data, train_s, test_s, all_features, student_features,
                   teacher_hidden, student_hidden, teacher_epochs):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    num_risks, num_dur = 2, 12
    combined_dur = np.concatenate([teacher_data["duration"].values, train_s["duration"].values])
    time_grid = fit_time_grid(combined_dur, num_dur)
    test_dur = test_s["duration"].to_numpy()
    test_ev = test_s["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    true_cif = true_cif_at_grid(test_s, time_grid, num_risks=num_risks)

    print(f"  train={len(train_s)}, test={len(test_s)}, features={len(student_features)}")

    # Teacher
    t0 = time.time()
    teacher = DiscreteSurvivalModel(
        num_risks=num_risks, num_durations=num_dur, hidden_dim=teacher_hidden,
        epochs=teacher_epochs, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(teacher_data, feature_cols=all_features)
    print(f"  Teacher: loss={teacher.history.losses[-1]:.4f} ({time.time()-t0:.0f}s)")

    # AdamW baseline
    adamw = DiSKDStudent(
        num_risks=num_risks, num_durations=num_dur, hidden_dim=student_hidden,
        epochs=50, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=time_grid, optimizer="adamw",
    ).fit(train_s, feature_cols=student_features)
    cif_a = adamw.predict_cif(test_s)
    ctd_a = competing_risk_c_index(cif_a, test_dur, test_ev)
    dev_a = predictive_deviance(adamw.predict_interval_probs(test_s).numpy(), test_idx, test_ev)
    print(f"  AdamW:   Ctd1={ctd_a[0]:.4f}  Ctd2={ctd_a[1]:.4f}  dev={dev_a:.3f}  loss={adamw.history.losses[-1]:.4f}")

    # SGLD modes
    print(f"\n  {'mode':>8s} | {'best':>4s} | {'Ctd1':>28s} | {'Ctd2':>28s} | {'Deviance':>28s} | {'CIF cov':>7s} | {'CIF wid':>7s} | {'loss':>7s} | {'time':>5s}")
    print(f"  {'-'*140}")

    mode_results = {}
    for mode in MODES:
        t0 = time.time()
        base = DiSKDStudent(
            num_risks=num_risks, num_durations=num_dur, hidden_dim=student_hidden,
            epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=time_grid, optimizer="sgld",
            sgld_step_size=SGLD_STEP_SIZE, sgld_final_step_size=SGLD_FINAL,
            sgld_gamma=SGLD_GAMMA, sgld_drift_mode=mode,
            sgld_burnin_epochs=SGLD_EPOCHS // 2,
            sgld_samples_per_chain=SAMPLES_PER_CHAIN,
        )
        sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
        sampler.fit(train_s, feature_cols=student_features)
        elapsed = time.time() - t0

        chains = evaluate_chains(sampler, test_s, true_cif, test_dur, test_ev, test_idx, num_risks)
        bc_idx, bc = best_chain(chains)
        mode_results[mode] = {"chains": chains, "best_idx": bc_idx, "best": bc}

        if bc is not None:
            print(f"  {mode:>8s} | {bc_idx:>4d} | {fmt(bc['ctd1']):>28s} | {fmt(bc['ctd2']):>28s} | "
                  f"{fmt(bc['dev']):>28s} | {bc['cif_cov']:>7.3f} | {bc['cif_wid']:>7.3f} | "
                  f"{bc['loss']:>7.2f} | {elapsed:>4.0f}s", flush=True)

            # Also print all chains summary
            for ci, cr in enumerate(chains):
                if cr is None: continue
                tag = " <--best" if ci == bc_idx else ""
                print(f"    chain {ci}: Ctd1={cr['ctd1'][1]:.3f} Ctd2={cr['ctd2'][1]:.3f} "
                      f"dev={cr['dev'][1]:.2f} cov={cr['cif_cov']:.3f} wid={cr['cif_wid']:.3f} "
                      f"loss={cr['loss']:.2f}{tag}", flush=True)

    return {"adamw": {"ctd": ctd_a, "dev": dev_a, "loss": adamw.history.losses[-1]},
            "sgld": mode_results}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    print(f"Modes: {MODES}, chains={N_CHAINS}, draws/chain={SAMPLES_PER_CHAIN}, "
          f"epochs={SGLD_EPOCHS}, eps={SGLD_STEP_SIZE:.0e}->{SGLD_FINAL:.0e}, gamma={SGLD_GAMMA}")

    # === Old simulation ===
    data_old = simulate_competing_risks(n=5000, seed=7, censor_max=0.05)
    train_old, test_old = train_test_split(data_old, test_size=0.25, random_state=7)
    feats_old = [c for c in data_old.columns if c.startswith("x")]

    res_old = run_simulation(
        "OLD SIMULATION (N=5000 shared, student 8 features)",
        train_old, train_old, test_old,
        all_features=feats_old, student_features=feats_old[:8],
        teacher_hidden=48, student_hidden=32, teacher_epochs=30)

    # === New simulation ===
    teacher_data = simulate_competing_risks(n=10000, seed=SEED, censor_max=0.05)
    student_data = simulate_competing_risks(n=500, seed=SEED + 1, censor_max=0.05)
    train_new, test_new = train_test_split(student_data, test_size=0.25, random_state=SEED)
    feats_new = [c for c in teacher_data.columns if c.startswith("x")]

    res_new = run_simulation(
        "NEW SIMULATION (teacher N=10k, student N=500, 12 features)",
        teacher_data, train_new, test_new,
        all_features=feats_new, student_features=feats_new,
        teacher_hidden=128, student_hidden=32, teacher_epochs=100)

    # Save summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sgld_psgld_probe.txt"
    with out.open("w") as f:
        for sim_label, res in [("old", res_old), ("new", res_new)]:
            a = res["adamw"]
            f.write(f"# {sim_label}: AdamW Ctd1={a['ctd'][0]:.4f} Ctd2={a['ctd'][1]:.4f} dev={a['dev']:.3f}\n")
            for mode, mr in res["sgld"].items():
                bc = mr["best"]
                if bc:
                    f.write(f"{sim_label}\t{mode}\tbest={mr['best_idx']}\t"
                            f"ctd1={bc['ctd1'][1]:.4f}\tctd2={bc['ctd2'][1]:.4f}\t"
                            f"dev={bc['dev'][1]:.3f}\tcov={bc['cif_cov']:.4f}\twid={bc['cif_wid']:.4f}\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
