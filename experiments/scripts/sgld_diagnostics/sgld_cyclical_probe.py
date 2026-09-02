"""Cyclical SGLD probe with warm-start, Gaussian prior, and new simulation.

New simulation design (teacher-student data asymmetry):
  - Teacher cohort: N=10,000 (large, mimics national registry)
  - Student cohort: N=500   (small, mimics local center)
  - Teacher (perfect): trained on all 12 features on the large cohort
  - Teacher (degraded): trained on x1-x8 only on the large cohort
  - Student: 12 features, small network, small cohort
  - Internal-only: same as student but no teacher

Expected: perfect teacher > degraded teacher > internal-only on Ctd.

SGLD configurations compared (on CR->CR with perfect teacher):
  A. Cyclical SGLD + warm-start + prior (1 draw/cycle)
  B. Cyclical SGLD + warm-start + prior (5 draws/cycle)
  C. Monotone SGLD baseline (polynomial γ=1.0, no warm-start, flat prior)

Reports per-chain AND pooled CIF coverage, width, Ctd, deviance, R-hat.

Run:
    PYTHONPATH=src python examples/sgld_cyclical_probe.py
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    MultiChainSampler,
    competing_risk_c_index,
    effective_sample_size,
    fit_time_grid,
    gelman_rubin_rhat,
    predictive_deviance,
    simulate_competing_risks,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import _iter_samples

SEED = 42
N_TEACHER = int(os.environ.get("N_TEACHER", 10000))
N_STUDENT = int(os.environ.get("N_STUDENT", 500))
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = int(os.environ.get("TEACHER_HIDDEN", 128))
STUDENT_HIDDEN = int(os.environ.get("STUDENT_HIDDEN", 32))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
WARMSTART_EPOCHS = int(os.environ.get("WARMSTART_EPOCHS", 50))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 2000))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SGLD_STEP_SIZE = float(os.environ.get("SGLD_STEP_SIZE", 1e-3))
SGLD_GAMMA = float(os.environ.get("SGLD_GAMMA", 1.0))
CYCLE_LENGTH = int(os.environ.get("CYCLE_LENGTH", 50))
BURN_IN_CYCLES = int(os.environ.get("BURN_IN_CYCLES", 3))
PRIOR_SIGMA = float(os.environ.get("PRIOR_SIGMA", 5.0))

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))
OUT_TABLE = OUT_DIR / "sgld_cyclical_probe.txt"
OUT_FIGURE = OUT_DIR / "sgld_cyclical_probe.pdf"
OUT_FIGURE_PNG = OUT_FIGURE.with_suffix(".png")


def evaluate_sampler(sampler, test, true_cif, test_durations, test_events, test_idx):
    lead = sampler.lead_chain
    n_chains = len(sampler.chains)

    chain_results = []
    all_cifs = []
    for c_idx, chain in enumerate(sampler.chains):
        chain_cifs, ctds, devs = [], [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            chain_cifs.append(cif)
            all_cifs.append(cif)
            ctds.append(competing_risk_c_index(cif, test_durations, test_events))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_events))
        if len(chain_cifs) == 0:
            chain_results.append(None)
            continue
        arr = np.stack(chain_cifs, axis=0)
        q = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
        cov = coverage_from_quantiles(q, true_cif)
        ctd_arr = np.array(ctds)
        dev_arr = np.array(devs)
        chain_results.append({
            "n_draws": len(chain_cifs),
            "cif_coverage": cov["overall"],
            "cif_width": cov["mean_interval_width"],
            "ctd1_median": float(np.median(ctd_arr[:, 0])),
            "ctd2_median": float(np.median(ctd_arr[:, 1])),
            "ctd1_width": float(np.quantile(ctd_arr[:, 0], 0.975) - np.quantile(ctd_arr[:, 0], 0.025)),
            "dev_median": float(np.median(dev_arr)),
            "loss_final": chain.history.losses[-1],
        })

    arr_all = np.stack(all_cifs, axis=0) if all_cifs else np.empty((0,))
    if arr_all.ndim == 4 and arr_all.shape[0] > 0:
        q_all = np.quantile(arr_all, [0.025, 0.5, 0.975], axis=0)
        cov_all = coverage_from_quantiles(q_all, true_cif)
        pooled = {"cif_coverage": cov_all["overall"], "cif_width": cov_all["mean_interval_width"]}
    else:
        pooled = {"cif_coverage": float("nan"), "cif_width": float("nan")}

    # R-hat from the already-collected per-chain metrics.
    valid = [cr for cr in chain_results if cr is not None]
    min_draws = min(cr["n_draws"] for cr in valid) if valid else 0
    if min_draws >= 2 and len(valid) >= 2:
        ctd1_chains, dev_chains = [], []
        for c_idx, chain in enumerate(sampler.chains):
            c_ctd1, c_dev = [], []
            for _, m in _iter_samples(lead, chain.posterior_samples):
                c_ctd1.append(float(competing_risk_c_index(
                    m.predict_cif(test), test_durations, test_events)[0]))
                c_dev.append(float(predictive_deviance(
                    m.predict_interval_probs(test).numpy(), test_idx, test_events)))
                if len(c_ctd1) >= min_draws:
                    break
            ctd1_chains.append(c_ctd1[:min_draws])
            dev_chains.append(c_dev[:min_draws])
        pooled["rhat_ctd1"] = gelman_rubin_rhat(np.array(ctd1_chains))
        pooled["rhat_dev"] = gelman_rubin_rhat(np.array(dev_chains))
    else:
        pooled["rhat_ctd1"] = float("nan")
        pooled["rhat_dev"] = float("nan")

    return chain_results, pooled


def print_results(label, chain_results, pooled):
    print(f"\n  {'chain':>5s} | {'draws':>5s} | {'CIF cov':>8s} | {'CIF wid':>8s} | "
          f"{'Ctd1':>7s} | {'Dev':>8s} | {'loss':>8s}")
    print(f"  {'-'*65}")
    for c, cr in enumerate(chain_results):
        if cr is None:
            print(f"  {c:>5d} | {'---':>5s} |")
            continue
        print(f"  {c:>5d} | {cr['n_draws']:>5d} | {cr['cif_coverage']:8.4f} | {cr['cif_width']:8.4f} | "
              f"{cr['ctd1_median']:7.4f} | {cr['dev_median']:8.3f} | {cr['loss_final']:8.3f}")
    rhat_str = f"R̂={pooled.get('rhat_ctd1', float('nan')):.2f}"
    print(f"  {'pool':>5s} |       | {pooled['cif_coverage']:8.4f} | {pooled['cif_width']:8.4f} | "
          f"{'':>7s} | {'':>8s} | {rhat_str:>8s}")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    # --- Generate separate teacher and student cohorts ---
    print("=== New Simulation: teacher N={}, student N={} ===".format(N_TEACHER, N_STUDENT))
    teacher_data = simulate_competing_risks(n=N_TEACHER, seed=SEED, censor_max=0.05)
    student_data = simulate_competing_risks(n=N_STUDENT, seed=SEED + 1, censor_max=0.05)
    all_features = [c for c in teacher_data.columns if c.startswith("x")]

    # Student train/test split
    train_s, test_s = train_test_split(student_data, test_size=0.25, random_state=SEED)
    test_durations = test_s["duration"].to_numpy()
    test_events = test_s["event"].to_numpy()

    # Shared time grid from combined data
    combined_durations = np.concatenate([teacher_data["duration"].values, train_s["duration"].values])
    time_grid = fit_time_grid(combined_durations, NUM_DURATIONS)
    test_idx = transform_durations(test_durations, time_grid)
    true_cif = true_cif_at_grid(test_s, time_grid, num_risks=NUM_RISKS)

    print(f"  Teacher train: {len(teacher_data)}, Student train: {len(train_s)}, test: {len(test_s)}")
    print(f"  True CIF range: [{true_cif.min():.4f}, {true_cif.max():.4f}]")

    # --- Train teachers ---
    print("\n--- Training teachers ---", flush=True)

    t0 = time.time()
    teacher_perfect = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(teacher_data, feature_cols=all_features)
    print(f"  Perfect teacher (12 features): loss={teacher_perfect.history.losses[-1]:.4f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    degraded_features = all_features[:8]
    teacher_degraded = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(teacher_data, feature_cols=degraded_features)
    print(f"  Degraded teacher (8 features): loss={teacher_degraded.history.losses[-1]:.4f} ({time.time()-t0:.1f}s)")

    # --- AdamW baselines (point estimates) ---
    print("\n--- AdamW baselines on student cohort ---", flush=True)

    adamw_common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=WARMSTART_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid, optimizer="adamw",
    )
    adamw_results = {}
    for label, teacher, teacher_type in [
        ("internal", None, None),
        ("perfect", teacher_perfect, "competing"),
        ("degraded", teacher_degraded, "competing"),
    ]:
        t0 = time.time()
        if teacher is None:
            m = DiscreteSurvivalModel(**adamw_common).fit(train_s, feature_cols=all_features)
        else:
            m = DiSKDStudent(
                teacher_model=teacher, teacher_type=teacher_type, eta=1.0, temperature=2.0,
                **adamw_common,
            ).fit(train_s, feature_cols=all_features)
        cif = m.predict_cif(test_s)
        ctd = competing_risk_c_index(cif, test_durations, test_events)
        dev = predictive_deviance(m.predict_interval_probs(test_s).numpy(), test_idx, test_events)
        adamw_results[label] = {"ctd": ctd, "dev": dev, "loss": m.history.losses[-1]}
        print(f"  {label:>10s}: Ctd1={ctd[0]:.4f} Ctd2={ctd[1]:.4f} dev={dev:.3f} loss={m.history.losses[-1]:.4f} ({time.time()-t0:.1f}s)")

    # Save the warm-start weights (from perfect-teacher AdamW)
    pretrain = DiSKDStudent(
        teacher_model=teacher_perfect, teacher_type="competing", eta=1.0, temperature=2.0,
        **adamw_common,
    )
    pretrain.fit(train_s, feature_cols=all_features)
    pretrained_state = {k: v.clone() for k, v in pretrain.net.state_dict().items()}

    # --- SGLD configurations ---
    print(f"\n--- SGLD configurations (CR->CR perfect teacher, {N_CHAINS} chains) ---", flush=True)
    print(f"  cyclical: cycle_length={CYCLE_LENGTH}ep, burn_in_cycles={BURN_IN_CYCLES}, "
          f"prior_sigma={PRIOR_SIGMA}")

    sgld_base = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher_perfect, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=time_grid, optimizer="sgld",
        sgld_step_size=SGLD_STEP_SIZE, sgld_drift_mode="literal",
    )

    configs = {
        "cyc-1draw": dict(
            **sgld_base,
            sgld_schedule="cyclical", sgld_cycle_length=CYCLE_LENGTH,
            sgld_burn_in_cycles=BURN_IN_CYCLES, sgld_draws_per_cycle=1,
            sgld_prior_sigma=PRIOR_SIGMA,
        ),
        "cyc-5draw": dict(
            **sgld_base,
            sgld_schedule="cyclical", sgld_cycle_length=CYCLE_LENGTH,
            sgld_burn_in_cycles=BURN_IN_CYCLES, sgld_draws_per_cycle=5,
            sgld_prior_sigma=PRIOR_SIGMA,
        ),
        "monotone": dict(
            **sgld_base,
            sgld_schedule="polynomial",
            sgld_final_step_size=1e-5, sgld_gamma=SGLD_GAMMA,
            sgld_burnin_epochs=SGLD_EPOCHS // 2,
            sgld_samples_per_chain=500,
        ),
    }

    all_results = {}
    for cfg_label, cfg in configs.items():
        t0 = time.time()
        is_warm = cfg_label.startswith("cyc")
        print(f"\n  [{cfg_label}] {'warm-start' if is_warm else 'cold-start'}...", flush=True)

        base_model = DiSKDStudent(**cfg)
        sampler = MultiChainSampler(base_model, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))

        if is_warm:
            # Warm-start: inject pretrained weights before SGLD fit
            for chain_id, seed in enumerate(sampler.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                chain = copy.deepcopy(base_model)
                chain._prepare_fit_data(train_s, all_features, "duration", "event")
                chain._build_net(len(all_features))
                chain.net.load_state_dict(copy.deepcopy(pretrained_state))
                chain.fit(train_s, feature_cols=all_features)
                sampler.chains.append(chain)
                sampler.posterior_samples.extend(chain.posterior_samples)
        else:
            sampler.fit(train_s, feature_cols=all_features)

        elapsed = time.time() - t0
        n_draws = sum(len(c.posterior_samples) for c in sampler.chains)
        print(f"    done ({elapsed:.0f}s, {n_draws} total draws)", flush=True)

        chain_res, pooled = evaluate_sampler(
            sampler, test_s, true_cif, test_durations, test_events, test_idx)
        all_results[cfg_label] = (chain_res, pooled, sampler)
        print_results(cfg_label, chain_res, pooled)

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nAdamW point estimates (student cohort N={N_STUDENT}):")
    for label, r in adamw_results.items():
        print(f"  {label:>10s}: Ctd1={r['ctd'][0]:.4f} Ctd2={r['ctd'][1]:.4f} dev={r['dev']:.3f}")
    print(f"\nSGLD per-chain median CIF coverage:")
    for cfg_label, (chain_res, pooled, _) in all_results.items():
        valid = [cr for cr in chain_res if cr is not None]
        if valid:
            covs = [cr["cif_coverage"] for cr in valid]
            wids = [cr["cif_width"] for cr in valid]
            print(f"  {cfg_label:>12s}: per-chain cov={np.median(covs):.3f} [{min(covs):.3f}-{max(covs):.3f}]  "
                  f"width={np.median(wids):.3f}  pooled cov={pooled['cif_coverage']:.3f}  "
                  f"R̂={pooled.get('rhat_ctd1', float('nan')):.2f}")

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w") as f:
        f.write(f"# Cyclical SGLD probe: teacher N={N_TEACHER}, student N={N_STUDENT}\n")
        f.write(f"# AdamW baselines:\n")
        for label, r in adamw_results.items():
            f.write(f"# {label}: Ctd1={r['ctd'][0]:.4f} Ctd2={r['ctd'][1]:.4f} dev={r['dev']:.3f}\n")
        for cfg_label, (chain_res, pooled, _) in all_results.items():
            for c, cr in enumerate(chain_res):
                if cr is None:
                    continue
                f.write(f"{cfg_label}\t{c}\t{cr['cif_coverage']:.4f}\t{cr['cif_width']:.4f}\t"
                        f"{cr['ctd1_median']:.4f}\t{cr['dev_median']:.3f}\t{cr['loss_final']:.3f}\n")
            f.write(f"{cfg_label}\tpooled\t{pooled['cif_coverage']:.4f}\t{pooled['cif_width']:.4f}\t"
                    f"rhat={pooled.get('rhat_ctd1', float('nan')):.3f}\n")
    print(f"\nTable saved to {OUT_TABLE}")


if __name__ == "__main__":
    main()
