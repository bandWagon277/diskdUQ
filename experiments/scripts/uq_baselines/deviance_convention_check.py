"""Quantify the deviance reporting-convention gap for warm-start SGLD.

The proposal reports SGLD C_td as the metric of the *averaged* (pooled-median) CIF
(BMA-style) but SGLD deviance as the *average of per-draw* deviances. Because deviance
is a convex proper score, deviance(averaged prediction) <= mean(per-draw deviance), so
the per-draw-average convention inflates the reported deviance. This script computes
both conventions side by side at the headline cell so the gap is explicit.
"""
from __future__ import annotations

import os
import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent, DiscreteSurvivalModel, WarmStartMultiChainSampler,
    competing_risk_c_index, fit_time_grid, predictive_deviance,
    simulate_competing_risk_cohorts, transform_durations,
)
from diskd.uncertainty import _iter_samples

SEED = int(os.environ.get("SGLD_SEED", 42))
TEACHER_N, STUDENT_N, TEST_N = 5000, 500, 500
NUM_RISKS, NUM_DURATIONS = 2, 12
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))

torch.set_num_threads(1)
cohorts = simulate_competing_risk_cohorts(
    n_teacher=TEACHER_N, n_student=STUDENT_N, n_test=TEST_N,
    seed=SEED, teacher_feature_quality="full")
combined = np.concatenate([cohorts.teacher["duration"].values, cohorts.student["duration"].values])
time_grid = fit_time_grid(combined, NUM_DURATIONS)
teacher = DiscreteSurvivalModel(
    num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=128,
    epochs=100, batch_size=64, device=DEVICE, time_grid=time_grid,
).fit(cohorts.teacher, feature_cols=cohorts.teacher_features)

test = cohorts.test
test_dur = test["duration"].to_numpy(); test_ev = test["event"].to_numpy()
test_idx = transform_durations(test_dur, time_grid)

COMMON = dict(num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=32,
              epochs=SGLD_EPOCHS, batch_size=64, device=DEVICE, time_grid=time_grid,
              optimizer="sgld", sgld_step_size=2e-4, sgld_final_step_size=2e-4,
              sgld_gamma=0.55, sgld_drift_mode="literal", sgld_noise_scale=1.0,
              sgld_burnin_epochs=0, sgld_samples_per_chain=SAMPLES_PER_CHAIN)


def build(method):
    if method == "internal":
        return DiscreteSurvivalModel(**COMMON)
    return DiSKDStudent(teacher_model=teacher, teacher_type="competing",
                        eta=1.0, temperature=2.0, **COMMON)


def adamw_map(method):
    common = dict(num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=32,
                  epochs=50, batch_size=64, device=DEVICE, time_grid=time_grid, optimizer="adamw")
    torch.manual_seed(1000 * SEED); np.random.seed(1000 * SEED)
    if method == "internal":
        m = DiscreteSurvivalModel(**common)
    else:
        m = DiSKDStudent(teacher_model=teacher, teacher_type="competing",
                         eta=1.0, temperature=2.0, **common)
    m.fit(cohorts.student, feature_cols=cohorts.student_features)
    dev = float(predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_ev))
    cif = np.asarray(m.predict_cif(test))
    ctd = competing_risk_c_index(cif, test_dur, test_ev)
    return float(ctd[0]), dev, {k: v.clone() for k, v in m.net.state_dict().items()}


print(f"=== Deviance convention check (seed {SEED}, full teacher) ===")
print(f"{'method':10s} | AdamW dev | per-draw mean dev | BMA dev (deviance of mean prob) | "
      f"per-draw mean Ctd1 | BMA Ctd1")
for method in ["internal", "cr_to_cr"]:
    a_ctd1, a_dev, pretrained = adamw_map(method)
    base = build(method)
    sampler = WarmStartMultiChainSampler(
        base, pretrained, n_chains=N_CHAINS, seeds=[100 * SEED + ci for ci in range(N_CHAINS)])
    sampler.fit(cohorts.student, feature_cols=cohorts.student_features)
    lead = sampler.lead_chain
    perdraw_dev, perdraw_ctd1, ip_accum, cif_accum = [], [], [], []
    for chain in sampler.chains:
        for _, m in _iter_samples(lead, chain.posterior_samples):
            ip = m.predict_interval_probs(test).numpy()
            cif = np.asarray(m.predict_cif(test))
            ip_accum.append(ip); cif_accum.append(cif)
            perdraw_dev.append(float(predictive_deviance(ip, test_idx, test_ev)))
            perdraw_ctd1.append(float(competing_risk_c_index(cif, test_dur, test_ev)[0]))
    perdraw_mean_dev = float(np.mean(perdraw_dev))                       # proposal's current convention
    bma_ip = np.mean(np.stack(ip_accum, axis=0), axis=0)                 # averaged predictive distribution
    bma_dev = float(predictive_deviance(bma_ip, test_idx, test_ev))      # consistent BMA convention
    bma_ctd1 = float(competing_risk_c_index(np.median(np.stack(cif_accum, 0), 0), test_dur, test_ev)[0])
    print(f"{method:10s} |   {a_dev:.3f}   |      {perdraw_mean_dev:.3f}       |             "
          f"{bma_dev:.3f}              |       {np.median(perdraw_ctd1):.3f}        |  {bma_ctd1:.3f}")
