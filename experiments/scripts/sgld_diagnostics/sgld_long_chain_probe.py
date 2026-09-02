"""Quick R-hat probe for the SGLD step-size schedule.

Runs a single distillation configuration (CR -> CR) at the same N=5000
synthetic setting. The drift convention is selected by the ``SGLD_MODE``
env var:

  - ``SGLD_MODE=welling_teh`` (default, Option A):
      sgld_step_size       = 3e-7  → effective Adam-equiv lr ~ 1e-3 at N=5000
      sgld_final_step_size = 3e-9  → effective Adam-equiv lr ~ 1e-5
  - ``SGLD_MODE=literal`` (Option B, no n_train multiplier in the drift):
      sgld_step_size       = 1e-3  (literal SGD-equivalent lr at t=0)
      sgld_final_step_size = 1e-5

Step-size env vars (``SGLD_STEP_SIZE``, ``SGLD_FINAL``) override the
per-mode defaults.

Use this probe before the full Table 1 job to confirm R-hat actually
drops. If Welling-Teh chains freeze in local modes (R-hat blows up,
within-chain variance ~ 0) try Option B, which scales the per-step noise
to drift ratio up by ~N and tends to mix substantially better.

Run from the repository root with:

    PYTHONPATH=src python examples/sgld_long_chain_probe.py
    PYTHONPATH=src SGLD_MODE=literal python examples/sgld_long_chain_probe.py
"""
from __future__ import annotations

import time

import numpy as np
import os
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
from diskd.uncertainty import _iter_samples


SEED = 7
N_SUBJECTS = 5000
NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 256

TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 15))
SGLD_EPOCHS_PER_CHAIN = int(os.environ.get("SGLD_EPOCHS", 500))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 20))
SGLD_MODE = os.environ.get("SGLD_MODE", "welling_teh")
# Per-mode step-size defaults. Override at submit time with SGLD_STEP_SIZE /
# SGLD_FINAL. Welling-Teh: eps ~ 1/N; literal: eps ~ direct lr.
_DEFAULT_STEPS = {
    "welling_teh": (3e-7, 3e-9),
    "literal":     (1e-3, 1e-5),
}
if SGLD_MODE not in _DEFAULT_STEPS:
    raise ValueError(f"SGLD_MODE must be one of {sorted(_DEFAULT_STEPS)}, got {SGLD_MODE!r}")
_default_init, _default_final = _DEFAULT_STEPS[SGLD_MODE]
SGLD_STEP_SIZE = float(os.environ.get("SGLD_STEP_SIZE", _default_init))
SGLD_FINAL_STEP_SIZE = float(os.environ.get("SGLD_FINAL", _default_final))
SGLD_GAMMA = float(os.environ.get("SGLD_GAMMA", 1.0))


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    data = simulate_competing_risks(n=N_SUBJECTS, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()

    print(f"Setup: N={N_SUBJECTS}, train={len(train)}, test={len(test)}, "
          f"epochs/chain={SGLD_EPOCHS_PER_CHAIN}, chains={N_CHAINS}, "
          f"samples/chain={SAMPLES_PER_CHAIN}", flush=True)
    print(f"  SGLD drift mode: {SGLD_MODE} "
          f"({'multiplies grad by N=n_train' if SGLD_MODE == 'welling_teh' else 'mean-batch grad, no N multiplier (Option B)'})",
          flush=True)
    print(f"  step_size 0 -> T: {SGLD_STEP_SIZE:.0e} -> {SGLD_FINAL_STEP_SIZE:.0e}  (gamma={SGLD_GAMMA})", flush=True)

    print("\n--- teacher (deterministic AdamW) ---", flush=True)
    t0 = time.time()
    time_grid = fit_time_grid(train["duration"].values, NUM_DURATIONS)
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=48,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(train, feature_cols=all_features)
    test_idx = transform_durations(test_durations, time_grid)
    print(f"  teacher NLL = {teacher.history.losses[-1]:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n--- SGLD multi-chain CR -> CR (this is the slow part) ---", flush=True)
    t0 = time.time()
    base = DiSKDStudent(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
        epochs=SGLD_EPOCHS_PER_CHAIN, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=time_grid, optimizer="sgld",
        sgld_step_size=SGLD_STEP_SIZE,
        sgld_final_step_size=SGLD_FINAL_STEP_SIZE,
        sgld_gamma=SGLD_GAMMA,
        sgld_burnin_epochs=SGLD_EPOCHS_PER_CHAIN // 2,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
        sgld_drift_mode=SGLD_MODE,
    )
    sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
    sampler.fit(train, feature_cols=student_features)
    train_time = time.time() - t0
    print(f"  training done  ({train_time:.0f}s, ~{train_time / N_CHAINS:.0f}s/chain)", flush=True)
    print(f"  posterior samples collected: {len(sampler.posterior_samples)}", flush=True)

    print(f"\n--- per-chain metric evaluation ---", flush=True)
    t0 = time.time()
    ctd_by_chain = np.zeros((N_CHAINS, SAMPLES_PER_CHAIN, NUM_RISKS))
    dev_by_chain = np.zeros((N_CHAINS, SAMPLES_PER_CHAIN))
    lead = sampler.lead_chain
    for c_idx, chain in enumerate(sampler.chains):
        for s_idx, (_, m) in enumerate(_iter_samples(lead, chain.posterior_samples)):
            ctd_by_chain[c_idx, s_idx] = competing_risk_c_index(
                m.predict_cif(test), test_durations, test_events
            )
            dev_by_chain[c_idx, s_idx] = predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_events
            )
    print(f"  evaluation done ({time.time()-t0:.1f}s)", flush=True)

    print("\n--- R-hat / ESS diagnostics across chains ---", flush=True)
    for name, arr in (("Ctd cause 1", ctd_by_chain[:, :, 0]),
                      ("Ctd cause 2", ctd_by_chain[:, :, 1]),
                      ("Deviance",    dev_by_chain)):
        rhat = gelman_rubin_rhat(arr)
        ess = effective_sample_size(arr)
        flat = arr.reshape(-1)
        q = np.quantile(flat, [0.025, 0.5, 0.975])
        print(f"  {name:14s}  median={q[1]:8.4f}  CI=[{q[0]:.4f}, {q[2]:.4f}]  "
              f"R-hat={rhat:.3f}  ESS={ess:6.1f}  "
              f"(ESS / {N_CHAINS * SAMPLES_PER_CHAIN} total draws)", flush=True)

    print("\n--- per-chain trajectory of the final Ctd (cause 1) ---", flush=True)
    for c_idx, chain in enumerate(sampler.chains):
        last5 = ctd_by_chain[c_idx, -5:, 0]
        print(f"  chain {c_idx}: last 5 Ctd1 draws = {last5}", flush=True)


if __name__ == "__main__":
    main()
