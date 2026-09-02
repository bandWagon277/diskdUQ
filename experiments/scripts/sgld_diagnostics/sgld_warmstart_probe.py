"""Warm-start SGLD probe: initialize all chains from a converged AdamW solution.

Instead of random initialization, all 5 SGLD chains start from the same
pre-trained AdamW weights. This eliminates the between-chain mode-selection
problem (chains landing in different basins) and isolates the within-basin
posterior exploration.

Compares:
  - cold-start literal SGLD (current baseline, random init)
  - warm-start literal SGLD (all chains from AdamW-converged weights)

For each, reports BOTH pooled and per-chain CIF coverage so we can see
whether within-chain posterior spread is sufficient for valid coverage.

Run from the repository root:

    PYTHONPATH=src python examples/sgld_warmstart_probe.py
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

SEED = 7
N_SUBJECTS = 5000
NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 256

TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 30))
WARMSTART_EPOCHS = int(os.environ.get("WARMSTART_EPOCHS", 50))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 2000))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
SGLD_STEP_SIZE = float(os.environ.get("SGLD_STEP_SIZE", 1e-3))
SGLD_FINAL = float(os.environ.get("SGLD_FINAL", 1e-5))
SGLD_GAMMA = float(os.environ.get("SGLD_GAMMA", 1.0))
SGLD_MODE = os.environ.get("SGLD_MODE", "literal")

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))
OUT_TABLE = OUT_DIR / "sgld_warmstart_probe.txt"
OUT_FIGURE = OUT_DIR / "sgld_warmstart_probe.pdf"
OUT_FIGURE_PNG = OUT_FIGURE.with_suffix(".png")


class WarmStartMultiChainSampler:
    """Like MultiChainSampler but initializes every chain from a shared state_dict."""

    def __init__(self, base_model, pretrained_state, n_chains=5, seeds=None):
        self.base_model = base_model
        self.pretrained_state = pretrained_state
        self.n_chains = n_chains
        self.seeds = list(seeds) if seeds is not None else list(range(n_chains))
        self.chains = []
        self.posterior_samples = []

    def fit(self, data, feature_cols=None, **kwargs):
        self.chains = []
        self.posterior_samples = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            chain = copy.deepcopy(self.base_model)
            # Prepare data + build network (same as normal fit preamble)
            chain._prepare_fit_data_and_build_net(data, feature_cols)
            # Overwrite the freshly-initialized weights with the pretrained ones
            chain.net.load_state_dict(copy.deepcopy(self.pretrained_state))
            # Now run the actual SGLD training from the warm start
            chain.fit(data, feature_cols=feature_cols, **kwargs)
            self.chains.append(chain)
            self.posterior_samples.extend(chain.posterior_samples)
        return self

    @property
    def lead_chain(self):
        return self.chains[0]


def _add_prepare_method():
    """Patch DiscreteSurvivalModel to expose a 'prepare + build net' step
    without running the full fit, so warm-start can inject weights."""
    def _prepare_fit_data_and_build_net(self, data, feature_cols=None,
                                         duration_col="duration", event_col="event"):
        x, idx, events = self._prepare_fit_data(data, feature_cols, duration_col, event_col)
        self._build_net(x.shape[1])
        return x, idx, events
    DiscreteSurvivalModel._prepare_fit_data_and_build_net = _prepare_fit_data_and_build_net
    DiSKDStudent._prepare_fit_data_and_build_net = _prepare_fit_data_and_build_net

_add_prepare_method()


def evaluate_chain_coverage(sampler, test, true_cif, test_durations, test_events, test_idx):
    """Per-chain AND pooled CIF coverage + Ctd + deviance."""
    lead = sampler.lead_chain
    n_chains = len(sampler.chains)
    spc = len(sampler.chains[0].posterior_samples)

    all_cifs = []
    chain_results = []
    for c_idx, chain in enumerate(sampler.chains):
        chain_cifs = []
        ctds = []
        devs = []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            chain_cifs.append(cif)
            all_cifs.append(cif)
            ctds.append(competing_risk_c_index(cif, test_durations, test_events))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_events))
        arr = np.stack(chain_cifs, axis=0)
        q = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
        cov = coverage_from_quantiles(q, true_cif)
        ctd_arr = np.array(ctds)
        dev_arr = np.array(devs)
        chain_results.append({
            "cif_coverage": cov["overall"],
            "cif_width": cov["mean_interval_width"],
            "ctd1_median": float(np.median(ctd_arr[:, 0])),
            "ctd1_width": float(np.quantile(ctd_arr[:, 0], 0.975) - np.quantile(ctd_arr[:, 0], 0.025)),
            "dev_median": float(np.median(dev_arr)),
            "loss_final": chain.history.losses[-1],
        })

    # Pooled
    arr_all = np.stack(all_cifs, axis=0)
    q_all = np.quantile(arr_all, [0.025, 0.5, 0.975], axis=0)
    cov_all = coverage_from_quantiles(q_all, true_cif)

    return chain_results, {
        "cif_coverage": cov_all["overall"],
        "cif_width": cov_all["mean_interval_width"],
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    data = simulate_competing_risks(n=N_SUBJECTS, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()
    time_grid = fit_time_grid(train["duration"].values, NUM_DURATIONS)
    test_idx = transform_durations(test_durations, time_grid)
    true_cif = true_cif_at_grid(test, time_grid, num_risks=NUM_RISKS)

    print(f"Setup: N={N_SUBJECTS}, train={len(train)}, test={len(test)}")
    print(f"SGLD: mode={SGLD_MODE}, eps {SGLD_STEP_SIZE:.0e}->{SGLD_FINAL:.0e}, "
          f"gamma={SGLD_GAMMA}, epochs={SGLD_EPOCHS}")
    print(f"Chains={N_CHAINS}, samples/chain={SAMPLES_PER_CHAIN}")
    print(f"Warm-start: {WARMSTART_EPOCHS} AdamW epochs")

    # --- Shared teacher ---
    print("\n--- Teacher (AdamW) ---", flush=True)
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=48,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid,
    ).fit(train, feature_cols=all_features)
    print(f"  NLL={teacher.history.losses[-1]:.4f}")

    # --- Warm-start: pre-train student with AdamW ---
    print(f"\n--- Pre-training student with AdamW ({WARMSTART_EPOCHS} epochs) ---", flush=True)
    t0 = time.time()
    pretrain = DiSKDStudent(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
        epochs=WARMSTART_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=time_grid, optimizer="adamw",
    )
    pretrain.fit(train, feature_cols=student_features)
    pretrained_state = {k: v.clone() for k, v in pretrain.net.state_dict().items()}
    print(f"  AdamW loss={pretrain.history.losses[-1]:.4f}  ({time.time()-t0:.1f}s)")

    # --- SGLD common config ---
    sgld_common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
        epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=time_grid, optimizer="sgld",
        sgld_step_size=SGLD_STEP_SIZE, sgld_final_step_size=SGLD_FINAL,
        sgld_gamma=SGLD_GAMMA, sgld_burnin_epochs=SGLD_EPOCHS // 2,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN, sgld_drift_mode=SGLD_MODE,
    )

    results_all = {}

    # --- Cold-start SGLD (baseline) ---
    print(f"\n--- Cold-start SGLD ({N_CHAINS} chains) ---", flush=True)
    t0 = time.time()
    base_cold = DiSKDStudent(**sgld_common)
    sampler_cold = MultiChainSampler(base_cold, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
    sampler_cold.fit(train, feature_cols=student_features)
    elapsed_cold = time.time() - t0
    print(f"  Training done ({elapsed_cold:.0f}s). Evaluating...", flush=True)
    chains_cold, pooled_cold = evaluate_chain_coverage(
        sampler_cold, test, true_cif, test_durations, test_events, test_idx)
    results_all["cold"] = (chains_cold, pooled_cold, sampler_cold)

    # --- Warm-start SGLD ---
    print(f"\n--- Warm-start SGLD ({N_CHAINS} chains from AdamW solution) ---", flush=True)
    t0 = time.time()
    base_warm = DiSKDStudent(**sgld_common)
    sampler_warm = WarmStartMultiChainSampler(
        base_warm, pretrained_state, n_chains=N_CHAINS, seeds=list(range(100, 100 + N_CHAINS)))
    sampler_warm.fit(train, feature_cols=student_features)
    elapsed_warm = time.time() - t0
    print(f"  Training done ({elapsed_warm:.0f}s). Evaluating...", flush=True)
    chains_warm, pooled_warm = evaluate_chain_coverage(
        sampler_warm, test, true_cif, test_durations, test_events, test_idx)
    results_all["warm"] = (chains_warm, pooled_warm, sampler_warm)

    # --- Print comparison table ---
    print(f"\n{'='*100}")
    print(f"{'':>12s} | {'chain':>5s} | {'CIF cov':>8s} | {'CIF wid':>8s} | "
          f"{'Ctd1 med':>8s} | {'Ctd1 wid':>8s} | {'Dev med':>8s} | {'loss[-1]':>9s}")
    print(f"{'-'*100}")
    for label, (chains, pooled, _) in results_all.items():
        for c, cr in enumerate(chains):
            print(f"{label:>12s} | {c:>5d} | {cr['cif_coverage']:8.4f} | {cr['cif_width']:8.4f} | "
                  f"{cr['ctd1_median']:8.4f} | {cr['ctd1_width']:8.4f} | "
                  f"{cr['dev_median']:8.4f} | {cr['loss_final']:9.4f}")
        print(f"{label:>12s} | {'pool':>5s} | {pooled['cif_coverage']:8.4f} | {pooled['cif_width']:8.4f} |")
        print(f"{'-'*100}")
    print(f"{'='*100}")

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w") as f:
        f.write(f"# Warm-start probe: {SGLD_MODE}, eps {SGLD_STEP_SIZE:.0e}->{SGLD_FINAL:.0e}, "
                f"gamma={SGLD_GAMMA}, {SGLD_EPOCHS} ep, {N_CHAINS}x{SAMPLES_PER_CHAIN}\n")
        for label, (chains, pooled, _) in results_all.items():
            for c, cr in enumerate(chains):
                f.write(f"{label}\t{c}\t{cr['cif_coverage']:.4f}\t{cr['cif_width']:.4f}\t"
                        f"{cr['ctd1_median']:.4f}\t{cr['dev_median']:.4f}\t{cr['loss_final']:.4f}\n")
            f.write(f"{label}\tpooled\t{pooled['cif_coverage']:.4f}\t{pooled['cif_width']:.4f}\n")
    print(f"Table saved to {OUT_TABLE}")

    # --- Trace figure ---
    make_warmstart_figure(results_all, OUT_FIGURE)
    make_warmstart_figure(results_all, OUT_FIGURE_PNG)
    print(f"Figure saved to {OUT_FIGURE} (and .png)")


def make_warmstart_figure(results_all, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    burnin = SGLD_EPOCHS // 2

    for row, (label, (chains, pooled, sampler)) in enumerate(results_all.items()):
        # Col 0: per-chain training loss
        ax = axes[row, 0]
        for c, chain in enumerate(sampler.chains):
            ax.plot(chain.history.losses, lw=0.7, alpha=0.8, label=f"chain {c}")
        if len(sampler.chains[0].history.losses) > burnin:
            ax.axvspan(burnin, len(sampler.chains[0].history.losses) - 1,
                       color="#a1d99b", alpha=0.25, lw=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title(f"{label}-start: per-chain loss")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(ls=":", alpha=0.5)

        # Col 1: per-chain CIF coverage bar chart
        ax = axes[row, 1]
        covs = [cr["cif_coverage"] for cr in chains]
        colors = ["#2ca02c" if c > 0.9 else "#d62728" for c in covs]
        ax.bar(range(len(covs)), covs, color=colors)
        ax.axhline(0.95, color="gray", ls="--", lw=0.8)
        ax.axhline(pooled["cif_coverage"], color="blue", ls="-", lw=1.2,
                    label=f"pooled={pooled['cif_coverage']:.3f}")
        ax.set_xlabel("chain")
        ax.set_ylabel("CIF coverage")
        ax.set_title(f"{label}-start: per-chain CIF coverage")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(axis="y", ls=":", alpha=0.5)

        # Col 2: per-chain CIF width bar chart
        ax = axes[row, 2]
        widths = [cr["cif_width"] for cr in chains]
        ax.bar(range(len(widths)), widths, color="#ff7f0e")
        ax.axhline(pooled["cif_width"], color="blue", ls="-", lw=1.2,
                    label=f"pooled={pooled['cif_width']:.3f}")
        ax.set_xlabel("chain")
        ax.set_ylabel("CIF width")
        ax.set_title(f"{label}-start: per-chain CIF width")
        ax.legend(fontsize=7)
        ax.grid(axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        f"Cold-start vs Warm-start SGLD — {SGLD_MODE}, "
        f"ε={SGLD_STEP_SIZE:.0e}→{SGLD_FINAL:.0e}, γ={SGLD_GAMMA}, "
        f"{SGLD_EPOCHS} ep, {N_CHAINS}×{SAMPLES_PER_CHAIN}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
