"""Study A: SGLD parameter-space diagnostic.

This probe tests whether weak SGLD behavior is primarily caused by full-network
parameter-space geometry or by the generalized posterior / loss calibration.

For each hidden dimension it compares:
  - full-network warm-start SGLD
  - last-layer-only warm-start SGLD (feature extractor frozen, head sampled)

Default settings are intended for a cluster run. For a quick smoke test, reduce
TEACHER_EPOCHS, ADAMW_EPOCHS, SGLD_EPOCHS, N_CHAINS, SAMPLES_PER_CHAIN, and
HIDDEN_DIMS via environment variables.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (  # noqa: E402
    DiSKDStudent,
    DiscreteSurvivalModel,
    WarmStartMultiChainSampler,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid  # noqa: E402
from diskd.uncertainty import _iter_samples, effective_sample_size, gelman_rubin_rhat  # noqa: E402


TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
TEST_N = int(os.environ.get("TEST_N", 500))
TEACHER_FEATURE_QUALITY = os.environ.get("TEACHER_FEATURE_QUALITY", "full")

NUM_RISKS = 2
NUM_DURATIONS = 12
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
TEACHER_HIDDEN = int(os.environ.get("TEACHER_HIDDEN", 128))
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))

HIDDEN_DIMS = [int(x) for x in os.environ.get("HIDDEN_DIMS", "8,16,32,64,128").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]

SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 2e-4))
EPS_T = float(os.environ.get("EPS_T", 2e-4))
GAMMA = float(os.environ.get("GAMMA", 0.55))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", 1.0))

OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_study_a")))


class LastLayerSGLDDiSKDStudent(DiSKDStudent):
    """DiSKD student that samples only the final head during SGLD.

    The time-MLP and transformer backbones expose a module named ``head``.
    Freezing everything except ``head`` gives a direct diagnostic of whether
    full parameter-space geometry is responsible for poor SGLD diagnostics.
    """

    def _build_optimizer(self, n_train: int | None = None, total_steps: int | None = None):
        if self.net is None:
            raise RuntimeError("Network is not built.")
        if self.optimizer == "sgld":
            n_head = 0
            for name, param in self.net.named_parameters():
                keep = name.startswith("head.")
                param.requires_grad_(keep)
                n_head += int(keep)
            if n_head == 0:
                raise RuntimeError("Last-layer SGLD requires a backbone with a 'head' module.")
        return super()._build_optimizer(n_train=n_train, total_steps=total_steps)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def setup(seed: int):
    cohorts = simulate_competing_risk_cohorts(
        n_teacher=TEACHER_N,
        n_student=STUDENT_N,
        n_test=TEST_N,
        seed=seed,
        teacher_feature_quality=TEACHER_FEATURE_QUALITY,
    )
    durations = np.concatenate([cohorts.teacher["duration"].values, cohorts.student["duration"].values])
    time_grid = fit_time_grid(durations, NUM_DURATIONS)
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        time_grid=time_grid,
    ).fit(cohorts.teacher, feature_cols=cohorts.teacher_features)
    true_cif = true_cif_at_grid(cohorts.test, time_grid, num_risks=NUM_RISKS)
    return cohorts, time_grid, teacher, true_cif


def train_map(cohorts, time_grid, teacher, hidden_dim: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DiSKDStudent(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=hidden_dim,
        epochs=ADAMW_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        time_grid=time_grid,
        optimizer="adamw",
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
    ).fit(cohorts.student, feature_cols=cohorts.student_features)
    if model.net is None:
        raise RuntimeError("MAP model was not fitted.")
    state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
    return model, state


def build_sgld(teacher, time_grid, hidden_dim: int, target: str):
    cls = LastLayerSGLDDiSKDStudent if target == "last_layer" else DiSKDStudent
    return cls(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=hidden_dim,
        epochs=SGLD_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        time_grid=time_grid,
        optimizer="sgld",
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
        sgld_step_size=EPS0,
        sgld_final_step_size=EPS_T,
        sgld_gamma=GAMMA,
        sgld_drift_mode="literal",
        sgld_noise_scale=NOISE_SCALE,
        sgld_burnin_epochs=BURNIN_EPOCHS,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )


def collect_chain_predictions(sampler, test, test_idx, test_events):
    lead = sampler.lead_chain
    chains = []
    for chain in sampler.chains:
        cifs, devs = [], []
        for _, sample_model in _iter_samples(lead, chain.posterior_samples):
            cifs.append(np.asarray(sample_model.predict_cif(test)))
            devs.append(
                predictive_deviance(
                    sample_model.predict_interval_probs(test).numpy(),
                    test_idx,
                    test_events,
                )
            )
        chains.append({"cifs": np.stack(cifs, axis=0), "dev": np.asarray(devs, dtype=float)})
    return chains


def chain_diagnostics(chains):
    rhats, esss = [], []
    for cause in range(NUM_RISKS):
        fn = np.stack([c["cifs"][:, :, cause, -1].mean(axis=1) for c in chains], axis=0)
        if fn.shape[0] >= 2 and fn.shape[1] >= 2:
            try:
                rhats.append(gelman_rubin_rhat(fn))
                esss.append(effective_sample_size(fn))
            except (ValueError, ZeroDivisionError):
                pass
    return (
        float(np.nanmax(rhats)) if rhats else float("nan"),
        float(np.nanmin(esss)) if esss else float("nan"),
    )


def evaluate_target(cohorts, time_grid, teacher, true_cif, map_state, hidden_dim: int, target: str, seed: int):
    test = cohorts.test
    test_dur = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)

    base = build_sgld(teacher, time_grid, hidden_dim, target)
    sampler = WarmStartMultiChainSampler(
        base,
        pretrained_state=copy.deepcopy(map_state),
        n_chains=N_CHAINS,
        seeds=[10_000 * seed + i for i in range(N_CHAINS)],
    ).fit(cohorts.student, feature_cols=cohorts.student_features)

    chains = collect_chain_predictions(sampler, test, test_idx, test_events)
    pooled = np.concatenate([c["cifs"] for c in chains], axis=0)
    q = np.quantile(pooled, [0.025, 0.5, 0.975], axis=0)
    cov = coverage_from_quantiles(q, true_cif)
    ctd = competing_risk_c_index(q[1], test_dur, test_events)
    rhat, ess = chain_diagnostics(chains)
    dev = float(np.mean([c["dev"].mean() for c in chains]))
    return {
        "target": target,
        "ctd1": float(ctd[0]),
        "ctd2": float(ctd[1]),
        "dev": dev,
        "coverage": float(cov["overall"]),
        "width": float(cov["mean_interval_width"]),
        "rhat": rhat,
        "ess": ess,
    }


def write_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "sgld_parameter_diagnostic.csv"
    fields = [
        "seed",
        "hidden_dim",
        "params",
        "target",
        "ctd1",
        "ctd2",
        "dev",
        "coverage",
        "width",
        "rhat",
        "ess",
    ]
    with csv.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")

    md = []
    md.append("# Study A: SGLD Parameter-Space Diagnostic")
    md.append("")
    md.append(f"Cell: teacher N={TEACHER_N}, student N={STUDENT_N}, test N={TEST_N}, "
              f"teacher quality={TEACHER_FEATURE_QUALITY}.")
    md.append(f"SGLD: eps={EPS0:.0e}->{EPS_T:.0e}, gamma={GAMMA}, noise scale={NOISE_SCALE}, "
              f"{N_CHAINS} chains x {SAMPLES_PER_CHAIN} draws, {SGLD_EPOCHS} epochs.")
    md.append("")
    md.append("Interpretation rule:")
    md.append("- last-layer much better than full-network: full parameter-space geometry is the problem.")
    md.append("- both weak: loss scale / generalized-posterior calibration is the likely problem.")
    md.append("- small hidden dimensions still weak: parameter count is not the main explanation.")
    md.append("")
    md.append("| Hidden | Params | Target | Ctd1 | Ctd2 | Dev | Coverage | Width | R-hat | ESS |")
    md.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    grouped = {}
    for row in rows:
        key = (row["hidden_dim"], row["params"], row["target"])
        grouped.setdefault(key, []).append(row)
    for (hidden, params, target), vals in sorted(grouped.items()):
        md.append(
            f"| {hidden} | {params} | {target} | "
            f"{np.median([v['ctd1'] for v in vals]):.3f} | "
            f"{np.median([v['ctd2'] for v in vals]):.3f} | "
            f"{np.median([v['dev'] for v in vals]):.3f} | "
            f"{np.median([v['coverage'] for v in vals]):.3f} | "
            f"{np.median([v['width'] for v in vals]):.3f} | "
            f"{np.nanmedian([v['rhat'] for v in vals]):.2f} | "
            f"{np.nanmedian([v['ess'] for v in vals]):.0f} |"
        )
    md.append("")
    md.append(f"Raw seed-level results: `{csv.name}`.")
    (OUT_DIR / "sgld_parameter_diagnostic_report.md").write_text("\n".join(md) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    print("=== Study A: SGLD parameter-space diagnostic ===")
    print(f"device={DEVICE}; seeds={SEEDS}; hidden_dims={HIDDEN_DIMS}")
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n--- seed {seed} ---", flush=True)
        cohorts, time_grid, teacher, true_cif = setup(seed)
        for hidden_dim in HIDDEN_DIMS:
            map_model, map_state = train_map(cohorts, time_grid, teacher, hidden_dim, seed)
            params = parameter_count(map_model.net)
            print(f"hidden={hidden_dim} params={params}", flush=True)
            for target in ("full", "last_layer"):
                result = evaluate_target(
                    cohorts,
                    time_grid,
                    teacher,
                    true_cif,
                    map_state,
                    hidden_dim,
                    target,
                    seed,
                )
                row = {"seed": seed, "hidden_dim": hidden_dim, "params": params, **result}
                rows.append(row)
                print(
                    f"  {target:10s} cov={row['coverage']:.3f} wid={row['width']:.3f} "
                    f"ctd1={row['ctd1']:.3f} dev={row['dev']:.3f} "
                    f"Rhat={row['rhat']:.2f} ESS={row['ess']:.0f}",
                    flush=True,
                )
        print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min", flush=True)
        write_outputs(rows)
    write_outputs(rows)
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
