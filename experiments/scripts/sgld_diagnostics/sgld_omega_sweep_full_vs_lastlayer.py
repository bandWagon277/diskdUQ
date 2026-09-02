"""Omega sweep for full-network vs last-layer-only SGLD.

This probe treats omega as the generalized-posterior learning-rate /
spread-calibration parameter:

    Pi_omega(theta | data, teacher) proportional to exp{-omega L_eta(theta)}.

In the repository's literal mean-loss SGLD convention, this is implemented by
multiplying the SGLD drift by ``omega`` while leaving the Gaussian diffusion
term unchanged. Thus omega < 1 is hotter/wider; omega > 1 is colder/tighter.

The script runs the headline synthetic cell and compares:
  - full-network warm-start SGLD
  - last-layer-only warm-start SGLD

For each omega it reports calibration-split coverage/width and independent
test metrics. Non-Bayesian UQ engines are not included here; this is purely a
Bayesian/generalized-Bayesian calibration diagnostic.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

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
CALIB_FRACTION = float(os.environ.get("CALIB_FRACTION", 0.2))

NUM_RISKS = 2
NUM_DURATIONS = 12
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
TEACHER_HIDDEN = int(os.environ.get("TEACHER_HIDDEN", 128))
STUDENT_HIDDEN = int(os.environ.get("STUDENT_HIDDEN", 32))
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))

SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
OMEGAS = [float(x) for x in os.environ.get("OMEGAS", "0.25,0.5,1.0,2.0").split(",")]
TARGETS = [x.strip() for x in os.environ.get("TARGETS", "full,last_layer").split(",") if x.strip()]
TARGET_COVERAGE = float(os.environ.get("TARGET_COVERAGE", 0.95))

SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 300))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 3))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 300))
EPS0 = float(os.environ.get("EPS0", 2e-4))
EPS_T = float(os.environ.get("EPS_T", 2e-4))
GAMMA = float(os.environ.get("GAMMA", 0.55))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", 1.0))

OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_omega_full_lastlayer")))


class OmegaDiSKDStudent(DiSKDStudent):
    """DiSKD student whose SGLD drift targets exp{-omega L_eta}."""

    def __init__(self, *args, sgld_omega: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if sgld_omega <= 0:
            raise ValueError("sgld_omega must be positive.")
        self.sgld_omega = float(sgld_omega)

    def _sgld_loss_scale(self) -> float:
        return self.sgld_omega * super()._sgld_loss_scale()


class LastLayerOmegaDiSKDStudent(OmegaDiSKDStudent):
    """Omega SGLD that samples only the final prediction head."""

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


def setup(seed: int):
    cohorts = simulate_competing_risk_cohorts(
        n_teacher=TEACHER_N,
        n_student=STUDENT_N,
        n_test=TEST_N,
        seed=seed,
        teacher_feature_quality=TEACHER_FEATURE_QUALITY,
    )
    train_student, calib_student = train_test_split(
        cohorts.student,
        test_size=CALIB_FRACTION,
        random_state=seed,
        shuffle=True,
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
    true_cal = true_cif_at_grid(calib_student, time_grid, num_risks=NUM_RISKS)
    true_test = true_cif_at_grid(cohorts.test, time_grid, num_risks=NUM_RISKS)
    return cohorts, train_student, calib_student, time_grid, teacher, true_cal, true_test


def train_map(train_df, feature_cols, time_grid, teacher, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DiSKDStudent(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=STUDENT_HIDDEN,
        epochs=ADAMW_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        time_grid=time_grid,
        optimizer="adamw",
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
    ).fit(train_df, feature_cols=feature_cols)
    if model.net is None:
        raise RuntimeError("MAP model was not fitted.")
    state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
    return state


def build_sgld_base(target: str, omega: float, time_grid, teacher):
    cls = LastLayerOmegaDiSKDStudent if target == "last_layer" else OmegaDiSKDStudent
    return cls(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=STUDENT_HIDDEN,
        epochs=SGLD_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        time_grid=time_grid,
        optimizer="sgld",
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
        sgld_omega=omega,
        sgld_step_size=EPS0,
        sgld_final_step_size=EPS_T,
        sgld_gamma=GAMMA,
        sgld_drift_mode="literal",
        sgld_noise_scale=NOISE_SCALE,
        sgld_burnin_epochs=BURNIN_EPOCHS,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )


def collect_predictions(sampler, data):
    lead = sampler.lead_chain
    chains = []
    all_cifs = []
    all_probs = []
    for chain in sampler.chains:
        chain_cifs = []
        for _, sample_model in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(sample_model.predict_cif(data))
            probs = sample_model.predict_interval_probs(data).numpy()
            chain_cifs.append(cif)
            all_cifs.append(cif)
            all_probs.append(probs)
        chains.append(np.stack(chain_cifs, axis=0))
    return np.stack(all_cifs, axis=0), np.stack(all_probs, axis=0), chains


def chain_diagnostics(chains):
    rhats, esss = [], []
    for cause in range(NUM_RISKS):
        fn = np.stack([chain[:, :, cause, -1].mean(axis=1) for chain in chains], axis=0)
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


def evaluate_samples(cifs, probs, truth, frame, time_grid, chains=None):
    q = np.quantile(cifs, [0.025, 0.5, 0.975], axis=0)
    cov = coverage_from_quantiles(q, truth)
    durations = frame["duration"].to_numpy()
    events = frame["event"].to_numpy()
    idx = transform_durations(durations, time_grid)
    ctd = competing_risk_c_index(q[1], durations, events)
    dev = predictive_deviance(probs.mean(axis=0), idx, events)
    rhat, ess = (float("nan"), float("nan")) if chains is None else chain_diagnostics(chains)
    return {
        "ctd1": float(ctd[0]),
        "ctd2": float(ctd[1]),
        "dev": float(dev),
        "coverage": float(cov["overall"]),
        "width": float(cov["mean_interval_width"]),
        "rhat": rhat,
        "ess": ess,
    }


def run_one(seed, target, omega, train_df, calib_df, test_df, feature_cols, time_grid, teacher, map_state, true_cal, true_test):
    base = build_sgld_base(target, omega, time_grid, teacher)
    sampler = WarmStartMultiChainSampler(
        base,
        pretrained_state=copy.deepcopy(map_state),
        n_chains=N_CHAINS,
        seeds=[70_000 * seed + int(1000 * omega) + i for i in range(N_CHAINS)],
    ).fit(train_df, feature_cols=feature_cols)

    cal_cifs, cal_probs, _ = collect_predictions(sampler, calib_df)
    test_cifs, test_probs, test_chains = collect_predictions(sampler, test_df)
    cal = evaluate_samples(cal_cifs, cal_probs, true_cal, calib_df, time_grid)
    test = evaluate_samples(test_cifs, test_probs, true_test, test_df, time_grid, chains=test_chains)
    return cal, test


def write_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "target",
        "omega",
        "cal_coverage",
        "cal_width",
        "test_coverage",
        "test_width",
        "test_ctd1",
        "test_ctd2",
        "test_dev",
        "test_rhat",
        "test_ess",
    ]
    csv = OUT_DIR / "sgld_omega_sweep_full_vs_lastlayer.csv"
    with csv.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")

    md = []
    md.append("# Omega Sweep: Full-Network vs Last-Layer SGLD")
    md.append("")
    md.append(f"Cell: teacher N={TEACHER_N}, student N={STUDENT_N}, test N={TEST_N}, "
              f"teacher quality={TEACHER_FEATURE_QUALITY}, calibration fraction={CALIB_FRACTION}.")
    md.append(f"SGLD: eps={EPS0:.0e}->{EPS_T:.0e}, gamma={GAMMA}, noise_scale={NOISE_SCALE}, "
              f"{N_CHAINS} chains x {SAMPLES_PER_CHAIN} draws, {SGLD_EPOCHS} epochs.")
    md.append("")
    md.append("Omega is implemented as a drift multiplier for `exp{-omega L_eta}`. "
              "Lower omega is hotter/wider; higher omega is colder/tighter.")
    md.append("")
    md.append("| Target | Omega | Cal cov | Cal width | Test cov | Test width | Ctd1 | Ctd2 | Dev | R-hat | ESS |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    grouped = {}
    for row in rows:
        key = (row["target"], row["omega"])
        grouped.setdefault(key, []).append(row)
    for (target, omega), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        md.append(
            f"| {target} | {omega:g} | "
            f"{np.median([v['cal_coverage'] for v in vals]):.3f} | "
            f"{np.median([v['cal_width'] for v in vals]):.3f} | "
            f"{np.median([v['test_coverage'] for v in vals]):.3f} | "
            f"{np.median([v['test_width'] for v in vals]):.3f} | "
            f"{np.median([v['test_ctd1'] for v in vals]):.3f} | "
            f"{np.median([v['test_ctd2'] for v in vals]):.3f} | "
            f"{np.median([v['test_dev'] for v in vals]):.3f} | "
            f"{np.nanmedian([v['test_rhat'] for v in vals]):.2f} | "
            f"{np.nanmedian([v['test_ess'] for v in vals]):.0f} |"
        )

    md.append("")
    md.append("Calibration-selected omega by target:")
    md.append("")
    md.append("| Target | Selected omega | Median cal cov | Median test cov | Median test width |")
    md.append("|---|---:|---:|---:|---:|")
    for target in sorted({r["target"] for r in rows}):
        candidates = []
        for omega in sorted({r["omega"] for r in rows if r["target"] == target}):
            vals = [r for r in rows if r["target"] == target and r["omega"] == omega]
            cal_cov = float(np.median([v["cal_coverage"] for v in vals]))
            test_cov = float(np.median([v["test_coverage"] for v in vals]))
            test_width = float(np.median([v["test_width"] for v in vals]))
            candidates.append((abs(cal_cov - TARGET_COVERAGE), test_width, omega, cal_cov, test_cov))
        _, test_width, omega, cal_cov, test_cov = min(candidates)
        md.append(f"| {target} | {omega:g} | {cal_cov:.3f} | {test_cov:.3f} | {test_width:.3f} |")
    md.append("")
    md.append(f"Raw seed-level results: `{csv.name}`.")
    (OUT_DIR / "sgld_omega_sweep_full_vs_lastlayer_report.md").write_text("\n".join(md) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    print("=== Omega sweep: full-network vs last-layer SGLD ===")
    print(f"device={DEVICE}; seeds={SEEDS}; omegas={OMEGAS}; targets={TARGETS}")
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n--- seed {seed} ---", flush=True)
        cohorts, train_df, calib_df, time_grid, teacher, true_cal, true_test = setup(seed)
        feature_cols = cohorts.student_features
        map_state = train_map(train_df, feature_cols, time_grid, teacher, seed)
        for target in TARGETS:
            if target not in {"full", "last_layer"}:
                raise ValueError("TARGETS entries must be 'full' or 'last_layer'.")
            for omega in OMEGAS:
                print(f"  target={target} omega={omega:g}", flush=True)
                cal, test = run_one(
                    seed,
                    target,
                    omega,
                    train_df,
                    calib_df,
                    cohorts.test,
                    feature_cols,
                    time_grid,
                    teacher,
                    map_state,
                    true_cal,
                    true_test,
                )
                row = {
                    "seed": seed,
                    "target": target,
                    "omega": omega,
                    "cal_coverage": cal["coverage"],
                    "cal_width": cal["width"],
                    "test_coverage": test["coverage"],
                    "test_width": test["width"],
                    "test_ctd1": test["ctd1"],
                    "test_ctd2": test["ctd2"],
                    "test_dev": test["dev"],
                    "test_rhat": test["rhat"],
                    "test_ess": test["ess"],
                }
                rows.append(row)
                print(
                    f"    cal cov={row['cal_coverage']:.3f} wid={row['cal_width']:.3f}; "
                    f"test cov={row['test_coverage']:.3f} wid={row['test_width']:.3f} "
                    f"ctd1={row['test_ctd1']:.3f} Rhat={row['test_rhat']:.2f}",
                    flush=True,
                )
                write_outputs(rows)
        print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    write_outputs(rows)
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
