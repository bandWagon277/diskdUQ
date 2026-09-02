"""Study B/C: headline-cell UQ engine comparator and scalar calibration.

This probe is designed for a proposal discussion, not for a final paper table.
It runs the current headline cell and compares practical uncertainty engines:

  - deep ensemble
  - patient bootstrap ensemble
  - MC dropout
  - late AdamW trajectory ensemble ("SWAG-lite" diagnostic, not true SWAG)
  - warm-start SGLD

It also applies a synthetic oracle scalar calibration on a held-out calibration
split, using the closed-form true CIF. This is intentionally labeled oracle:
it is a fast diagnostic showing whether undercoverage is mostly a scalar spread
problem. A deployable version would replace this with observed-data/conformal
calibration under censoring.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
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
from diskd.uncertainty import _iter_samples  # noqa: E402
from diskd.utils import competing_cif, competing_interval_probs  # noqa: E402


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
DROPOUT = float(os.environ.get("DROPOUT", 0.1))

SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]
ENGINES = [x.strip() for x in os.environ.get(
    "ENGINES", "deep,bootstrap,mc_dropout,trajectory,sgld"
).split(",") if x.strip()]

ENSEMBLE_SIZE = int(os.environ.get("ENSEMBLE_SIZE", 8))
MC_DROPOUT_SAMPLES = int(os.environ.get("MC_DROPOUT_SAMPLES", 300))
TRAJECTORY_SNAPSHOTS = int(os.environ.get("TRAJECTORY_SNAPSHOTS", 8))
TRAJECTORY_EPOCHS = int(os.environ.get("TRAJECTORY_EPOCHS", 5))
TRAJECTORY_LR = float(os.environ.get("TRAJECTORY_LR", 3e-4))

SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 0))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 2e-4))
EPS_T = float(os.environ.get("EPS_T", 2e-4))
GAMMA = float(os.environ.get("GAMMA", 0.55))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", 1.0))

TARGET_COVERAGE = float(os.environ.get("TARGET_COVERAGE", 0.95))
WIDEN_ONLY = bool(int(os.environ.get("WIDEN_ONLY", "1")))

OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_study_bc")))


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


def train_student(data, feature_cols, time_grid, teacher, seed: int, epochs: int | None = None, lr: float = 1e-3, warm_state=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DiSKDStudent(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=STUDENT_HIDDEN,
        dropout=DROPOUT,
        epochs=epochs or ADAMW_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=lr,
        device=DEVICE,
        time_grid=time_grid,
        optimizer="adamw",
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
    )
    if warm_state is not None:
        model._warm_start_state = copy.deepcopy(warm_state)
    model.fit(data, feature_cols=feature_cols)
    return model


def predict_cif_and_probs(model, data):
    cif = np.asarray(model.predict_cif(data))
    probs = model.predict_interval_probs(data).numpy()
    return cif, probs


def collect_model_samples(models, data):
    cifs, probs = [], []
    for model in models:
        cif, prob = predict_cif_and_probs(model, data)
        cifs.append(cif)
        probs.append(prob)
    return np.stack(cifs, axis=0), np.stack(probs, axis=0)


@torch.no_grad()
def mc_dropout_samples(model, data, n_samples: int):
    if model.net is None:
        raise RuntimeError("Model is not fitted.")
    x = torch.tensor(model._transform_features(data), dtype=torch.float32, device=model.device)
    cifs, probs = [], []
    original_training = model.net.training
    try:
        model.net.train()
        for _ in range(n_samples):
            logits = model._reshape_logits(model.net(x)).detach().cpu()
            prob = competing_interval_probs(logits)
            cif = competing_cif(prob)
            probs.append(prob.numpy())
            cifs.append(cif.numpy())
    finally:
        model.net.train(original_training)
    return np.stack(cifs, axis=0), np.stack(probs, axis=0)


def train_sgld_sampler(train_data, feature_cols, time_grid, teacher, map_state, seed: int):
    base = DiSKDStudent(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=STUDENT_HIDDEN,
        dropout=DROPOUT,
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
    sampler = WarmStartMultiChainSampler(
        base,
        pretrained_state=copy.deepcopy(map_state),
        n_chains=N_CHAINS,
        seeds=[50_000 * seed + i for i in range(N_CHAINS)],
    ).fit(train_data, feature_cols=feature_cols)
    return sampler


def samples_from_sgld_sampler(sampler, eval_data):
    lead = sampler.lead_chain
    cifs, probs = [], []
    for chain in sampler.chains:
        for _, sample_model in _iter_samples(lead, chain.posterior_samples):
            cif, prob = predict_cif_and_probs(sample_model, eval_data)
            cifs.append(cif)
            probs.append(prob)
    return np.stack(cifs, axis=0), np.stack(probs, axis=0)


def run_engine(engine, cohorts, train_student_df, calib_student, time_grid, teacher, seed: int):
    feature_cols = cohorts.student_features

    if engine == "deep":
        models = [
            train_student(train_student_df, feature_cols, time_grid, teacher, seed=1000 * seed + i)
            for i in range(ENSEMBLE_SIZE)
        ]
        cal = collect_model_samples(models, calib_student)
        test = collect_model_samples(models, cohorts.test)
        return cal, test

    if engine == "bootstrap":
        models = []
        rng = np.random.default_rng(seed)
        n = len(train_student_df)
        for i in range(ENSEMBLE_SIZE):
            idx = rng.integers(0, n, size=n)
            boot = train_student_df.iloc[idx].copy()
            models.append(train_student(boot, feature_cols, time_grid, teacher, seed=2000 * seed + i))
        return collect_model_samples(models, calib_student), collect_model_samples(models, cohorts.test)

    if engine == "mc_dropout":
        model = train_student(train_student_df, feature_cols, time_grid, teacher, seed=3000 * seed)
        return (
            mc_dropout_samples(model, calib_student, MC_DROPOUT_SAMPLES),
            mc_dropout_samples(model, cohorts.test, MC_DROPOUT_SAMPLES),
        )

    if engine == "trajectory":
        model = train_student(train_student_df, feature_cols, time_grid, teacher, seed=4000 * seed)
        state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
        snapshots = []
        for i in range(TRAJECTORY_SNAPSHOTS):
            model = train_student(
                train_student_df,
                feature_cols,
                time_grid,
                teacher,
                seed=4100 * seed + i,
                epochs=TRAJECTORY_EPOCHS,
                lr=TRAJECTORY_LR,
                warm_state=state,
            )
            state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
            snapshots.append(model)
        return collect_model_samples(snapshots, calib_student), collect_model_samples(snapshots, cohorts.test)

    if engine == "sgld":
        map_model = train_student(train_student_df, feature_cols, time_grid, teacher, seed=5000 * seed)
        map_state = {k: v.detach().cpu().clone() for k, v in map_model.net.state_dict().items()}
        sampler = train_sgld_sampler(train_student_df, feature_cols, time_grid, teacher, map_state, seed)
        return (
            samples_from_sgld_sampler(sampler, calib_student),
            samples_from_sgld_sampler(sampler, cohorts.test),
        )

    raise ValueError(f"Unknown engine {engine!r}")


def interval_quantiles(cif_samples):
    return np.quantile(cif_samples, [0.025, 0.5, 0.975], axis=0)


def oracle_scale_from_calibration(cal_q, true_cal):
    lo, med, hi = cal_q
    half_width = np.maximum((hi - lo) / 2.0, 1e-8)
    ratio = np.abs(true_cal - med) / half_width
    scale = float(np.quantile(ratio.reshape(-1), TARGET_COVERAGE))
    if WIDEN_ONLY:
        scale = max(1.0, scale)
    return scale


def apply_scale(test_q, scale: float):
    _, med, _ = test_q
    half_width = np.maximum((test_q[2] - test_q[0]) / 2.0, 1e-8)
    lo = np.clip(med - scale * half_width, 0.0, 1.0)
    hi = np.clip(med + scale * half_width, 0.0, 1.0)
    return np.stack([lo, med, hi], axis=0)


def summarize(engine, seed, test_q, prob_samples, true_test, test_df, time_grid, calibrated: bool, scale: float):
    cov = coverage_from_quantiles(test_q, true_test)
    test_dur = test_df["duration"].to_numpy()
    test_events = test_df["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    ctd = competing_risk_c_index(test_q[1], test_dur, test_events)
    prob_mean = np.mean(prob_samples, axis=0)
    dev = predictive_deviance(prob_mean, test_idx, test_events)
    width = test_q[2] - test_q[0]
    abs_err = np.abs(test_q[1] - true_test)
    if np.std(width) <= 0 or np.std(abs_err) <= 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(width.reshape(-1), abs_err.reshape(-1)).statistic)
    return {
        "seed": seed,
        "engine": engine,
        "calibrated": int(calibrated),
        "scale": scale,
        "ctd1": float(ctd[0]),
        "ctd2": float(ctd[1]),
        "dev": float(dev),
        "coverage": float(cov["overall"]),
        "width": float(cov["mean_interval_width"]),
        "rho_width_error": rho,
    }


def write_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "engine",
        "calibrated",
        "scale",
        "ctd1",
        "ctd2",
        "dev",
        "coverage",
        "width",
        "rho_width_error",
    ]
    csv = OUT_DIR / "uq_engine_comparator.csv"
    with csv.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")

    md = []
    md.append("# Study B/C: Headline-Cell UQ Engine Comparator")
    md.append("")
    md.append(f"Cell: teacher N={TEACHER_N}, student N={STUDENT_N}, test N={TEST_N}, "
              f"quality={TEACHER_FEATURE_QUALITY}, calibration fraction={CALIB_FRACTION}.")
    md.append(f"Engines: {', '.join(ENGINES)}.")
    md.append("")
    md.append("Calibration note: calibrated rows use a synthetic oracle scalar widening "
              "chosen on the calibration split against closed-form true CIF. This is a "
              "diagnostic for tomorrow's discussion, not a deployable conformal method.")
    md.append("")
    md.append("| Engine | Calibrated | Scale | Ctd1 | Ctd2 | Dev | Coverage | Width | rho(width,error) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for engine in ENGINES:
        for calibrated in (0, 1):
            vals = [r for r in rows if r["engine"] == engine and r["calibrated"] == calibrated]
            if not vals:
                continue
            md.append(
                f"| {engine} | {calibrated} | "
                f"{np.median([v['scale'] for v in vals]):.2f} | "
                f"{np.median([v['ctd1'] for v in vals]):.3f} | "
                f"{np.median([v['ctd2'] for v in vals]):.3f} | "
                f"{np.median([v['dev'] for v in vals]):.3f} | "
                f"{np.median([v['coverage'] for v in vals]):.3f} | "
                f"{np.median([v['width'] for v in vals]):.3f} | "
                f"{np.nanmedian([v['rho_width_error'] for v in vals]):+.3f} |"
            )
    md.append("")
    md.append("Discussion readout:")
    md.append("- If deep/bootstrap ensembles beat SGLD on coverage-width tradeoff, use them as strong proposal baselines.")
    md.append("- If scalar calibration fixes coverage with acceptable width inflation, pitch calibration as a formal layer.")
    md.append("- If all methods need large scale factors, the issue is not SGLD alone; the predictive loss/UQ target needs calibration.")
    md.append("")
    md.append(f"Raw seed-level results: `{csv.name}`.")
    (OUT_DIR / "uq_engine_comparator_report.md").write_text("\n".join(md) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    print("=== Study B/C: UQ engine comparator + scalar calibration ===")
    print(f"device={DEVICE}; seeds={SEEDS}; engines={ENGINES}")
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n--- seed {seed} ---", flush=True)
        cohorts, train_df, calib_df, time_grid, teacher, true_cal, true_test = setup(seed)
        for engine in ENGINES:
            print(f"  engine={engine}", flush=True)
            (cal_cif, _cal_prob), (test_cif, test_prob) = run_engine(
                engine,
                cohorts,
                train_df,
                calib_df,
                time_grid,
                teacher,
                seed,
            )
            cal_q = interval_quantiles(cal_cif)
            test_q = interval_quantiles(test_cif)
            scale = oracle_scale_from_calibration(cal_q, true_cal)
            rows.append(
                summarize(
                    engine,
                    seed,
                    test_q,
                    test_prob,
                    true_test,
                    cohorts.test,
                    time_grid,
                    calibrated=False,
                    scale=1.0,
                )
            )
            rows.append(
                summarize(
                    engine,
                    seed,
                    apply_scale(test_q, scale),
                    test_prob,
                    true_test,
                    cohorts.test,
                    time_grid,
                    calibrated=True,
                    scale=scale,
                )
            )
            print(
                f"    raw cov={rows[-2]['coverage']:.3f} wid={rows[-2]['width']:.3f}; "
                f"cal scale={scale:.2f} cov={rows[-1]['coverage']:.3f} wid={rows[-1]['width']:.3f}",
                flush=True,
            )
            write_outputs(rows)
        print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    write_outputs(rows)
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
