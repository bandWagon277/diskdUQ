"""Last-layer sandwich/Godambe correction for generalized Bayesian DiSKD.

This is a low-dimensional Bayesian correction probe. It freezes a DiSKD
representation learned by AdamW, treats only the final prediction head as the
parameter vector, and compares:

  - raw generalized Laplace covariance: Sigma_2 = H^{-1}
  - information correction: Sigma_1 = Sigma_2 V^{-1} Sigma_2
  - optional score-meat correction: H^{-1} J H^{-1}

Here H is the curvature of NLL_internal + eta * KL_teacher, and V^{-1} is the
internal-only likelihood information. This mirrors external-KL surrogate
corrections where the KL term adds curvature but is not a real external raw-data
likelihood with matching sampling variability.

The correction is intentionally last-layer only. Full-network covariance is
not identifiable/stable enough for this first diagnostic.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (  # noqa: E402
    DiSKDStudent,
    DiscreteSurvivalModel,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid  # noqa: E402
from diskd.losses import CompetingRiskKDLoss, CompetingRiskNLLLoss  # noqa: E402
from diskd.networks import sinusoidal_time_embedding  # noqa: E402


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
HIDDEN_DIMS = [int(x) for x in os.environ.get("HIDDEN_DIMS", "32").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]

ETA = float(os.environ.get("ETA", 1.0))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 2.0))
OMEGA = float(os.environ.get("OMEGA", 1.0))
RIDGE = float(os.environ.get("RIDGE", 1e-4))
N_POSTERIOR_SAMPLES = int(os.environ.get("N_POSTERIOR_SAMPLES", 1000))
PREDICT_CHUNK = int(os.environ.get("PREDICT_CHUNK", 256))
MEAT_MODE = os.environ.get("MEAT_MODE", "information")  # information | score | both

OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        str(Path(__file__).resolve().parent.parent / "responses_sandwich"),
    )
)


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
        backbone="time_mlp",
        hidden_dim=hidden_dim,
        epochs=ADAMW_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        time_grid=time_grid,
        optimizer="adamw",
        teacher_model=teacher,
        teacher_type="competing",
        eta=ETA,
        temperature=TEMPERATURE,
    ).fit(cohorts.student, feature_cols=cohorts.student_features)
    if model.net is None:
        raise RuntimeError("MAP model was not fitted.")
    return model


@torch.no_grad()
def frozen_time_mlp_features(model: DiSKDStudent, data) -> torch.Tensor:
    """Return frozen hidden features with shape [N, K, H]."""
    if model.net is None:
        raise RuntimeError("Model is not fitted.")
    net = model.net
    required = ("feature_projection", "blocks", "head")
    if not all(hasattr(net, name) for name in required):
        raise RuntimeError("This probe currently supports the time_mlp backbone only.")
    x = torch.tensor(model._transform_features(data), dtype=torch.float64, device=model.device)
    net = net.to(dtype=torch.float64)
    net.eval()
    h = net.feature_projection(x).unsqueeze(1)
    emb = sinusoidal_time_embedding(
        model.num_durations,
        h.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )
    h = h + emb.unsqueeze(0)
    h = net.blocks(h)
    return h.detach()


def head_to_beta(model: DiSKDStudent) -> torch.Tensor:
    if model.net is None or not hasattr(model.net, "head"):
        raise RuntimeError("Model must expose a final head.")
    head = model.net.head
    beta = torch.cat(
        [
            head.weight.detach().to(dtype=torch.float64, device=model.device).reshape(-1),
            head.bias.detach().to(dtype=torch.float64, device=model.device),
        ]
    )
    return beta


def split_beta(beta: torch.Tensor, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_weight = NUM_RISKS * hidden_dim
    weight = beta[:n_weight].reshape(NUM_RISKS, hidden_dim)
    bias = beta[n_weight : n_weight + NUM_RISKS]
    return weight, bias


def logits_from_beta(beta: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Map last-layer parameters and frozen features to logits [N, J, K]."""
    weight, bias = split_beta(beta, features.shape[-1])
    logits_nkj = torch.einsum("nkh,jh->nkj", features, weight) + bias.view(1, 1, NUM_RISKS)
    return logits_nkj.permute(0, 2, 1)


def stable_inverse_psd(matrix: torch.Tensor, ridge: float) -> tuple[torch.Tensor, float, float]:
    matrix = 0.5 * (matrix + matrix.T)
    if ridge > 0:
        matrix = matrix + ridge * torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    max_eval = torch.clamp(eigvals.max(), min=1.0)
    floor = max(float(ridge), float(max_eval.detach().cpu()) * 1e-8)
    eig_clamped = torch.clamp(eigvals, min=floor)
    inv = (eigvecs * (1.0 / eig_clamped).unsqueeze(0)) @ eigvecs.T
    return 0.5 * (inv + inv.T), float(eigvals.min().detach().cpu()), float(eigvals.max().detach().cpu())


def project_psd(matrix: torch.Tensor, ridge: float = 0.0) -> tuple[torch.Tensor, float, float]:
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    max_eval = torch.clamp(eigvals.max(), min=1.0)
    floor = max(float(ridge), float(max_eval.detach().cpu()) * 1e-10)
    eig_clamped = torch.clamp(eigvals, min=floor)
    psd = (eigvecs * eig_clamped.unsqueeze(0)) @ eigvecs.T
    return 0.5 * (psd + psd.T), float(eigvals.min().detach().cpu()), float(eigvals.max().detach().cpu())


def compute_curvatures(model, teacher, cohorts, time_grid, beta_map):
    features = frozen_time_mlp_features(model, cohorts.student)
    idx = torch.tensor(
        transform_durations(cohorts.student["duration"].values, time_grid),
        dtype=torch.long,
        device=model.device,
    )
    events = torch.tensor(cohorts.student["event"].to_numpy(), dtype=torch.long, device=model.device)
    teacher_probs = teacher.predict_interval_probs(cohorts.student)[:, :NUM_RISKS, :].to(
        dtype=torch.float64,
        device=model.device,
    )
    nll = CompetingRiskNLLLoss()
    kd = CompetingRiskKDLoss(eta=ETA, temperature=TEMPERATURE)

    def internal_sum(beta: torch.Tensor) -> torch.Tensor:
        logits = logits_from_beta(beta, features)
        return nll(logits, idx, events, reduction="sum")

    def generalized_sum(beta: torch.Tensor) -> torch.Tensor:
        logits = logits_from_beta(beta, features)
        # CompetingRiskKDLoss returns (NLL + eta * KL) / (1 + eta).
        return (1.0 + ETA) * kd(logits, idx, events, teacher_probs, reduction="sum")

    beta_req = beta_map.detach().clone().requires_grad_(True)
    h_gen = torch.autograd.functional.hessian(generalized_sum, beta_req, vectorize=True)
    h_int = torch.autograd.functional.hessian(internal_sum, beta_req, vectorize=True)
    h_gen = 0.5 * (h_gen.detach() + h_gen.detach().T)
    h_int = 0.5 * (h_int.detach() + h_int.detach().T)

    score_meat = None
    if MEAT_MODE in {"score", "both"}:
        logits = logits_from_beta(beta_req, features)
        nll_i = nll(logits, idx, events, reduction="none")
        grads = []
        for i in range(nll_i.shape[0]):
            grad_i = torch.autograd.grad(nll_i[i], beta_req, retain_graph=True)[0].detach()
            grads.append(grad_i)
        g = torch.stack(grads, dim=0)
        score_meat = g.T @ g
        score_meat = 0.5 * (score_meat + score_meat.T)

    return h_gen, h_int, score_meat


def sample_gaussian(mean: torch.Tensor, cov: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
    cov, _, _ = project_psd(cov, ridge=0.0)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eig_floor = float(torch.clamp(eigvals.max(), min=1.0).detach().cpu()) * 1e-10
    eigvals = torch.clamp(eigvals, min=eig_floor)
    root = eigvecs * torch.sqrt(eigvals).unsqueeze(0)
    gen = torch.Generator(device=mean.device)
    gen.manual_seed(seed)
    z = torch.randn(n_samples, mean.numel(), dtype=mean.dtype, device=mean.device, generator=gen)
    return mean.unsqueeze(0) + z @ root.T


@torch.no_grad()
def predict_cif_and_probs_from_betas(beta_samples: torch.Tensor, features: torch.Tensor):
    cifs = []
    probs = []
    for start in range(0, beta_samples.shape[0], PREDICT_CHUNK):
        beta_chunk = beta_samples[start : start + PREDICT_CHUNK]
        weight = beta_chunk[:, : NUM_RISKS * features.shape[-1]].reshape(
            beta_chunk.shape[0],
            NUM_RISKS,
            features.shape[-1],
        )
        bias = beta_chunk[:, NUM_RISKS * features.shape[-1] :].reshape(beta_chunk.shape[0], NUM_RISKS)
        logits = torch.einsum("nkh,sjh->snjk", features, weight) + bias[:, None, :, None]
        no_event = torch.zeros(
            logits.shape[0],
            logits.shape[1],
            1,
            logits.shape[3],
            dtype=logits.dtype,
            device=logits.device,
        )
        full_logits = torch.cat([logits, no_event], dim=2)
        prob = torch.softmax(full_logits, dim=2)
        event_prob = prob[:, :, :NUM_RISKS, :]
        no_event_prob = prob[:, :, NUM_RISKS, :]
        survival_before = torch.ones_like(no_event_prob)
        if no_event_prob.shape[-1] > 1:
            survival_before[:, :, 1:] = torch.cumprod(no_event_prob[:, :, :-1], dim=-1)
        cif = torch.cumsum(survival_before[:, :, None, :] * event_prob, dim=-1)
        cifs.append(cif.detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
    return np.concatenate(cifs, axis=0), np.concatenate(probs, axis=0)


def evaluate_covariance(kind, cov, beta_map, model, cohorts, time_grid, true_cif, seed):
    features_test = frozen_time_mlp_features(model, cohorts.test)
    beta_samples = sample_gaussian(beta_map, cov, N_POSTERIOR_SAMPLES, seed=seed)
    cifs, probs = predict_cif_and_probs_from_betas(beta_samples, features_test)
    q = np.quantile(cifs, [0.025, 0.5, 0.975], axis=0)
    coverage = coverage_from_quantiles(q, true_cif)
    test_dur = cohorts.test["duration"].to_numpy()
    test_events = cohorts.test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    ctd = competing_risk_c_index(q[1], test_dur, test_events)
    dev = predictive_deviance(probs.mean(axis=0), test_idx, test_events)
    return {
        "kind": kind,
        "ctd1": float(ctd[0]),
        "ctd2": float(ctd[1]),
        "dev": float(dev),
        "coverage": float(coverage["overall"]),
        "width": float(coverage["mean_interval_width"]),
    }


def write_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "hidden_dim",
        "params",
        "kind",
        "ctd1",
        "ctd2",
        "dev",
        "coverage",
        "width",
        "h_gen_min_eig",
        "h_gen_max_eig",
        "h_int_min_eig",
        "h_int_max_eig",
        "cov_min_eig",
        "cov_max_eig",
    ]
    csv = OUT_DIR / "last_layer_sandwich_correction.csv"
    with csv.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")

    md = []
    md.append("# Last-Layer Sandwich/Godambe Correction Probe")
    md.append("")
    md.append(f"Cell: teacher N={TEACHER_N}, student N={STUDENT_N}, test N={TEST_N}, "
              f"teacher quality={TEACHER_FEATURE_QUALITY}.")
    md.append(f"Correction: raw Sigma2=H^-1; corrected_info=Sigma2 V^-1 Sigma2; "
              f"omega={OMEGA}, ridge={RIDGE}.")
    md.append(f"Posterior samples per covariance: {N_POSTERIOR_SAMPLES}.")
    md.append("")
    md.append("| Hidden | Kind | Ctd1 | Ctd2 | Dev | Coverage | Width |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    grouped = {}
    for row in rows:
        key = (row["hidden_dim"], row["kind"])
        grouped.setdefault(key, []).append(row)
    for (hidden, kind), vals in sorted(grouped.items()):
        md.append(
            f"| {hidden} | {kind} | "
            f"{np.median([v['ctd1'] for v in vals]):.3f} | "
            f"{np.median([v['ctd2'] for v in vals]):.3f} | "
            f"{np.median([v['dev'] for v in vals]):.3f} | "
            f"{np.median([v['coverage'] for v in vals]):.3f} | "
            f"{np.median([v['width'] for v in vals]):.3f} |"
        )
    md.append("")
    md.append("Interpretation:")
    md.append("- If corrected_info materially improves coverage/width versus raw, the external-KL covariance correction is useful.")
    md.append("- If corrected_info narrows already-undercover intervals, we need omega or function-space calibration in addition to covariance correction.")
    md.append("- Non-Bayesian UQ engines should remain baselines; this probe is the Bayesian correction route.")
    md.append("")
    md.append(f"Raw seed-level results: `{csv.name}`.")
    (OUT_DIR / "last_layer_sandwich_correction_report.md").write_text("\n".join(md) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    print("=== Last-layer sandwich/Godambe correction probe ===")
    print(f"device={DEVICE}; seeds={SEEDS}; hidden_dims={HIDDEN_DIMS}; meat_mode={MEAT_MODE}")
    rows = []
    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n--- seed {seed} ---", flush=True)
        cohorts, time_grid, teacher, true_cif = setup(seed)
        for hidden_dim in HIDDEN_DIMS:
            print(f"hidden_dim={hidden_dim}", flush=True)
            model = train_map(cohorts, time_grid, teacher, hidden_dim, seed)
            beta_map = head_to_beta(model)
            params = int(beta_map.numel())
            h_gen, h_int, score_meat = compute_curvatures(model, teacher, cohorts, time_grid, beta_map)
            raw_cov, hgen_min, hgen_max = stable_inverse_psd(OMEGA * h_gen, RIDGE)
            hint_min = float(torch.linalg.eigvalsh(0.5 * (h_int + h_int.T)).min().detach().cpu())
            hint_max = float(torch.linalg.eigvalsh(0.5 * (h_int + h_int.T)).max().detach().cpu())

            covariances = {"raw": raw_cov}
            covariances["corrected_info"] = raw_cov @ h_int @ raw_cov
            if score_meat is not None:
                covariances["corrected_score"] = raw_cov @ score_meat @ raw_cov

            for kind, cov in covariances.items():
                cov, cov_min, cov_max = project_psd(cov, ridge=0.0)
                result = evaluate_covariance(
                    kind,
                    cov,
                    beta_map,
                    model,
                    cohorts,
                    time_grid,
                    true_cif,
                    seed=90_000 * seed + hidden_dim + len(rows),
                )
                row = {
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "params": params,
                    **result,
                    "h_gen_min_eig": hgen_min,
                    "h_gen_max_eig": hgen_max,
                    "h_int_min_eig": hint_min,
                    "h_int_max_eig": hint_max,
                    "cov_min_eig": cov_min,
                    "cov_max_eig": cov_max,
                }
                rows.append(row)
                print(
                    f"  {kind:16s} cov={row['coverage']:.3f} width={row['width']:.3f} "
                    f"ctd1={row['ctd1']:.3f} dev={row['dev']:.3f}",
                    flush=True,
                )
                write_outputs(rows)
        print(f"seed {seed} done in {(time.time() - t_seed) / 60:.1f} min", flush=True)
    write_outputs(rows)
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
