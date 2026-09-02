"""Distillation-lift matrix: seed x student-N x teacher-quality x method (Adam-only).

Goal: map the regime where distillation actually helps over an Internal-CR
baseline. Adam-only (no SGLD) so the sweep can cover a wider matrix quickly.

  Seeds          : 5
  Student N      : {100, 200, 500}
  Teacher quality: {"low", "reduced", "full"}
  Methods        : Internal CR, CR->CR, Overall->CR, Binary-1->CR, Binary-2->CR
  Per cell       : 5 AdamW restarts (different init seeds)

Total fits = 5 seeds * 3 N * 3 quality * 5 methods * 5 restarts = 1125.
Each fit ~5s on GPU -> ~1.5 hr.

For each (seed, N, quality) cell we compute the lift of each distillation
method over the Internal-CR baseline of the SAME (seed, N, quality), using
median across the 5 restarts. This isolates the distillation effect from
init-noise and seed-noise.

Outputs (in $OUT_DIR):
  uq_lift_matrix_results.csv
  uq_lift_matrix_summary.md
  uq_lift_matrix_ctd1.{pdf,png}   # heatmap per method: lift in Ctd1 vs Internal CR
  uq_lift_matrix_ctd2.{pdf,png}
  uq_lift_matrix_dev.{pdf,png}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid


# ---------- Sweep configuration ----------
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
TEST_N = int(os.environ.get("TEST_N", 500))
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 32
BATCH_SIZE = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))
N_RESTARTS = int(os.environ.get("N_RESTARTS", 5))
HORIZON_INDEX = int(os.environ.get("HORIZON_INDEX", 7))

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
STUDENT_NS = [int(n) for n in os.environ.get("STUDENT_NS", "100,200,500").split(",")]
QUALITIES = [q.strip() for q in os.environ.get("QUALITIES", "low,reduced,full").split(",")]

METHODS = ["internal", "cr_to_cr", "overall_to_cr", "binary1_to_cr", "binary2_to_cr"]
METHOD_LABEL = {
    "internal":      "Internal CR",
    "cr_to_cr":      "CR -> CR",
    "overall_to_cr": "Overall -> CR",
    "binary1_to_cr": "Binary-1 -> CR",
    "binary2_to_cr": "Binary-2 -> CR",
}

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))


# ---------- Helpers ----------

def setup_cohorts(seed, student_n, quality):
    cohorts = simulate_competing_risk_cohorts(
        n_teacher=TEACHER_N,
        n_student=student_n,
        n_test=TEST_N,
        seed=seed,
        teacher_feature_quality=quality,
    )
    combined_dur = np.concatenate([cohorts.teacher["duration"].values,
                                    cohorts.student["duration"].values])
    time_grid = fit_time_grid(combined_dur, NUM_DURATIONS)
    return cohorts, time_grid


def train_cr_teacher(cohorts, time_grid):
    return DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(cohorts.teacher, feature_cols=cohorts.teacher_features)


def train_overall_teacher(cohorts, time_grid):
    td = cohorts.teacher.copy()
    td["event_any"] = (td["event"] > 0).astype(int)
    return DiscreteSurvivalModel(
        num_risks=1, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(td, feature_cols=cohorts.teacher_features, event_col="event_any")


def train_binary_teacher(cohorts, time_grid, risk_label):
    horizon_time = float(time_grid.cuts[HORIZON_INDEX])
    td = cohorts.teacher
    y = ((td["event"] == risk_label) & (td["duration"] <= horizon_time)).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    teacher_features = list(cohorts.teacher_features)
    clf.fit(td[teacher_features], y)

    def predict(frame):
        return clf.predict_proba(frame[teacher_features])[:, 1]

    return predict


def build_student(method, teachers, time_grid):
    common = dict(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
        epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid, optimizer="adamw",
    )
    if method == "internal":
        return DiscreteSurvivalModel(**common)
    if method == "cr_to_cr":
        return DiSKDStudent(teacher_model=teachers["cr"], teacher_type="competing",
                            eta=1.0, temperature=2.0, **common)
    if method == "overall_to_cr":
        return DiSKDStudent(teacher_model=teachers["overall"], teacher_type="overall",
                            eta=1.0, **common)
    if method == "binary1_to_cr":
        return DiSKDStudent(teacher_model=teachers["bin1"], teacher_type="binary_horizon",
                            binary_risk_index=0, binary_horizon_index=HORIZON_INDEX,
                            eta=1.0, **common)
    if method == "binary2_to_cr":
        return DiSKDStudent(teacher_model=teachers["bin2"], teacher_type="binary_horizon",
                            binary_risk_index=1, binary_horizon_index=HORIZON_INDEX,
                            eta=1.0, **common)
    raise ValueError(method)


def evaluate(model, test, test_dur, test_ev, test_idx, true_cif):
    cif = np.asarray(model.predict_cif(test))
    ctd = competing_risk_c_index(cif, test_dur, test_ev)
    dev = predictive_deviance(model.predict_interval_probs(test).numpy(), test_idx, test_ev)
    # Per-entry absolute error vs closed-form ground truth (per cell summary metric).
    abs_err = np.abs(cif - true_cif).mean()
    return float(ctd[0]), float(ctd[1]), float(dev), float(abs_err)


# ---------- Main sweep ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    print(f"=== Distillation lift matrix sweep ===")
    print(f"Seeds: {SEEDS}")
    print(f"Student N: {STUDENT_NS}")
    print(f"Teacher quality: {QUALITIES}")
    print(f"Methods: {METHODS}")
    print(f"Per-cell restarts: {N_RESTARTS}")

    rows = []
    for seed in SEEDS:
        for student_n in STUDENT_NS:
            for quality in QUALITIES:
                cell_t0 = time.time()
                cohorts, time_grid = setup_cohorts(seed, student_n, quality)
                test = cohorts.test
                test_dur = test["duration"].to_numpy()
                test_ev = test["event"].to_numpy()
                test_idx = transform_durations(test_dur, time_grid)
                true_cif = true_cif_at_grid(test, time_grid, num_risks=NUM_RISKS)

                # ---- Train teachers up front for this cell ----
                teachers = {}
                teachers["cr"] = train_cr_teacher(cohorts, time_grid)
                teachers["overall"] = train_overall_teacher(cohorts, time_grid)
                teachers["bin1"] = train_binary_teacher(cohorts, time_grid, 1)
                teachers["bin2"] = train_binary_teacher(cohorts, time_grid, 2)

                # ---- For each method, run N_RESTARTS independent fits ----
                for method in METHODS:
                    for r in range(N_RESTARTS):
                        torch.manual_seed(1000 * seed + r)
                        np.random.seed(1000 * seed + r)
                        student = build_student(method, teachers, time_grid)
                        student.fit(cohorts.student, feature_cols=cohorts.student_features)
                        c1, c2, dev, abs_err = evaluate(
                            student, test, test_dur, test_ev, test_idx, true_cif)
                        rows.append({
                            "seed": seed, "student_n": student_n, "quality": quality,
                            "method": method, "restart": r,
                            "ctd1": c1, "ctd2": c2, "dev": dev, "abs_err": abs_err,
                        })
                print(f"  seed={seed} N={student_n} quality={quality}: "
                      f"done in {time.time()-cell_t0:.0f}s "
                      f"({len(METHODS)*N_RESTARTS} fits)", flush=True)

    # ---- Aggregate to CSV ----
    csv_path = OUT_DIR / "uq_lift_matrix_results.csv"
    with csv_path.open("w") as f:
        f.write("seed,student_n,quality,method,restart,ctd1,ctd2,dev,abs_err\n")
        for r in rows:
            f.write(f"{r['seed']},{r['student_n']},{r['quality']},{r['method']},"
                    f"{r['restart']},{r['ctd1']:.4f},{r['ctd2']:.4f},"
                    f"{r['dev']:.4f},{r['abs_err']:.6f}\n")
    print(f"\nResults saved to {csv_path}", flush=True)

    # ---- Aggregate per cell (median across restarts), then lift vs Internal ----
    # Build a dict: (seed, N, quality, method) -> median(ctd1, ctd2, dev, abs_err)
    cell_median = {}
    for r in rows:
        key = (r["seed"], r["student_n"], r["quality"], r["method"])
        cell_median.setdefault(key, []).append(r)
    cell_summary = {}
    for key, items in cell_median.items():
        c1 = np.median([x["ctd1"] for x in items])
        c2 = np.median([x["ctd2"] for x in items])
        dv = np.median([x["dev"] for x in items])
        cell_summary[key] = {"ctd1": c1, "ctd2": c2, "dev": dv}

    # Lift = method - internal, evaluated per (seed, N, quality)
    lifts = []  # (seed, N, quality, method, dCtd1, dCtd2, dDev)
    for seed in SEEDS:
        for student_n in STUDENT_NS:
            for quality in QUALITIES:
                internal_key = (seed, student_n, quality, "internal")
                if internal_key not in cell_summary:
                    continue
                base = cell_summary[internal_key]
                for method in METHODS:
                    if method == "internal":
                        continue
                    key = (seed, student_n, quality, method)
                    if key not in cell_summary:
                        continue
                    m = cell_summary[key]
                    lifts.append({
                        "seed": seed, "student_n": student_n, "quality": quality,
                        "method": method,
                        "dCtd1": m["ctd1"] - base["ctd1"],
                        "dCtd2": m["ctd2"] - base["ctd2"],
                        "dDev":  m["dev"]  - base["dev"],
                    })

    # ---- Per (N, quality, method) median + IQR across seeds ----
    matrix = {}  # (N, quality, method) -> {dCtd1, dCtd2, dDev: median, iqr}
    for student_n in STUDENT_NS:
        for quality in QUALITIES:
            for method in METHODS:
                if method == "internal":
                    continue
                vals1 = [x["dCtd1"] for x in lifts
                         if x["student_n"] == student_n and x["quality"] == quality
                         and x["method"] == method]
                vals2 = [x["dCtd2"] for x in lifts
                         if x["student_n"] == student_n and x["quality"] == quality
                         and x["method"] == method]
                valsd = [x["dDev"]  for x in lifts
                         if x["student_n"] == student_n and x["quality"] == quality
                         and x["method"] == method]
                if not vals1:
                    continue
                matrix[(student_n, quality, method)] = {
                    "dCtd1_med": float(np.median(vals1)),
                    "dCtd1_q25": float(np.quantile(vals1, 0.25)),
                    "dCtd1_q75": float(np.quantile(vals1, 0.75)),
                    "dCtd2_med": float(np.median(vals2)),
                    "dCtd2_q25": float(np.quantile(vals2, 0.25)),
                    "dCtd2_q75": float(np.quantile(vals2, 0.75)),
                    "dDev_med":  float(np.median(valsd)),
                    "dDev_q25":  float(np.quantile(valsd, 0.25)),
                    "dDev_q75":  float(np.quantile(valsd, 0.75)),
                    "n_seeds":   len(vals1),
                }

    # ---- Heatmaps: one figure per metric (dCtd1, dCtd2, dDev), 4 methods x (N x quality) ----
    distill_methods = [m for m in METHODS if m != "internal"]
    for metric_key, metric_label, suffix in [
        ("dCtd1_med", "$\\Delta C_{\\rm td}$ cause 1 (vs Internal CR)", "ctd1"),
        ("dCtd2_med", "$\\Delta C_{\\rm td}$ cause 2 (vs Internal CR)", "ctd2"),
        ("dDev_med",  "$\\Delta$Deviance (vs Internal CR)",                 "dev"),
    ]:
        fig, axes = plt.subplots(1, len(distill_methods), figsize=(4 * len(distill_methods), 4))
        if len(distill_methods) == 1:
            axes = [axes]
        # For deviance, sign convention: more negative is better (lower dev).
        # For Ctd, more positive is better.
        higher_is_better = metric_key.startswith("dCtd")
        # Determine global vmax for shared color scale
        all_vals = [matrix[k][metric_key] for k in matrix if k[2] in distill_methods]
        vabs = max(abs(min(all_vals)), abs(max(all_vals)), 1e-3) if all_vals else 0.05
        for ax, method in zip(axes, distill_methods):
            mat = np.full((len(STUDENT_NS), len(QUALITIES)), np.nan)
            for i, n in enumerate(STUDENT_NS):
                for j, q in enumerate(QUALITIES):
                    if (n, q, method) in matrix:
                        mat[i, j] = matrix[(n, q, method)][metric_key]
            cmap = "RdBu_r" if higher_is_better else "RdBu"
            im = ax.imshow(mat, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
            for i in range(len(STUDENT_NS)):
                for j in range(len(QUALITIES)):
                    if not np.isnan(mat[i, j]):
                        color = "white" if abs(mat[i, j]) > 0.5 * vabs else "black"
                        ax.text(j, i, f"{mat[i, j]:+.3f}",
                                ha="center", va="center", fontsize=8, color=color)
            ax.set_xticks(range(len(QUALITIES)))
            ax.set_xticklabels(QUALITIES, fontsize=8)
            ax.set_yticks(range(len(STUDENT_NS)))
            ax.set_yticklabels([f"N={n}" for n in STUDENT_NS], fontsize=8)
            ax.set_xlabel("teacher feature quality")
            ax.set_title(METHOD_LABEL[method], fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"{metric_label} — median across {len(SEEDS)} seeds, "
                     f"{N_RESTARTS} Adam restarts/cell", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = OUT_DIR / f"uq_lift_matrix_{suffix}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Heatmap saved: {out}", flush=True)

    # ---- Markdown summary ----
    md = []
    md.append("# Distillation lift matrix — Internal-CR vs 4 distillation schemes")
    md.append("")
    md.append("**Audience:** Jian, Kevin.")
    md.append(f"**Sweep:** {len(SEEDS)} seeds × {len(STUDENT_NS)} student-N × "
              f"{len(QUALITIES)} teacher qualities × {len(METHODS)} methods × {N_RESTARTS} restarts. "
              f"AdamW only; SGLD calibration is the companion probe.")
    md.append("")
    md.append(f"- Seeds: `{SEEDS}`")
    md.append(f"- Student N values: `{STUDENT_NS}`")
    md.append(f"- Teacher feature qualities: `{QUALITIES}` "
              "(`low` = teacher sees x1..x4; `reduced` = x1..x8; `full` = x1..x12)")
    md.append(f"- Teacher cohort N = {TEACHER_N} (held fixed across cells)")
    md.append(f"- Test cohort N = {TEST_N} (independent of student)")
    md.append("")
    md.append("Lift = `method - Internal CR`, paired within the same (seed, N, quality). "
              "Reported numbers are median across seeds. Positive ΔCtd is good; negative ΔDeviance is good.")
    md.append("")
    md.append("## Heatmaps")
    md.append("")
    md.append("- `uq_lift_matrix_ctd1.png` — ΔCtd cause 1")
    md.append("- `uq_lift_matrix_ctd2.png` — ΔCtd cause 2")
    md.append("- `uq_lift_matrix_dev.png` — ΔDeviance (negative = distillation helps)")
    md.append("")
    md.append("## Cells where distillation helps (median ΔCtd1 > 0)")
    md.append("")
    md.append("| Method | N | Quality | ΔCtd1 (med) | ΔCtd1 [Q25, Q75] | ΔDev (med) |")
    md.append("|---|---|---|---|---|---|")
    rows_md = []
    for (n, q, method), v in matrix.items():
        rows_md.append((method, n, q, v))
    rows_md.sort(key=lambda x: -x[3]["dCtd1_med"])
    for method, n, q, v in rows_md:
        sign = "+" if v["dCtd1_med"] >= 0 else ""
        md.append(f"| {METHOD_LABEL[method]} | {n} | {q} | "
                  f"{sign}{v['dCtd1_med']:.3f} | "
                  f"[{v['dCtd1_q25']:+.3f}, {v['dCtd1_q75']:+.3f}] | "
                  f"{v['dDev_med']:+.3f} |")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `uq_lift_matrix_results.csv` — all per-restart raw rows")
    md.append("- `uq_lift_matrix_{ctd1,ctd2,dev}.{pdf,png}` — heatmaps")
    md.append("- `uq_lift_matrix_summary.md` — this file")

    out_md = OUT_DIR / "uq_lift_matrix_summary.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Report saved to {out_md}", flush=True)


if __name__ == "__main__":
    main()
