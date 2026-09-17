# Bayesian DiSKD manuscript

LaTeX source for **Prediction and Uncertainty in Bayesian Survival Distillation**.

This snapshot corresponds to manuscript commit `1e78a4a` (2026-09-17) in the
separate manuscript repository. It includes the restored Introduction and Related
Work, concise main-text theoretical conclusions with appendix derivations, complete
simulation comparison tables, and revised real-data presentation.

## Build

Use a TeX distribution with pdfLaTeX, BibTeX, and the standard AMS, graphics,
booktabs, longtable, array, float, and hyperref packages. The conference style,
bibliography style, and required figure PDFs are included.

From the repository root:

```sh
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error iclr/main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error iclr/main.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr/main.tex
```

The output is `manuscript/main.pdf`. Run pdfLaTeX once more if it requests another
pass to settle cross-references. The source snapshot was compiled and visually
checked before publication.

## Contents

- [Main source](iclr/main.tex)
- [References](iclr/references.bib)
- [Revision notes](planning/manuscript_restoration_20260917.md)
- Required aggregate simulation figures under `iclr/results/figures/`

The existing `pending_validation_placeholders.tex` is included as a source
dependency; it contains comments only and adds no printed results.

This directory contains no patient-level records, posterior draws, raw experiment
outputs, or cluster logs. It is a manuscript snapshot, not a new experiment run.
