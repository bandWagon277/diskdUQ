# Manuscript restoration, 2026-09-17

## Scope

Writing and presentation revision only. No page-limit compression, new model fits,
posterior sampling, selection changes, or experiment exclusions were introduced.

## Historical sources and preserved corrections

- `b852a95:iclr/main.tex`: restored the clinical motivation, prediction-only transfer
  setting, borrowing/uncertainty distinction, and fuller Related Work organization.
- Compared the later `74c39de` and current manuscript before restoration. Kept the
  single-event formulation, MALA computation, current inference scopes, and corrected
  matched-baseline definitions rather than reverting to obsolete SGLD or competing-risk text.
- Restored the full Simulation I architecture and target-information tables and
  Simulation II teacher comparison from the existing appendix. Removed their abbreviated
  main-text duplicates. Historical baseline numbers superseded by matched-inference
  corrections were not restored.
- Added TinyNN's three matched rows from the existing `matched_capacity.csv` generated
  by `iclr/rebuild_main_figures.py` in the September 16 revision (100 cohorts).
- Added selected-student concordance to the Simulation II table from the existing
  `formal__table3_imperfect_teachers.csv` under `results_revision/empsd_tagfix`.
  Restricted the extraction to the seven fixed-teacher regimes already reported.
- Confirmed the direct oracle row summarizes 100 cohorts in the historical capacity
  aggregation; clarified that replication scope in the caption.

## Organization

- Main theory now states the functional response, posterior/sampling variance distinction,
  oracle variance ratio, and scope of inference. Full propositions, assumptions, and
  derivations remain in the appendix with working cross-references.
- Tables compare architectures, target-information settings, and selected teacher-guided
  fits. Figures show horizon-specific behavior and fixed borrowing paths, not a second
  visualization of the same selected-fit table.
- Removed the redundant METABRIC prediction-path plot, repeated selected-model outcome
  panel, and derived regret table. Retained the full numeric path, distinct selection
  scores, interval summaries, and numerical diagnostics.
- Analysis prose emphasizes interpretation and comparisons rather than repeating table
  entries. Dataset sizes, protocol settings, and numerical values in tables are retained.

## Independent review

A fresh reviewer without project history inspected the revised manuscript. Addressed:
TinyNN evidence visibility; unsupported isolation of posterior averaging in METABRIC;
the overbroad claim that the teacher table contained every score; oracle replication scope;
explicit mention of binned censoring-weighted evaluation; duplicated METABRIC evidence;
and nonmonotone pooled-reference inclusion. No new experiments were requested or run.

## Verification

LaTeX/BibTeX compilation and visual checks cover the expanded opening, concise theory,
restored main tables, experimental figures, real-data sections, and appendix statements.
The PDF is a local build artifact, not a source-controlled experimental result.
