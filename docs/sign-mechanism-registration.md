# What sets a wrapper's scaling sign on the standard scoring set

Registered 2026-11-08, before any quantity below was computed. Same ladders, conventions, activation
files and full-task directions as `scripts/pod_ladders.py` (the paper's estimator on all 60 tasks).

## Quantities

- Benchmark class axis, per model and layer: u = unit(mean pooled scoring-set activation of the items
  labelled evaluation minus that of the items labelled deployment), on the paper's scoring set and
  pooling.
- Read layer of a direction: the layer at which the paper's folded statistic, max over layers of
  |AUROC - 0.5|, attains its maximum for that (model, direction).
- Alignment of a direction: |cos(direction, u)| at its read layer (the folded statistic ignores sign).
- Content axis, per model and layer: unit(mean of the 36 unit evaluation-minus-deployment directions),
  the axis whose cross-wording orientation `docs/controlled-transfer-registration.md` (Amendment 1)
  measured on byte-identical tasks.
- Content-free comparison: the 30 same-class contrasts (evaluation minus evaluation, deployment minus
  deployment), full-task, same estimator.

## M1a. Is a direction's score set by its alignment with the class axis?

Per model, Spearman correlation across the 36 wrappers between the paper's statistic and the alignment.
Reported per model with the same quantity for the 30 content-free contrasts. "The score is set by
alignment" on a ladder if the median over its models is at least 0.5 for the real directions.

## M1b. Does a wrapper's scaling sign follow how its alignment changes with size?

Per ladder, for each wrapper, r_score = Pearson r of the statistic with log10 parameters and
r_align = Pearson r of the alignment with log10 parameters; Spearman correlation of r_score and r_align
across the 36 wrappers, and the share of wrappers whose two r have the same sign. The same for the 30
content-free contrasts. "The alignment trend sets the sign" on a ladder if the Spearman is at least 0.5
for the real directions. Whether it also holds for the content-free contrasts is reported, not judged.

## M1c. Does the benchmark's class axis point along the evaluation content?

Per model: cos(u, content axis) at every layer; reported at the median read layer of the 36 wrappers and
as the maximum of |cos| over layers, next to the value expected for an unrelated direction,
1/sqrt(width). Descriptive. It answers whether scores on the scoring set measure the evaluation axis the
framings define, or another axis.
