# Format-controlled transfer, vector averaging and framing-disjoint subsets

Registered 2026-11-08, before any quantity below was computed. Analyses Q1 to Q3 use the stored
framing activations; Q4 and Q5 use the stored per-wrapper cells of `results/post_submission/cells`
(computed in October for the matched-null audit) and only add new aggregations of them. Nothing
here changes a reported number of the paper; departures from this document are reported as such.

## Data and estimator

- Ladders and checkpoints exactly as `scripts/pod_ladders.py` `LADDERS` (Qwen3 and Qwen3.5 in both
  modes, OLMo-2, and Qwen2.5 under its own convention from `results/gstudy/*/pooled.npz`).
  Activation files are read from `$EAS_ACTIVATION_ROOT`, as in `scripts/matched_null_cells.py`.
- Framing rows: the 6 evaluation (`eval_*`) and 6 deployment (`deploy_*`) framings, 60 tasks each,
  the array the ladder's directions are built from (`framing_validated`; `framing` for Qwen2.5).
- Task splits: 20 random 30/30 splits of the 60 tasks, seed 0. Directions use train tasks only.
- Real direction for wrapper (e, d): per layer, unit(mean over train tasks under e minus mean over
  train tasks under d), the paper's estimator restricted to train tasks.
- Content-free null: the 15 evaluation-minus-evaluation and 15 deployment-minus-deployment
  contrasts, same estimator, same train tasks. They carry framing differences but no
  evaluation-versus-deployment content.
- Geometric floor: 30 isotropic random unit directions per layer (seed 1).

## Format-controlled scoring set

For a test wrapper (e', d') and a split, the items are the 30 test tasks under e' (label 1) and the
same 30 test tasks under d' (label 0). Every task appears once in each class, so the task text is
identical across classes and only the framing differs. A real direction (e, d) is scored only on
disjoint test wrappers, e' != e and d' != d (25 per direction); a null contrast only on test
wrappers that share no framing with it (24 per direction).

Statistics, per direction, test wrapper and split:
- fold: the paper's statistic, max over all layers of |AUROC - 0.5|;
- sign: the mean over layers at relative depth 0.25 to 0.75 of (AUROC - 0.5), unfolded, so a
  direction pointing the wrong way scores below zero. No layer is selected on test data.

A direction's controlled score is the median over its test wrappers and the 20 splits.

## Q1. Do evaluation-minus-deployment directions carry content on format-controlled items?

Per model, R_fold is the median of the 36 real controlled fold scores and R_sign the median of the
36 real controlled sign scores. A model passes if R_fold exceeds the 95th percentile of the 30
null fold scores and R_sign exceeds the 95th percentile of the 30 null sign scores. Q1 holds on a
ladder if at least 75 percent of its models pass (3 of 4, 4 of 5, 5 of 6). The isotropic floor is
reported beside the null.

## Q2. Does the wrapper still decide the sign of the scaling trend on the controlled set?

Per ladder, for each of the 36 real wrappers, r_j = Pearson correlation of the controlled score
with log10 parameters, for fold and for sign. Reported: range, counts of each sign, evaluation-
and deployment-fixed spanning counts as in `pod_ladders.py`, next to the same quantities on the
paper's scoring set and for the null contrasts. Descriptive; it is not interpreted on a ladder
where Q1 fails. The claim "the wrapper decides the sign" is said to hold on the controlled set only
if Q1 holds and at least 6 of the 36 wrappers fall on each side of zero for fold.

## Q3. Variance components with the test framing as a facet

x[model, train wrapper, test wrapper] = fold score averaged over the 20 splits, all 36 x 36 cells.
Components, shares, E rho^2 for one wrapper (k = 1, n = 1) and the k for 0.80 with n = 36 test
wrappers, from `gstudy_gtheory.components_3way` and `coefficients`. Descriptive, reported next to
the paper-scoring-set values of `pod_gtheory.json`. Cells where the test wrapper shares a framing
with the train wrapper are included because the facet must be fully crossed; a sensitivity version
replacing them by the mean of the disjoint cells of the same (model, train wrapper) is reported.

## Q4. Averaging direction vectors versus averaging the scalar statistic (paper's scoring set)

On the paper's scoring set and statistic, for k in {1, 2, 3, 4, 6, 9, 12}: 50 random partitions of
the 36 wrappers into 36/k groups (k = 1: one partition). A group's score is the statistic of the
unit-normalised mean of its k unit directions (full-task directions, as in the paper). E rho^2 for
one group (k = 1, n = 1 in the crossed design model x group x the 4 seed-0 item blocks), median over
partitions, against the D-study E rho^2 of averaging k scalar statistics, `coefficients(c, k, 1)`
from the paper's components. Vector averaging "rises faster" on a ladder if its E rho^2 exceeds the
scalar value at every k >= 2.

## Q5. Wrapper subsets that share no framing

All 200 partitions of the 6 evaluation framings into 3 + 3 and the 6 deployment framings into 3 + 3
give subsets A (3 x 3 wrappers) and B (the complementary 3 x 3) with no framing in common. Per
partition: Spearman correlation across the ladder's models between the A-average and the B-average
of the per-wrapper statistic (`real_full`), and whether r with log parameters has the same sign in A
and B. Reported as distributions over partitions, and the same for each of the 100 `taskperm` null
draws. Real agreement "exceeds the null" if its median exceeds the 95th percentile of the null
draws' medians.

## Also reported, no new computation

Cosine between each null family's directions and the real directions, from
`results/post_submission/matched_null.json`.
