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

## Amendment 1 (2026-11-08, after the OLMo-2 summary, before any other ladder's summary was read)

**Observation.** On OLMo-2 every direction separates the two held-out framings perfectly at some
layer: the fold statistic is 0.500 for real directions, null contrasts and isotropic directions
alike, and real directions reach AUROC 1.0 across the whole band. The two classes differ in their
whole framing text, so any direction separates them. The registered fold statistic and the Q1 rule
therefore tie at the ceiling and cannot discriminate on this scoring set; Q1 to Q3 are reported as
registered and marked uninformative. Q4 and Q5 are unaffected.

**What remains informative is orientation**: whether a direction built from one evaluation and one
deployment wording points from an unseen deployment framing to an unseen evaluation framing.

**A1. Orientation transfer.** Per split and layer in the registered band, train means m and test
means t of the 12 framings (30 tasks each). For a direction u - v (unit(m_u - m_v)) and a test pair
(u', v'), orientation is correct if t_u' . dir > t_v' . dir. Real: u, v = (e, d) with disjoint test
pairs (e', d'). Baseline: all 200 crossing partitions of the 12 framings into two groups of six, each
holding 3 evaluation and 3 deployment framings; a partition's directions are (a - b) for a in group 1,
b in group 2, scored on test pairs (a', b') with a' != a, b' != b, orientation taken with respect to
the groups. Per model, R_o = mean correct orientation over real directions, test pairs, band layers
and splits; the baseline is the 200 per-partition means. A model passes if R_o exceeds the
baseline's 95th percentile; A1 holds on a ladder if at least 75 percent of its models pass.

**A2. Alignment, for the scaling question and the variance components.** For a direction (u, v) and
a test pair (u', v'), alignment = cos(unit(m_u - m_v), unit(t_u' - t_v')), averaged over band
layers and splits. Real and crossing-partition baseline as in A1. Because the cosine of unrelated
directions shrinks with width, the per-model score used across models is the real median minus the
baseline median (excess alignment). Q2 is recomputed with excess alignment per wrapper (median over
its disjoint test pairs) in place of the fold score, under the same rule (A1 must hold, at least 6
wrappers of each sign). Q3 is recomputed with x[model, train wrapper, test wrapper] = alignment
averaged over splits and band layers, all 36 x 36 cells, with the same imputation sensitivity.

## Amendment 2 (2026-11-08, after the Amendment 1 summary of all ladders, before computing these)

The amended Q2 counts both signs on every ladder, but the registered rule compares real directions
with nothing. Two analyses are added, both on the alignment statistic of Amendment 1.

**B1. Null for the sign split.** For each of the 200 crossing partitions, its 36 directions (a in
group 1, b in group 2) get a per-direction excess alignment (median over test pairs sharing no
framing with the direction, minus the model's baseline median) and a Pearson r with log10
parameters. Reported: the real minority-sign count and the real SD of r, each placed as a mid-rank
percentile in the 200 partition values.

**B2. Split-half reliability of the per-wrapper sign on controlled items.** For each of the 200
partitions of the evaluation framings into 3 + 3 and the deployment framings into 3 + 3, test set A
holds the test pairs with both framings in the first halves, test set B the pairs with both in the
second halves. For each real direction, r_A and r_B use only the test pairs of that set sharing no
framing with the direction (directions with fewer than 2 such pairs in a set are dropped for that
partition). Reliability = Pearson correlation of r_A and r_B across directions, Spearman-Brown
corrected, median over partitions. The same computation on each crossing partition's 36 directions
(test sets restricted to that partition's groups in the same way) gives the null distribution of
the median. The per-wrapper sign is "reliable beyond the null" if the real value exceeds the null's
95th percentile.
