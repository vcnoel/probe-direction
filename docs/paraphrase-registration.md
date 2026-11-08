# Paraphrased framings: does wrapper sensitivity persist when the wording changes?

Registered 2026-11-08, before any activation of a paraphrased framing was extracted. The question:
does the sensitivity of the probe statistic to the wrapper persist when each wrapper states the
same scenario in different words? Nothing here changes a reported number of the paper; departures
from this document are reported as such.

## Framings

`data/framings_paraphrase.json` holds two paraphrases (`_p1`, `_p2`) of each of the 6 evaluation
and 6 deployment framings of `data/framings_multi.json`. Only the `system` text is reworded;
`family` and `user_template` are copied verbatim. Rules, checked by `scripts/check_paraphrases.py`
(which exits non-zero if any is broken): every scenario fact is kept (setting, audience, whether the
exchange is recorded, graded or monitored, stakes, the instruction to the model) and no fact is
added; the whitespace word count is within 10 percent of the original's; no 4-word sequence is
shared with the original or with the other paraphrase unless all four words are function words.

Version v of a framing f is f_0 (the original), f_1 or f_2. Version v of wrapper (e, d) is
(e_v, d_v). There are 36 wrappers per version, 108 per model.

## Models, prompts and activations

- Now, on a 16 GB laptop GPU: Qwen3-0.6B, 1.7B and 4B. Later, on one A100 80 GB: Qwen3-8B, 14B
  and 32B. Weights at the revisions recorded in the paper's extraction
  (`results/pod_qwen3/<model>__nothink.npz`, field `meta.revision`), bfloat16, never quantised.
- Rendering: the paper's non-thinking Qwen3 rendering, `enable_thinking=False`, date pinned to
  01 Jan 2025, through `eval_awareness_spectral.prompts.build_prompts` exactly as
  `scripts/render_prompts.py` renders the Qwen3 framing rows (LF newlines, no truncation, no
  position matching). The chat-template hash must equal the paper's (`a55ee1b16601`).
- Prompts: each of the 36 framings (12 originals, 24 paraphrases) x the same 60 tasks of
  `data/tasks_bench.jsonl`, 2160 prompts per model. Every rendered prompt must contain its task
  text verbatim.
- Pooling: `pod/extract.py` `pool_reductions` (the paper's Qwen3 pooling code, one forward pass,
  sdpa attention, right padding, batch 16, float16 storage). Stored: the validated array (BOS and
  whitespace-only tokens dropped) and the all-token array. Activation files are not committed.
- Scoring set: the paper's SAD stages-oversight set for each model, `sad_validated` and
  `sad_label` of `results/pod_qwen3/<model>__nothink.npz` (400 items), not re-extracted.
- Reproduction gate, before any paraphrase number is read: the 12 original framings' validated
  rows of the new extraction are compared with the stored `framing_validated` rows (max absolute
  difference reported), and the paper's statistic of the 36 version-0 wrappers is computed from
  both. If any wrapper's statistic differs by more than 0.01, the analysis stops and the
  discrepancy is reported instead.

## Directions and statistics

- Directions: full-task, the paper's estimator. Per layer, unit(mean over the 60 tasks under e_v
  minus mean over the 60 tasks under d_v), from the validated array.
- fold: the paper's statistic, max over all hidden states (embedding output included, as in
  `scripts/pod_ladders.py`) of |max(AUROC, 1 - AUROC) - 0.5| on the SAD scoring set.
- sign: the mean over hidden states at relative depth 0.25 to 0.75 (index / (number of states - 1),
  as in `scripts/controlled_transfer.py`) of (AUROC - 0.5), unfolded.
- Verdicts are stated on fold, the paper's statistic. sign is computed with the same code and
  reported beside it as description.

## P7a. Reproduction under paraphrase

Per model: Spearman correlation across the 36 wrappers between version 0 and version v (v = 1, 2),
and between versions 1 and 2. Interval: bootstrap over evaluation and deployment framings, 2000
draws, seed 0; each draw resamples the 6 evaluation and the 6 deployment framings with
replacement, takes the 36 wrappers they cross (duplicates kept) and recomputes the correlation;
95 percent percentile interval. Draws with a constant vector are dropped and counted.
P7a holds on a model if both version-0 correlations (0 vs 1, 0 vs 2) have a 95 percent interval
above zero. It holds on the ladder if it holds on every model run.

## P7b. Variance components with version as a facet

x[model, wrapper, version] = fold, 3 x 36 x 3 now (6 x 36 x 3 after the pod),
`gstudy_gtheory.components_3way` with model, wrapper and version in that order. Reported: the
shares of model, wrapper, version, model x wrapper, model x version, wrapper x version and the
three-way term, and which components were truncated at zero. Descriptive. The sentence "the
model-by-wrapper term is carried by the scenario, not the wording" is used only if the
model x wrapper share exceeds the three-way share. The same decomposition of sign is reported.

## P7c. Scaling sign under paraphrase (pod step)

Run only once at least 4 rungs exist. Per version, r_j = Pearson correlation of wrapper j's fold
with log10 parameters (parameter counts of `scripts/pod_ladders.py`, ladder `qwen3_nothink`).
Statistic: the fraction of the 36 wrappers whose r has the same sign in version 0 and version v
(v = 1, 2). Reference: the same fraction under 1000 random re-pairings of wrappers (version-v
values assigned to a random permutation of the 36 wrappers, seed 0). Reported: real fraction,
re-pairing mean and 95th percentile, and the real value's percentile. The sign "reproduces beyond
re-pairing" for version v if the real fraction exceeds the re-pairing 95th percentile.

## P7d. Orientation on byte-identical tasks across wording

Amendment 1 (A1) of `docs/controlled-transfer-registration.md`, with training and test framings
taken from different versions. Splits: the registered 20 random 30/30 splits of the 60 tasks,
seed 0. Layers: relative depth 0.25 to 0.75. Per split and layer, train means m of the 12 original
framings (version 0, train tasks) and test means t of the 12 framings in version v (v = 1, 2, test
tasks). For a direction unit(m_u - m_w) and a test pair (u', w'), orientation is correct if
t_u' . dir > t_w' . dir.

- Real: the 36 directions (e_0, d_0) on test pairs (e'_v, d'_v) with e' != e and d' != d.
- Baseline: the 200 crossing partitions of the 12 framings into two groups of six (3 evaluation
  and 3 deployment each, `controlled_transfer.crossing_partitions`); a partition's directions are
  (a_0 - b_0) for a in group 1, b in group 2, scored on test pairs (a'_v, b'_v) with a' != a and
  b' != b, orientation taken with respect to the groups.
- R_o = mean correct orientation over real directions, test pairs, layers and splits; the baseline
  is the 200 per-partition means. A model passes for version v if R_o exceeds the baseline's 95th
  percentile. P7d holds on a model if it passes for both v = 1 and v = 2, and on the ladder if at
  least 75 percent of its models pass (3 of 3 now; 5 of 6 after the pod).
- Reported beside it, as a reference: the same statistic with test framings in version 0 (A1 on
  the new extraction).

## Outputs

`scripts/extract_paraphrase.py` extracts; `scripts/paraphrase_analysis.py` computes P7a to P7d and
writes `results/post_submission/paraphrase/paraphrase_analysis.json` through `result_io.write_result`.
The pod steps (Qwen3-8B, 14B, 32B) run the same extraction script through
`pod/run_post_submission.sh`.

## Companion analysis: the Qwen2.5 ladder under the validated convention

Not a paraphrase analysis; registered here because it is computed in the same session. The Qwen2.5
row of Table 1 builds directions from all-token pooled framing activations while every other
ladder drops BOS and whitespace-only tokens. The framing prompts of the Qwen2.5 rungs are
re-extracted with `scripts/extract_model.py`'s `pool_both` (both conventions in one pass, float32,
same prompts, same model loading as the paper's extraction). The scoring set is unchanged
(`results/gstudy/<model>/sad_filtered_lf.npz`, already BOS- and whitespace-dropped), so the only
change is the pooling of the direction arrays. Gate: the re-extracted all-token array must
reproduce the stored `results/gstudy/<model>/pooled.npz` framing array (max absolute difference and
the Table 1 quantities from both are reported). Then the Table 1 quantities of
`scripts/pod_ladders.py` (correlation range and signs, spanning counts, the seven variance shares,
E rho^2 for one wrapper on the full item set, k for 0.80) are recomputed with validated directions,
next to the submitted values and to all-token values on the same rungs. Rungs now: 0.5B, 1.5B, 3B;
the 7B rung is a pod step. Output: `results/post_submission/qwen25_validated/table1.json`.
