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

## Amendment 1 (2026-11-08, before any Qwen3.5 prompt was extracted and before any number of this analysis was computed)

The paraphrase test must not rest on one family, so the Qwen3.5 ladder is added in the paper's
non-thinking rendering: Qwen3.5-0.8B, 2B and 4B on the laptop, Qwen3.5-9B and 27B on the pod.

- Weights at the revisions recorded in `results/pod_qwen35/<model>__nothink.npz` (`meta.revision`),
  bfloat16, text decoder loaded alone through `AutoModelForCausalLM` as in the paper's extraction.
- Rendering: `enable_thinking=False` passed explicitly, date pinned to 01 Jan 2025, the same
  `build_prompts` call. The chat-template hash must equal the paper's for that rung: `273d8e0e683b`
  for 0.8B and 2B, `a4aee8afcf2e` for 4B, 9B and 27B.
- Scoring set: `sad_validated` and `sad_label` of `results/pod_qwen35/<model>__nothink.npz`.
- Parameter counts for P7c: `scripts/pod_ladders.py`, ladder `qwen3.5_nothink`.
- The reproduction gate of the original design applies per model, against the stored Qwen3.5
  arrays.
- The same P7a to P7d design and decision rules apply per ladder: P7a holds on a ladder if it holds
  on every model run on that ladder; P7b is computed within each ladder (x[model, wrapper, version]
  over that ladder's models); P7c runs on a ladder once it has at least 4 rungs; P7d holds on a
  ladder if at least 75 percent of that ladder's models pass. No verdict pools the two ladders.
- The pod run (`pod/run_post_submission.sh`) covers Qwen3-8B, 14B, 32B and Qwen3.5-9B, 27B in one
  resumable run, together with the Qwen2.5-7B rung of the companion analysis.

## Amendment 2 (2026-11-08, after the reproduction gate of the three Qwen3 laptop rungs and before any paraphrase number was computed)

**Observation.** The registered gate failed on Qwen3-1.7B (largest change of the paper's statistic
over the 36 version-0 wrappers 0.030) and Qwen3-4B (0.018); it passed on Qwen3-0.6B (0.0071). As
registered, P7 stops on those two models, and that outcome stands as the registered result.

**Diagnosis, on the original-framing rows only.** Embedding-layer rows are identical; relative row
differences grow with depth to at most 0.8 percent; every version-0 direction is within cosine
0.9988 of the stored one at every layer (median 0.99993); the mean change of the statistic is
0.0010 to 0.0013 and the 36 statistics correlate at Spearman 0.988 to 0.997. The largest change
sits at the last hidden state (Qwen3-1.7B, eval_benchmark x deploy_ide: 0.223 against 0.192 at
layer 28, with layers 0 to 18 within 0.002). This is the signature of bfloat16 arithmetic under a
different GPU, software stack and padded-batch composition (laptop RTX 5080, torch 2.11,
transformers 5.2, 36 framings per task; the paper's pod: A100, 18 framings per task), not of a
different prompt, weight or pooling. The tolerance of 0.01 on a maximum over 36 wrappers was set
without a measured floor.

**Replacement gate**, applied to every model of both ladders, the pod rungs included.
`scripts/paraphrase_noise_floor.py` re-extracts the paper's own framing prompt set (18 framings x
60 tasks, the order and batch 16 of the pod run) on the analysis GPU and records, for the 36
original wrappers, floor_batch = the largest change of the statistic between that extraction and
the paraphrase extraction's version-0 rows (same machine, batch composition only), and
floor_machine = the same between that extraction and the stored arrays. A model passes if every
version-0 direction is within cosine 0.995 of the stored one at every layer, the mean change of
the statistic over the 36 wrappers is at most 0.005, and the largest change is at most
max(0.01, 2 x floor_batch). A model that fails stops P7 on its ladder. Both the registered and the
amended verdict are recorded per model, and results computed under this amendment are reported as
such. Nothing else in the design changes.

## Note on the floor and an ungated run (2026-11-08, after the floor measurement of the Qwen3 laptop rungs, before any paraphrase number was computed)

The floor measurement of amendment 2 gives, for the 36 original wrappers (largest / mean change of
the statistic): Qwen3-0.6B floor_batch 0 / 0, floor_machine 0.0071 / 0.0010; Qwen3-1.7B
floor_batch 0 / 0, floor_machine 0.0301 / 0.0013; Qwen3-4B floor_batch 0.0183 / 0.0007,
floor_machine 0.0073 / 0.0008. On 0.6B and 1.7B the laptop extraction does not depend on batch
composition at all: the paraphrase extraction's version-0 rows equal, bit for bit in the statistic,
a laptop re-extraction in the paper's own prompt order and batch. The whole Qwen3-1.7B discrepancy
is therefore the change of machine and software stack, which amendment 2 does not admit (its
tolerance uses floor_batch only). Under amendment 2, Qwen3-1.7B fails and P7 stops on the Qwen3
ladder; that is the registered result for this ladder and it is not amended again.

The same code is also run with the gate disabled (`scripts/paraphrase_analysis.py --ungated`,
output `paraphrase_analysis_ungated.json`), with the registered decision rules evaluated unchanged.
Its numbers are descriptive and are not registered results; any use of them must state that the
registered reproduction gate failed on the change of machine (largest change 0.030 of the paper's
statistic, mean 0.0013, directions within cosine 0.9988).
