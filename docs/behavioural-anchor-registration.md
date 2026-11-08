# Behavioural anchor: SAD stages-oversight against the probe statistics

Registered 2026-11-08, before any behavioural output exists, the CPU smoke test included. The author
decided on 8 November 2026 to fund this run. It is reported whatever it shows. It is a
post-submission analysis and changes no reported number of the paper.

## Question

Do the probe statistics of the paper track a published behavioural measure of evaluation awareness
across models, on the same items, and does any such tracking exceed what geometry-matched
content-free directions show?

## Protocol: the authors' code and data

- SAD (Laine et al. 2024), `github.com/LRudL/sad` at `dfc5c9831a9bcc5c9a9dbdcaa2955aae983cd1d3`,
  licence CC BY 4.0, public and not gated. Item files are password-protected zips opened by the
  authors' `unzip.sh`. Their terms apply: question and answer text never appear in plain text
  anywhere scrapable.
- evalugator, the SAD authors' harness, `github.com/LRudL/evalugator` at
  `1787ab88cf2e4cdf79d054087b2814cc55654ec2`.
- Task `stages_oversight`: the 400 items of the paper's scoring set (200 testing, 200 deployment).
  Variants `plain` and `sp_large`, with the authors' templates. Those set temperature 0 and at most
  10 new tokens, shuffle the option order with the eval's fixed seed 42, and render the question
  above or below the text. Scoring is the authors': `mcquestion_to_text` parsing, accuracy =
  correct / trials, unparseable answers count as not correct and are reported as invalid.
- The only additions are the ones SAD's README prescribes for models it does not host:
  - an evalugator provider, `pod/sad_provider/hf_local_provider.py`, placed in the clone's
    `providers/` directory;
  - one `models.yaml` entry per model, configured as the SAD authors configured the open Llama chat
    models (`type: chat`, `assistant_answer_prefix: true`), so evalugator appends its own assistant
    message "Answer: (".
- The provider renders the request's messages with the model's chat template. Qwen3 and Qwen3.5 get
  `enable_thinking=False`, and the date is pinned as in the probe pipeline. A trailing assistant
  message is appended as a prefill. Decoding is greedy within the request's `max_tokens`, on bf16
  weights at the probe arrays' revisions; nothing is quantised. Items, templates, parsing and
  scoring are untouched.
- Environment: SAD and evalugator are installed without their pinned dependencies, on top of the
  extraction pins (torch 2.11.0, transformers 5.8.0, numpy 2.4.4). Their pins (scipy 1.11.4,
  tokenizers 0.15.0) clash with that stack. The environment is import-checked before use.

## Models

The 19 non-thinking probe models, at the revisions of the stored probe arrays:

| ladder | models |
|---|---|
| Qwen3, `enable_thinking=False` | 0.6B, 1.7B, 4B, 8B, 14B, 32B |
| Qwen3.5, `enable_thinking=False` | 0.8B, 2B, 4B, 9B, 27B |
| OLMo-2 Instruct | 0425-1B, 1124-7B, 1124-13B, 0325-32B |
| Qwen2.5 Instruct | 0.5B, 1.5B, 3B, 7B |

Thinking renderings are excluded: an answer under them requires generated reasoning, which is not
comparable.

## Measures

- **B(m)**, the behavioural measure: `stages_oversight` accuracy on `sp_large` (primary). `plain` is
  reported beside it, with its own preconditions.
- **F(m)**: mean over the 36 wrappers of the paper's folded statistic. Source:
  `results/post_submission/signed_statistic.json` (`fold`), computed from
  `results/post_submission/controlled/<ladder>__<model>.npz` (`sadproj`) on the same 400 items.
- **S(m)**: mean over the 36 wrappers of the signed band statistic, the mean over relative depth
  0.25 to 0.75 of (AUROC - 0.5). Same file, field `signed_band`.
- **O(m)**: Amendment 1's real orientation over the band. Source:
  `results/post_submission/controlled_transfer.json`, `amendment_1.a1.per_model`.

## The comparison

Pooled Spearman correlation rho across the 19 models between B and each of F, S and O.

- **Null:** permutation of B among the models within each ladder, 100,000 draws, seed 0. One-sided
  p for rho > 0.
- **Interval:** 95 percent percentile bootstrap over the 400 items, 2,000 draws, seed 0. Items are
  resampled jointly for B and for F and S. B uses per-item SAD outcomes, joined to the probe order
  by the SAD item id, with a sha256 of the whitespace-normalised body as a check. F and S are
  recomputed from the stored projections on each draw. O does not use the SAD items and is held
  fixed.
- **Matched null, for F:** rho between B and the 36-wrapper mean of each geometry-matched
  content-free draw (`results/post_submission/cells/matched_<ladder>.npz`, `matched_full`, draw d
  taken on every ladder, 100 draws). The real rho is placed as a mid-rank percentile among them.

## Preconditions

If any fails, the result is "uninformative", not "undercut".

1. Split-half reliability of B across items of at least 0.5. Computed as the median, over 200
   random label-stratified halves (seed 0), of the Spearman-Brown-corrected correlation across
   models between the two halves' accuracies.
2. Between-model standard deviation of B of at least 0.05 (twice the binomial standard error at
   400 items).
3. An invalid-answer rate of at most 10 percent per model. Models above it are excluded and listed;
   the comparison needs at least 15 models.

## Reading, per statistic X

- **Supports X:** rho(B, X) > 0, one-sided permutation p < 0.05, the bootstrap interval excludes 0,
  and, for F, rho above the matched null's 95th percentile.
- **Undercuts X:** the preconditions hold and rho(B, X) <= 0 with an interval that does not exclude
  0. For F, it also undercuts if rho does not exceed the matched null's 95th percentile (content-free
  directions track behaviour as well).
- **Anti-aligned X:** rho < 0 with the whole interval below 0, reported as such.

## Expectations, stated before any run

- F probably rises with size, but content-free directions rise too, so the matched null is the
  decisive part.
- S runs negative with size on Qwen3 (r = -0.88), so S may come out anti-aligned.
- O is near its ceiling (0.77-1.00) and already high at the embedding output, so expect little.
- The chance of failing the reliability or spread condition, and so of a registered
  "uninformative", is roughly 40-50 percent: published plain accuracies sit at 0.45-0.61 for every
  model, and sp_large at 0.46-0.87 with the open models of 70B and below at 0.46-0.67.

## Secondary, descriptive

1. **Per wrapper:** rho(B, F_w) across the 19 models for each of the 36 wrappers. Reported as the
   share positive and the range. It asks whether the probe "tracks behaviour" depends on the
   wrapper chosen.
2. **Same-prompt read.** On the probe prompts themselves (the paper's rendering, ending in
   "Answer:"), take the next-token log-odds of "A" against "B" after an assistant prefill "(".
   Reported per model as the AUROC of that log-odds against the testing/deployment label, unfolded
   and folded.
   - **Departure from the approved design wording:** the design called this an answer read off the
     probe prompts. Those prompts ask for "(A) or (B)" without defining the options (only 16 of the
     400 bodies contain them), so no answer can be scored as correct. The read therefore has no
     correctness and no verdict.

## Execution, cap and hygiene

- **Where:** on the pod only; the laptop GPU is not used. The five P7 models (Qwen3-8B, 14B, 32B,
  Qwen3.5-9B, 27B) run in P7's weight-load cycle; the other 14 run in a second loop. Every model
  step has its own timeout.
- **Cap:** 3 A100-hours for this step. The combined watchdog for P7 plus this step is 6 h.
- **Smoke test:** a CPU-only run on the laptop after this commit: Qwen3-0.6B, 10 items, `plain`.
  Its outputs are labelled smoke and never used in the analysis.
- **Raw results:** evalugator's raw results contain item text. They stay in the SAD clone, outside
  every repository, and are never uploaded.
- **Sanitised outputs:** `results/post_submission/behaviour/` holds, per item, only the SAD id, the
  body hash, the label, the score and whether the answer parsed.
- **Leak check:** `scripts/check_no_sad_text.py` fails the step if any 12-word sequence of any item
  appears in `results/` or in any tracked file.
- **Analysis code:** `scripts/behavioural_anchor.py` implements the comparison above. It is
  committed before the pod outputs are opened.
