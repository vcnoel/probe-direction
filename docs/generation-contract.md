# Generation termination and scoring contract

Generated text is not automatically a completed answer. This module separates how a sequence stopped, whether it is valid for scoring, where its final answer lives, and which generated tokens have aligned activations.

## Terminal IDs

`normalize_terminal_ids(model.generation_config, tokenizer)` accepts scalar, list, tuple, or absent `eos_token_id` values. IDs declared by the model generation configuration keep their order and are never collapsed to the tokenizer's scalar EOS. Tokenizer EOS IDs are appended only when they add a compatible fallback.

For Qwen3-0.6B, the tested configuration is `[151645, 151643]`: `151645` is the primary EOS and `151643` is a secondary terminal. A caller must use both.

## Structured status

`assess_generation(...)` returns one `GenerationAssessment` containing:

- the normalized terminal set;
- the repetition decision and its bounded evidence;
- a mutually exclusive reason: `eos`, `secondary_eos`, `repetition_loop`, `max_tokens`, or `error`;
- the terminal token and generated index when one exists;
- `complete`, `degenerate`, and `scoring_eligible` booleans;
- flat CSV/JSON-safe provenance from `provenance_dict()`.

A repetition loop is always `complete=false`, `degenerate=true`, and `scoring_eligible=false`. A close marker or partial answer inside the trace does not override that rule. Raw token IDs and provenance remain available for audit.

## Repetition detection

`detect_repetition` and `RepetitionDetector` use two bounded checks:

1. exact repeated suffixes over candidate periods of at most 32 tokens;
2. repeated 4-gram coverage in the last 64 tokens.

The defaults wait for 48 generated tokens, require at least 48 repeated tokens, require four exact suffix copies, and require 0.90 repeated 4-gram coverage for the rolling fallback. These guards intentionally leave short refrains, four-copy short choruses, ordinary enumerations, code-like output, and high-frequency punctuation unflagged in the regression suite.

`SingleSequenceRepetitionStoppingCriteria` adapts the same detector to a Transformers `StoppingCriteriaList` without importing Transformers into the core package. It rejects batches larger than one rather than stopping unrelated rows when one row loops. Multi-row generators should maintain one detector and done mask per unpadded row.

Online stopping changes output and must be opt-in. Post-hoc classification is always safe to add because it does not change sampling, terminal handling, or generated tokens.

## Final-answer spans

`parse_final_span(...)` locates the first configured single- or multi-token reasoning close marker. The final span ends before the first declared terminal or at the observed stop. The parse state distinguishes:

- EOS before the close marker;
- a close marker with no final tokens;
- a token cap after a partial final answer;
- a repetition loop after the close marker;
- a missing close marker;
- a valid final answer followed by a declared terminal.

Only the final state is complete and scoring-eligible. A parser may preserve the other spans for diagnosis but must not convert them into valid answers.

## Activation alignment

`GenerationRecord.from_unpadded_tokens(...)` retains the raw row, prompt length, provenance, generated nonterminal tokens, and their absolute transcript indices. Generated tokens and activation indices have equal length. Terminal tokens and right-padding are not activation targets.

For an online repetition stop, retain every generated nonterminal token through the stop and the corresponding activation captured when that token was fed. Never keep padded done-row tokens or relabel a loop suffix as a final answer.

## Regression provenance

`tests/fixtures/qwen3_repetition_loop_compact.json` is a compact derivative of the actual 4,096-token Qwen3-0.6B calibration failure. It stores the repeated token unit, source span, rollout ID, durable artifact reference, and source SHA256. The source text and omitted prefix are not copied. The regression test catches the repeated suffix before the original cap while legitimate repetitive controls remain negative.

The fixture proves a software property only. It does not repair, rerun, reinterpret, or validate the historical evaluation-awareness experiment.

## Legacy integration

The earlier generation, causal-steering and sandbagging paths (not part of this release) appended termination provenance after `model.generate`. Their decoding arguments remain unchanged. The sampled generation path alone exposes `--stop-repetition`; causal steering, benchmark scoring, and verbalized-judge paths use post-hoc classification so the classifier does not silently alter the historical generation distribution.

The verbalized baseline keeps its historical eight-token judge cap but exposes `judge_max_new_tokens` for new runs. If no response reaches a declared terminal, the baseline warns and reports zero eligible rows instead of scoring capped text; callers should raise the cap and record it.

New scoring excludes rows whose termination assessment is ineligible. Existing CSVs that lack termination metadata remain historical records, not certified completions. Audit cap hits and repetition by condition before reusing their scores.
