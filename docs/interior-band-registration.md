# Matched null under the registered interior band

Registered 2026-11-08, before any quantity below was computed.

## Why

Observation made on 2026-11-08 before this registration, from the stored scoring-set projections of the
full-task directions (`results/post_submission/controlled/*.npz`, key `sadproj`): the statistic used
throughout the paper, the folded maximum over ALL layers of |AUROC - 0.5|, attains its maximum in the first
10 percent of depth for 31 to 81 percent of wrappers on Qwen3, 56 to 75 percent on Qwen3.5 and 64 to 94
percent on OLMo-2 (rising with size on Qwen3 up to 8B and on OLMo-2), and in the last layers on Qwen2.5
(median relative depth 0.52 to 1.0). These are boundary layers, where the paper's own rule S2
(`docs/preregistered-analysis-rules.md`) says a maximum is not interpretable, and the first layers are
lexical. The same rules registered, in August 2026, the interior band (relative depth 0.25 to 0.75) as the
candidate layer set (L1). The matched-null audit of October used all layers.

## Analysis

Rerun `scripts/ladder_audit_cells.py` and `scripts/matched_null_cells.py` with the statistic restricted to
layers at relative depth 0.25 to 0.75 (layer index / (number of layers - 1)), everything else unchanged:
same ladders, directions, nulls, draws, seeds, item blocks and split halves. Then `scripts/matched_null.py`
with its verdict rules unchanged. Outputs go to `results/post_submission/cells_band/` and
`results/post_submission/matched_null_band.json`; the all-layer files are not touched.

## Reported

For every ladder, claims (a) to (d) of `matched_null.py` under the band next to the all-layer verdicts,
and the paper's headline quantities under the band: per-wrapper sign counts, r of the 36-wrapper average,
model and model-by-wrapper shares, E rho^2 and split-half Spearman-Brown.

## Decision

A claim is said to hold about evaluation content on a ladder only where `matched_null.py`'s unchanged rule
says the real directions are distinguishable from the geometry-matched content-free nulls under the band.
Both the all-layer and the band results are reported, whatever they show.
