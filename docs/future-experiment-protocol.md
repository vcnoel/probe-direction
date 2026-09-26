# Future experiment protocol

**Status: planned and not run.** This document defines the artifacts and decision rules for a future experiment. It records no outcomes.

## 1. Variables and factor schema

Keep environment, recognition, and behavior separate:

- **E:** independently assigned environment-factor vector. Candidate factors include evaluative wording, oversight statement, deployment wording, template formality, preamble length, absolute task position, filler amount, chat-template family, and task domain.
- **R:** recognition label from a preregistered gold endpoint or a judge validated against a blinded human-labeled set. R is nullable until such an endpoint exists.
- **P:** behavioral outcome or distribution from repeated generations and a preregistered scorer. P is nullable for prompt-only rows.

Each prompt-manifest row must include `row_id`, stable `source_task_id`, `template_family`, typed `environment_factors`, `condition_valence`, content hash, rendered-text hash, task-token hash, tokenizer revision, task-span bounds, split, and optional R/P fields. E factors must be crossed rather than encoded in one opaque condition label. Preserve byte-identical task content where the contrast requires it, and record failures rather than silently dropping cells.

## 2. Split construction

Assign splits before feature extraction. Group by both `source_task_id` and `template_family`; no source task, paraphrase family, or rendered duplicate may cross into confirmation. Use training data for feature engineering and model fitting, validation data for choices and sign-of-life decisions, test data for locked pipeline evaluation, and a sealed **confirmation** split for the final claim attempt.

The confirmation set must contain held-out task groups and held-out template families. Hash the split manifest and validate it locally before inference. If group/template isolation cannot be established, stop.

## 3. Endpoints

### Recognition endpoint

Prefer gold recognition labels when the task admits them. Otherwise validate a judge against blinded human labels before using it: define rubric, adjudication, disagreement handling, calibration set, target agreement measure, and an acceptance threshold in the protocol configuration. Freeze judge model/revision, prompt, decoding, and parser. An unvalidated judge may be exploratory only and cannot support an R claim.

### Behavioral endpoint

Use gold task scoring wherever possible. For open-ended behavior, use a separately validated judge and retain raw outputs. Estimate P with repeated samples at fixed decoding settings and seeds recorded per sample. Do not infer P from prompt-only activations.

## 4. Nested fitting and feature construction

All learned operations occur inside the outer grouped split:

1. fit normalization, dimensionality reduction, feature selection, direction estimation, layer selection, thresholds, and classifier hyperparameters on outer-training groups only;
2. choose among candidates with grouped inner validation;
3. refit the selected pipeline on the permitted outer data;
4. evaluate once on the outer-held-out group.

Confirmation remains sealed until the pipeline, contrasts, endpoints, and stop rules are frozen. Spectral analyses declare graph-only, fixed-reference-signal, or joint graph-and-signal mode. Basis-dependent quantities use the symmetric-normalized operator. Random-walk basis requests are rejected.

## 5. Complete-pipeline permutations

Permutation inference must rerun the complete fitted pipeline, not only reshuffle predictions after selection. Permute E labels within exchangeability blocks that preserve source-task pairing and any repeated-sample structure. For each permutation, repeat inner selection, preprocessing, fitting, calibration, and aggregation. Record the permutation map and seed. If exchangeability is not defensible, use a randomization test tied to the actual assignment mechanism or report descriptive estimates only.

## 6. Primary paired selectivity

Define a target E contrast and a matched non-target nuisance contrast before fitting. Compute a paired selectivity statistic from the same held-out source-task units. For a discrimination endpoint this may be target performance minus nuisance-control performance, but inference must preserve the pairing.

Make direction and eigenvector comparisons orientation invariant: use absolute cosine, squared projection, principal angle, projector distance, or a sign chosen exclusively from training data. Never orient a confirmation direction using confirmation labels. Report both components of selectivity, not only their difference.

## 7. Controls

At minimum include:

- token count, task-span length, absolute position, and tokenizer revision;
- lexical/template-only and condition-valence baselines;
- non-evaluative wrappers matched for length, position, and formality;
- residual norm, ordinary DC/AC, and degree-weighted null/perpendicular summaries;
- graph-only, fixed-reference-signal, and joint graph-and-signal comparisons;
- value-weighted or attention-output controls when making routing claims;
- random-label and shuffled-pair controls;
- model-family and task-family blocking;
- random, placebo, off-target-layer, and matched-norm intervention controls;
- generation-quality and scorer-stability controls for P.

A target contrast that is decodable while its negative control is equally decodable does not pass selectivity.

## 8. Disjoint transfer

Transfer tests must be disjoint along the dimension named in the claim. Template transfer holds out template families; task transfer holds out source-task families; model transfer holds out model families and training lineages; endpoint transfer holds out scorer or task families. Do not call a split “transfer” when near-duplicate prompts, shared templates, or checkpoint siblings span train and test. Fit depth resampling or feature alignment without target labels.

## 9. Hierarchical scaling evidence

Treat checkpoints as nested within model families and tasks/templates as crossed sampling units. Estimate scale effects with a hierarchical model or a prespecified family-stratified analysis that separates parameter scale from family, architecture, data, and post-training lineage. Require multiple sizes within multiple families and replication over multiple task families before using “scaling relationship” language. A monotone checkpoint table is a candidate observation, not a law. Report family-specific estimates, heterogeneity, and leave-one-family-out sensitivity.

## 10. Causal interventions

Only begin intervention work after a held-out observational sign of life and endpoint validation. Derive steering, patching, ablation, or interchange targets on training data. Test both predicted directions where meaningful, use a dose range selected without confirmation outcomes, and compare random, placebo, off-target, and matched-norm controls. Retain fluency/task-performance measurements.

For mediation, require that E changes the candidate mediator, the intervention changes the mediator as intended, and the preregistered R or P endpoint changes specifically relative to controls. Distinguish sufficiency, necessity, and statistical mediation. None alone establishes intent.

## 11. Gates and stopping rules

Proceed in order:

1. **Manifest gate:** prompt, split, config, model/tokenizer revisions, and endpoint definitions exist and validate. Otherwise stop before inference.
2. **Endpoint gate:** gold scorer exists or the judge meets its frozen validation threshold. Otherwise R/P analysis stops.
3. **Data-integrity gate:** task-token identity, crossed-cell completeness, and group/template isolation pass. Otherwise stop and rebuild manifests.
4. **Sign-of-life gate:** on validation data only, the locked target statistic exceeds its preregistered minimum and the key nuisance control does not. If it fails, stop expensive scaling and causal work; report an exploratory null without opening confirmation.
5. **Confirmation gate:** the frozen complete pipeline passes its randomization criterion and uncertainty requirement on confirmation. If it fails, do not retune on confirmation.
6. **Causal gate:** observational confirmation and endpoint validity pass before intervention. If specificity, dose response, or quality controls fail, do not claim mediation.
7. **Release gate:** every claim maps to a validated result manifest. Missing artifacts or checksums stop release.

Threshold values, multiplicity family, sample-size rationale, and maximum retries belong in the hashed configuration written before execution; this pre-results repository does not choose them post hoc.

## 12. Artifact manifests

Each run must produce local, checksummed artifacts:

- protocol configuration and hash;
- prompt manifest and split manifest;
- assignment/permutation manifest;
- exact model, tokenizer, dataset, adapter, and scorer revisions;
- environment lock or package inventory, hardware class, seeds, and command arguments;
- raw endpoint outputs and immutable activation/graph references where permitted;
- fitted-pipeline metadata for every outer fold;
- metric tables with unit of analysis and comparison mode;
- intervention target and dose manifests;
- failure/drop ledger with reasons;
- result manifest linking every artifact by checksum;
- claim ledger marking claims as planned, conjecture, validated, falsified, or reported-unverified.

Write artifacts atomically into a directory reserved by configuration hash. Validation and claim audit must succeed offline and without inference. Raw model outputs and activations need not be committed to Git, but a released claim requires a durable, reviewable manifest whose referenced artifacts are available to the reviewer.
